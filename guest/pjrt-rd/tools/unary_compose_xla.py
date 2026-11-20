"""Measures whether XLA CPU's tanh, erf, logistic and power equal a composition of operations pjrt-rd already runs
bit for bit (add, multiply, divide, exp, sqrt, maximum, minimum), each taken as its own jitted call so nothing fuses.

  pixi run --manifest-path <stablehlo-gen>/pixi.toml python unary_compose_xla.py

Prints, for each candidate, how many of the inputs differ from XLA CPU and by how many ulp at most. The polynomials are
XLA's own (xla/codegen/intrinsic/tanh.cc and erf.cc at revision d867969800d4); XLA evaluates them with fused
multiply-adds, which a chain of separate multiply and add kernels cannot reproduce.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf as jerf

J = lambda f: jax.jit(f)
mul, add, div, exp, sqrt = J(lambda a, b: a * b), J(lambda a, b: a + b), J(lambda a, b: a / b), J(jnp.exp), J(jnp.sqrt)
mx, mn, neg = J(jnp.maximum), J(jnp.minimum), J(lambda a: -a)
f32 = np.float32


def inputs():
    rng = np.random.default_rng(20261120)
    bits = rng.integers(0, 1 << 32, 200000, dtype=np.uint64).astype(np.uint32).view(np.float32)
    parts = [bits, rng.uniform(-10, 10, 400000), rng.uniform(-1, 1, 200000), rng.uniform(-4e-4, 4e-4, 20000), rng.uniform(-25, 25, 100000),
             np.array([0.0, -0.0, 1e-45, -1e-45, 1e-39, np.inf, -np.inf, np.nan, 4e-4, 7.9988117, 7.9053111, 20.0, 3.8325069, 88.7, -88.7], f32)]
    return np.concatenate(parts).astype(f32)


def ulp(a, b):
    def o(v):
        w = v.view(np.int32).astype(np.int64)
        return np.where(w < 0, -(w & 0x7fffffff), w)
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(o(a) - o(b))
    d[both_nan] = 0
    return int((d != 0).sum()), int(d.max())


def horner(x2, coeffs):
    acc = np.full_like(x2, coeffs[0])
    for c in coeffs[1:]:
        acc = np.asarray(add(mul(x2, acc), np.full_like(x2, f32(c))))
    return acc


def tanh_unfused(x, clamp):
    xc = np.asarray(mn(mx(x, np.full_like(x, f32(-clamp))), np.full_like(x, f32(clamp))))
    x2 = np.asarray(mul(xc, xc))
    num = np.asarray(mul(xc, horner(x2, [-2.76076847742355e-16, 2.00018790482477e-13, -8.60467152213735e-11, 5.12229709037114e-08, 1.48572235717979e-05, 6.37261928875436e-04, 4.89352455891786e-03])))
    den = horner(x2, [1.19825839466702e-06, 1.18534705686654e-04, 2.26843463243900e-03, 4.89352518554385e-03])
    r = np.asarray(div(num, den))
    r = np.where(np.abs(x) < f32(0.0004), x, r)
    return np.where(np.abs(x) >= 20.0, np.copysign(f32(1), x), r).astype(f32)


def erf_unfused(x):
    x2 = np.asarray(mul(x, x))
    p = np.asarray(mul(x, horner(x2, [0.00022905065861350646, 0.0034082910107109506, 0.050955695062380861, 0.18520832239976145, 1.128379143519084])))
    q = horner(x2, [-1.1791602954361697e-7, 0.000023547966471313185, 0.0010179625278914885, 0.014070470171167667, 0.11098505178285362, 0.49746925110067538, 1.0])
    r = np.asarray(div(p, q))
    return np.where(np.abs(x) >= f32(3.832506856900711), np.copysign(f32(1), x), r).astype(f32)


def main():
    x = inputs()
    one = np.ones_like(x)
    print("jax", jax.__version__, "inputs", x.size)
    ref = np.asarray(J(jax.nn.sigmoid)(x))
    print("logistic vs 1/(1+exp(-x))            differ=%d max_ulp=%d" % ulp(ref, np.asarray(div(one, add(one, exp(neg(x)))))))
    ref = np.asarray(J(jnp.tanh)(x))
    for clamp in (7.99881172180175781, 7.90531110763549805):
        print("tanh vs unfused rational, clamp %.4f differ=%d max_ulp=%d" % ((clamp,) + ulp(ref, tanh_unfused(x, clamp))))
    print("tanh vs numpy tanh                   differ=%d max_ulp=%d" % ulp(ref, np.tanh(x).astype(f32)))
    ref = np.asarray(J(jerf)(x))
    print("erf vs unfused rational              differ=%d max_ulp=%d" % ulp(ref, erf_unfused(x)))
    pos = np.abs(x)
    for e, name, comp in ((2.0, "x*x", lambda v: mul(v, v)), (3.0, "x*x*x", lambda v: mul(mul(v, v), v)), (0.5, "sqrt(x)", sqrt),
                          (-1.0, "1/x", lambda v: div(np.ones_like(v), v)), (1.0, "x", lambda v: v), (-0.5, "1/sqrt(x)", lambda v: div(np.ones_like(v), sqrt(v))),
                          (4.0, "(x*x)*(x*x)", lambda v: (lambda s: mul(s, s))(mul(v, v)))):
        for label, v in (("any sign", x), ("positive", pos)):
            ref = np.asarray(J(lambda a: a ** e)(v))
            print("power %-5s vs %-12s %s differ=%d max_ulp=%d" % ((e, name, label) + ulp(ref, np.asarray(comp(v)))))
    print("erf module text:", [l.strip() for l in jax.jit(jerf).lower(jax.ShapeDtypeStruct((4,), f32)).as_text().split("\n") if "erf" in l])
    print("gelu module text:", [l.strip().split(" : ")[0] for l in jax.jit(lambda a: jax.nn.gelu(a, approximate=False)).lower(jax.ShapeDtypeStruct((4,), f32)).as_text().split("\n") if "%" in l and "=" in l])
    print("pow2 module text:", [l.strip().split(" : ")[0] for l in jax.jit(lambda a: a ** 2.0).lower(jax.ShapeDtypeStruct((4,), f32)).as_text().split("\n") if "%" in l and "=" in l])
    print("pow int text:", [l.strip().split(" : ")[0] for l in jax.jit(lambda a: a ** 3).lower(jax.ShapeDtypeStruct((4,), f32)).as_text().split("\n") if "%" in l and "=" in l])


if __name__ == "__main__":
    main()
