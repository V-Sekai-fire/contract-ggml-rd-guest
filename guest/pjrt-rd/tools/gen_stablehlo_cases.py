"""Writes guest/ggml_test/stablehlo_cases.inc: real StableHLO modules from JAX, their inputs, the
output XLA's CPU backend gives for them, and a mutated module per case that must not give that output.

  pixi run --manifest-path <stablehlo-gen>/pixi.toml python gen_stablehlo_cases.py <out.inc>

The references of sin_stress, cos_stress and rope_neox are XLA CPU calling the C runtime's sinf and cosf, and
the lowering matches the Windows runtime (UCRT). On another platform their bytes are kept from the <out.inc>
that is being replaced (KEEP_REFERENCE); it is an error if that file does not have them.
"""
import platform
import re
import sys

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.scipy.special import erf as jax_erf

BF16 = ml_dtypes.bfloat16

# Cases whose reference is the Windows C runtime's sinf and cosf (tests/xla_sincos): generated on Windows only.
KEEP_REFERENCE = ("sin_stress", "cos_stress", "rope_neox")

# Cases whose output is not bit-equal to XLA CPU's, each with what was measured. The probe still holds
# them to the functional bar and fails if one turns bit-equal, so the list can only shrink.
XFAIL = {
    "dot_batched": "XLA CPU's dot accumulates in a shape-dependent tree (tools/reveal_dot_order.py), which ggml-rd's mul_mat does not reproduce for this shape",
    "reduce_sum_random": "XLA CPU's float summation order is an irregular tree that changes with the shape, the signature of LLVM reassociating an unrolled chain of adds: tools/reveal_sum_order.py gives (((5+(3+4))+(2+(0+1)))+((10+(9+(8+(6+7))))+((14+15)+(13+(11+12))))) for 8 rows of 16 and a different tree for 32, 48 and 64 elements, so no closed-form order reproduces it and only a per-shape table could; ggml-rd sums 256 interleaved lanes",
    "mean_last": "a float sum whose XLA CPU order is not reproduced (see reduce_sum_random), then a divide",
    "softmax_f32": "contains a float sum whose XLA CPU order is not reproduced (see reduce_sum_random); softmax_pinned, the same softmax with the sum order written out, is bit-equal",
    "rms_norm_f32": "contains a float sum whose XLA CPU order is not reproduced (see reduce_sum_random); pinned_rms_scale shows the sum and sqrt part is bit-equal when the order is written out",
    "mlp_pre": "XLA CPU's dot (a YNNPACK fusion) accumulates in a shape-dependent tree: for 4x16 @ 16x8 it is the same tree as a float reduce of 16, and a sequential fused multiply-add chain matches 14 of 32 elements; the bias add after it is exact (dot then add equals XLA in 32 of 32)",
    "mlp_f32": "its dot is not bit-equal (see mlp_pre); the exp after it is exact",
    "causal_attention": "its two dot_generals accumulate in XLA CPU's dot tree (see dot_batched) and its softmax sum in XLA's reduce tree (see reduce_sum_random); the mask, maximum, exp and divide around them are bit-equal (causal_softmax_pinned), and so are dots with two free or two batch dimensions when every partial sum is exact (dot_two_free_exact, dot_two_batch_exact)",
}


def ints(rng, shape, lo=-4, hi=5):
    return rng.integers(lo, hi, size=shape).astype(np.float32)


def rand(rng, shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def positive(rng, shape):
    return (rng.random(shape) * 3.0 + 0.25).astype(np.float32)


def tree_sum(a):
    """Sum the last axis in a pinned order: lane i plus lane i + n/2, halving until one is left.

    The adds are explicit slices added together. A chain of reduces would not hold: XLA's
    simplifier merges it into one reduce-window and the order changes, because StableHLO leaves
    the order of a reduce unspecified. An add of two slices it keeps as written.
    """
    while a.shape[-1] > 1:
        h = a.shape[-1] // 2
        a = a[..., :h] + a[..., h:]
    return a[..., 0]


def cases():
    rng = np.random.default_rng(20261031)
    out = []

    def add(name, fn, args, mutate, note):
        out.append((name, fn, args, mutate, note))

    x, y = rand(rng, (8, 16)), rand(rng, (8, 16))
    add("add_f32", lambda a, b: a + b, [x, y], ("stablehlo.add", "stablehlo.subtract"), "elementwise")
    add("subtract_f32", lambda a, b: a - b, [x, y], ("stablehlo.subtract", "stablehlo.add"), "elementwise")
    add("multiply_f32", lambda a, b: a * b, [x, y], ("stablehlo.multiply", "stablehlo.add"), "elementwise")
    add("divide_f32", lambda a, b: a / b, [x, positive(rng, (8, 16))], ("stablehlo.divide", "stablehlo.multiply"), "elementwise")
    add("negate_f32", lambda a: -a, [x], ("stablehlo.negate", "stablehlo.abs"), "elementwise")
    add("exp_f32", lambda a: jnp.exp(a), [x], ("stablehlo.exponential", "stablehlo.negate"), "transcendental")
    add("log_f32", lambda a: jnp.log(a), [positive(rng, (8, 16))], ("stablehlo.log", "stablehlo.negate"), "transcendental")
    # exp and log are XLA's own polynomials (polynomial_approximations.cc); these hold every input class the
    # model was checked on over all 2^32 patterns: the whole range, both clamp edges, the subnormal
    # boundary, zeros, subnormals, infinities and a NaN.
    srng = np.random.default_rng(20261031)
    bits = lambda n: srng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
    exp_x = np.concatenate([bits(1400), srng.uniform(-90, 90, 1400), srng.uniform(-88.0, -86.5, 600), srng.uniform(87.5, 89.5, 600),
                            np.array([0.0, -0.0, 1e-45, -1e-45, 1e-39, np.inf, -np.inf, np.nan, 88.72283, 88.722839, -87.33654, -103.97], np.float32)]).astype(np.float32)
    exp_x = np.concatenate([exp_x, np.zeros(4096 - len(exp_x), np.float32)])
    add("exp_stress", lambda a: jnp.exp(a), [exp_x], ("stablehlo.exponential", "stablehlo.negate"), "XLA's exp polynomial over every input class")
    log_x = np.concatenate([bits(1800), srng.uniform(0, 200, 1200), srng.uniform(0.5, 2.0, 800), np.float32(2.0) ** srng.integers(-126, 127, 200).astype(np.float32),
                            np.array([0.0, -0.0, 1e-45, -1e-45, 1e-39, np.inf, -np.inf, np.nan, 1.0, 0.70710677, 0.70710683, -1.0], np.float32)]).astype(np.float32)
    log_x = np.concatenate([log_x, np.ones(4096 - len(log_x), np.float32)])
    add("log_stress", lambda a: jnp.log(a), [log_x], ("stablehlo.log", "stablehlo.negate"), "XLA's log polynomial over every input class")
    rs_x = np.concatenate([bits(1500), srng.uniform(0, 100, 1200), srng.uniform(1e-3, 4.0, 900), np.float32(2.0) ** srng.integers(-126, 127, 300).astype(np.float32),
                           np.array([0.0, -0.0, 1e-45, -1e-45, 1e-39, 1.1754944e-38, np.inf, -np.inf, np.nan, 1.0, 4.0, -1.0], np.float32)]).astype(np.float32)
    rs_x = np.concatenate([rs_x, np.ones(4096 - len(rs_x), np.float32)])
    sc_x = np.concatenate([srng.uniform(-10, 10, 800), srng.uniform(-1000, 1000, 400), srng.uniform(-1, 1, 400), bits(1200),
                           srng.uniform(1.0e7, 1.0e10, 500), srng.uniform(-1.0e10, -1.0e7, 200),
                           np.array([0.0, -0.0, 1e-45, 1e-5, 3.14159274, 1.57079637, -1.57079637, 6.28318548, 0.78539819, 0.785398185,
                                     16777302.0, 16777216.0, 1e6, 3370000000.0, 3.3e9, 3.4e9, 1e30, 3.4028235e38, np.inf, -np.inf, np.nan, 1e-30], np.float32)]).astype(np.float32)
    sc_x = np.concatenate([sc_x, srng.uniform(-100, 100, 4096 - len(sc_x))]).astype(np.float32)[:4096]
    add("sin_stress", lambda a: jnp.sin(a), [sc_x], ("stablehlo.sine", "stablehlo.negate"), "XLA CPU's sin (the C runtime's sinf) over every range and edge input")
    add("cos_stress", lambda a: jnp.cos(a), [sc_x], ("stablehlo.cosine", "stablehlo.negate"), "XLA CPU's cos (the C runtime's cosf) over every range and edge input")
    def rope_neox(x, pos, inv):
        h = x.shape[-1] // 2
        ang = pos[:, None] * inv[None, :]
        c, sn = jnp.cos(ang)[:, None, :], jnp.sin(ang)[:, None, :]
        x1, x2 = x[..., :h], x[..., h:]
        return jnp.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], axis=-1)
    rope_inv = (1.0 / (10000.0 ** (np.arange(0, 8, dtype=np.float64) * 2.0 / 16.0))).astype(np.float32)
    add("rope_neox", rope_neox, [rand(rng, (8, 4, 16), 1.0), np.arange(8, dtype=np.float32) * 3.0 + 0.5, rope_inv],
        ("stablehlo.cosine", "stablehlo.negate"), "rotary embedding, rotate-halves layout, frequency table as an input")
    add("rsqrt_stress", lambda a: jax.lax.rsqrt(a), [rs_x], ("stablehlo.rsqrt", "stablehlo.negate"), "XLA CPU's rsqrt on a Zen 2 over every input class")
    add("sqrt_f32", lambda a: jnp.sqrt(a), [positive(rng, (8, 16))], ("stablehlo.sqrt", "stablehlo.negate"), "elementwise")

    add("convert_f32_bf16", lambda a: a.astype(jnp.bfloat16), [x], ("stablehlo.convert", "stablehlo.negate"), "convert")
    xb = rand(rng, (8, 16)).astype(BF16)
    add("convert_bf16_f32", lambda a: a.astype(jnp.float32), [xb], ("stablehlo.convert", "stablehlo.negate"), "convert")
    add("add_bf16", lambda a, b: a + b, [xb, rand(rng, (8, 16)).astype(BF16)], ("stablehlo.add", "stablehlo.subtract"), "elementwise")

    add("broadcast_scalar_add", lambda a: a + 1.5, [x], ("stablehlo.add", "stablehlo.subtract"), "broadcast")
    add("broadcast_row_add", lambda a, r: a + r[None, :], [x, rand(rng, (16,))], ("stablehlo.add", "stablehlo.subtract"), "broadcast")
    add("broadcast_col_mul", lambda a, c: a * c[:, None], [x, rand(rng, (8,))], ("stablehlo.multiply", "stablehlo.add"), "broadcast")

    add("transpose_2d", lambda a: a.T, [x], ("dims = [1, 0]", "dims = [0, 1]"), "layout")
    add("reshape_2d", lambda a: a.reshape(16, 8), [x], ("stablehlo.reshape", "stablehlo.negate"), "layout")
    t3 = rand(rng, (2, 3, 4))
    add("transpose_3d", lambda a: jnp.transpose(a, (2, 0, 1)), [t3], ("dims = [2, 0, 1]", "dims = [1, 2, 0]"), "layout")
    add("reshape_transpose", lambda a: a.reshape(4, 6).T + 1.0, [t3], ("stablehlo.add", "stablehlo.subtract"), "layout")

    def mutate_dot(t):
        a, b = "%arg0, %arg1", "%arg1, %arg0"
        return t.replace(a, "@@", 1).replace(b, a, 1).replace("@@", b, 1) if a in t else t

    sq = (ints(rng, (8, 8)), ints(rng, (8, 8)))
    add("dot_exact", lambda a, b: a @ b, list(sq), ("DOT", "DOT"), "dot, integer-valued: every partial sum is exact")
    a, b = rand(rng, (8, 24)), rand(rng, (24, 12))
    add("dot_random", lambda p, q: p @ q, [a, b], ("OPERANDS", ""), "dot, random: reduction order is unspecified")
    add("dot_rhs_t", lambda p, q: p @ q.T, [rand(rng, (8, 24)), rand(rng, (12, 24))], ("OPERANDS", ""), "dot, rhs contracted on its last dim")
    add("dot_batched", lambda p, q: jnp.einsum("bmk,bkn->bmn", p, q), [rand(rng, (3, 6, 10)), rand(rng, (3, 10, 5))], ("OPERANDS", ""), "dot, one batch dim")
    add("dot_bf16", lambda p, q: p @ q, [rand(rng, (8, 24)).astype(BF16), rand(rng, (24, 12)).astype(BF16)], ("OPERANDS", ""), "dot, bf16 operands")
    add("mlp_f32", lambda p, w, bb: jnp.exp(p @ w + bb[None, :]), [rand(rng, (4, 16), 0.5), rand(rng, (16, 8), 0.25), rand(rng, (8,), 0.1)],
        ("stablehlo.exponential", "stablehlo.negate"), "dot, bias broadcast, exp")

    sq8 = ints(rng, (8, 8))
    add("abs_f32", lambda a: jnp.abs(a), [x], ("stablehlo.abs", "stablehlo.negate"), "exact elementwise")
    add("floor_f32", lambda a: jnp.floor(a), [rand(rng, (8, 16), 3.0)], ("stablehlo.floor", "stablehlo.negate"), "exact elementwise")
    special = np.array([[-0.0, 0.0, -1.5, 2.5, np.inf, -np.inf, 1e-30, -1e-30], [3.0, -3.0, np.nan, -0.5, 7.25, -7.25, 1e-45, -1e-45]], np.float32)
    add("sign_f32", lambda a: jnp.sign(a), [rand(rng, (8, 16), 3.0)], ("stablehlo.sign", "stablehlo.negate"), "exact elementwise")
    add("sign_special", lambda a: jnp.sign(a), [special], ("stablehlo.sign", "stablehlo.negate"), "sign of zeros, infinities, a NaN, small values and denormals")
    add("recip_special", lambda a: 1.0 / a, [special], ("stablehlo.divide", "stablehlo.multiply"), "one over zeros, infinities and small values")
    add("slice_basic", lambda a: a[1:5, 2:6], [rand(rng, (6, 8))], ("[1:5, 2:6]", "[2:6, 1:5]"), "static slice")
    add("slice_strided", lambda a: a[::2, 1::3], [rand(rng, (6, 8))], ("[0:6:2, 1:8:3]", "[1:6:2, 1:8:3]"), "strided slice")
    add("slice_3d", lambda a: a[:, 1:3, ::2], [rand(rng, (2, 4, 6))], ("[0:2, 1:3, 0:6:2]", "[0:2, 0:2, 0:6:2]"), "slice of a rank-3 array")
    add("concat_dim1", lambda a, b: jnp.concatenate([a, b], axis=1), [rand(rng, (6, 8)), rand(rng, (6, 8))], ("%arg0, %arg1, dim = 1", "%arg1, %arg0, dim = 1"), "concatenate")
    add("concat_dim0_three", lambda a, b, c: jnp.concatenate([a, b, c], axis=0), [rand(rng, (2, 5)), rand(rng, (3, 5)), rand(rng, (1, 5))], ("%arg0, %arg1, %arg2", "%arg2, %arg1, %arg0"), "concatenate three operands")
    add("pad_edges", lambda a: jax.lax.pad(a, np.float32(0), ((1, 2, 0), (3, 0, 0))), [rand(rng, (6, 8))], ("low = [1, 3], high = [2, 0]", "low = [2, 3], high = [1, 0]"), "pad with zeros")
    add("pinned_sum_tree", lambda a: tree_sum(a) + 1.0, [rand(rng, (8, 64))], ("stablehlo.add %", "stablehlo.subtract %"), "float sum with the order pinned by explicit slice adds")
    add("pinned_mean", lambda a: tree_sum(a) / 64.0, [rand(rng, (8, 64))], ("stablehlo.divide", "stablehlo.multiply"), "mean with the order pinned")
    add("pinned_rms_scale", lambda a: jnp.sqrt(tree_sum(a * a) / 64.0 + 1e-6), [rand(rng, (8, 64))], ("stablehlo.sqrt", "stablehlo.negate"), "sqrt(mean(a*a) + eps) with the order pinned; dividing by it is rewritten to rsqrt by XLA")
    add("reduce_sum_exact", lambda a: a.sum(axis=-1), [sq8], ("dimensions = [1]", "dimensions = [0]"), "reduce add, integer-valued: every partial sum is exact")
    add("reduce_sum_first", lambda a: a.sum(axis=0), [ints(rng, (8, 8))], ("dimensions = [0]", "dimensions = [1]"), "reduce add over the leading dimension")
    cube = ints(rng, (3, 3, 3))
    add("reduce_sum_multi", lambda a: a.sum(axis=(0, 2)), [cube], ("dimensions = [0, 2]", "dimensions = [0, 1]"), "reduce add over two non-adjacent dimensions")
    add("reduce_sum_random", lambda a: a.sum(axis=-1), [rand(rng, (8, 64))], ("dimensions = [1]", "dimensions = [0]"), "reduce add, random: the summation order is unspecified")
    add("reduce_max_last", lambda a: a.max(axis=-1), [rand(rng, (8, 8))], ("dimensions = [1]", "dimensions = [0]"), "reduce maximum")
    add("reduce_max_multi", lambda a: a.max(axis=(0, 2)), [rand(rng, (3, 3, 3))], ("dimensions = [0, 2]", "dimensions = [0, 1]"), "reduce maximum over two dimensions")
    add("reduce_sum_bf16", lambda a: a.sum(axis=-1), [ints(rng, (8, 8)).astype(BF16)], ("dimensions = [1]", "dimensions = [0]"), "reduce add on bf16 operands")
    add("mean_last", lambda a: a.mean(axis=-1), [rand(rng, (8, 64))], ("stablehlo.divide", "stablehlo.multiply"), "reduce add then divide by the count")
    add("mlp_pre", lambda p, w, bb: p @ w + bb[None, :], [rand(rng, (4, 16), 0.5), rand(rng, (16, 8), 0.25), rand(rng, (8,), 0.1)], ("stablehlo.dot_general", "stablehlo.subtract"), "dot plus bias, the part of mlp_f32 before exp")
    add("softmax_pinned", lambda a: (lambda e: e / tree_sum(e)[:, None])(jnp.exp(a - jnp.max(a, axis=-1, keepdims=True))), [rand(rng, (8, 16), 2.0)], ("stablehlo.subtract", "stablehlo.add"), "softmax with the sum order pinned")
    add("divide_by_const3", lambda a: a / 3.0, [rand(rng, (8, 16), 2.0)], ("stablehlo.divide", "stablehlo.multiply"), "XLA rewrites a / broadcast(3) as a * broadcast(1/3)")
    add("divide_by_row", lambda a, b: a / b[:, None], [rand(rng, (8, 16), 2.0), positive(rng, (8,))], ("stablehlo.divide", "stablehlo.multiply"), "a / broadcast(b), b a vector")
    add("softmax_f32", lambda a: jax.nn.softmax(a, axis=-1), [rand(rng, (8, 16), 2.0)], ("stablehlo.subtract", "stablehlo.add"), "reduce maximum, exp, reduce add, divide")
    add("rms_norm_f32", lambda a: a * jax.lax.rsqrt((a * a).mean(axis=-1, keepdims=True) + 1e-6), [rand(rng, (8, 16))], ("stablehlo.rsqrt", "stablehlo.sqrt"), "reduce add, divide, rsqrt, broadcast, multiply")

    # Values known at compile time: iota, i32 arithmetic, compare, and, convert and constant i1 masks are folded
    # on the host; a select with a known predicate picks each element by a gather, which copies its bits.
    add("mask_where", lambda a: jnp.where(jnp.tril(jnp.ones((16, 16), bool)), a, -jnp.inf), [rand(rng, (16, 16))],
        ("compare GE", "compare LT"), "a causal mask: private functions, iota, compare and select")
    const_mask = np.random.default_rng(7).random((8, 16)) > 0.5
    add("select_const_mask", lambda a, b: jnp.where(const_mask, a, b), [rand(rng, (8, 16)), rand(rng, (8, 16))],
        ("%arg0, %arg1, %arg2", "%arg0, %arg2, %arg1"), "select by a constant i1 mask printed as hex bytes")
    add("iota_f32_add", lambda a: a + jnp.arange(16, dtype=jnp.float32), [rand(rng, (4, 16))], ("stablehlo.add %", "stablehlo.subtract %"), "an f32 iota folded to a constant")
    add("iota_i32_convert", lambda a: a * (jnp.arange(16) - 5).astype(jnp.float32), [rand(rng, (4, 16))], ("dense<5>", "dense<4>"), "i32 iota, subtract and convert folded at compile time")
    def causal_softmax_pinned(s):
        n = s.shape[-1]
        m = jnp.where(jnp.tril(jnp.ones((n, n), bool)), s, -jnp.inf)
        e = jnp.exp(m - jnp.max(m, axis=-1, keepdims=True))
        return e / tree_sum(e)[..., None]
    add("causal_softmax_pinned", causal_softmax_pinned, [rand(rng, (2, 16, 16), 2.0)], ("compare GE", "compare GT"), "causal mask, then softmax with the sum order pinned")
    add("dot_two_free_exact", lambda a, b: jnp.einsum("mk,knp->mnp", a, b), [ints(rng, (4, 8)), ints(rng, (8, 3, 5))], ("OPERANDS", ""), "dot_general with two free dimensions on one side, integer-valued")
    add("dot_two_batch_exact", lambda a, b: jnp.einsum("abmk,abkn->abmn", a, b), [ints(rng, (2, 3, 4, 8)), ints(rng, (2, 3, 8, 5))], ("OPERANDS", ""), "dot_general with two batch dimensions, integer-valued")
    # Values computed at run time: XLA's maximum, minimum and compares (denormals read as zero, NaN-propagating,
    # -0 below +0) run as the xla_compare kernels; a select of a run-time predicate gathers by a computed index.
    spv = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e-45, -1e-45, 1e-39, 1.17549435e-38, 2.0, -2.0, 3.4028235e38], np.float32)
    pa, pb = (np.array(t, np.float32) for t in zip(*[(x, y) for x in spv for y in spv]))
    add("relu_special", lambda a: jax.nn.relu(a), [np.concatenate([spv, rand(rng, (114,))]).astype(np.float32)], ("stablehlo.maximum", "stablehlo.minimum"), "ReLU: maximum against zero, special values included")
    add("clip_special", lambda a: jnp.clip(a, -1.0, 1.0), [np.concatenate([spv, rand(rng, (114,), 2.0)]).astype(np.float32)], ("stablehlo.minimum", "stablehlo.maximum"), "clip: maximum then minimum against constants")
    add("maximum_pairs", lambda a, b: jnp.maximum(a, b), [pa, pb], ("stablehlo.maximum", "stablehlo.minimum"), "maximum of every pair of special values")
    add("minimum_pairs", lambda a, b: jnp.minimum(a, b), [pa, pb], ("stablehlo.minimum", "stablehlo.maximum"), "minimum of every pair of special values")
    add("compare_pairs", lambda a, b: jnp.stack([a > b, a >= b, a < b, a <= b, a == b, a != b]).astype(jnp.float32), [pa, pb], ("compare GT", "compare GE"),
        "the six compares of every pair of special values, converted to f32")
    add("where_runtime", lambda a, b: jnp.where(a > b, a, b), [pa, pb], ("compare GT", "compare LT"), "select by a predicate computed at run time")
    tiny = (np.float32(2.0) ** -np.arange(60, 90, dtype=np.float32)).astype(np.float32)
    add("multiply_denormal", lambda a, b: a * b, [np.resize(tiny, 128).astype(np.float32), np.repeat(np.float32([1e-10, 3e-20, 0.5, 7e-30]), 32)], ("stablehlo.multiply", "stablehlo.add"),
        "products in the denormal range: XLA CPU flushes them to zero (FTZ)")
    add("logic_runtime", lambda a, b: jnp.where(((a > 0) & (b < 0)) | ~(a == b), a, b), [rand(rng, (8, 16)), rand(rng, (8, 16)).round()], ("stablehlo.or", "stablehlo.and"), "and, or and not of run-time predicates")
    qkv = [rand(rng, (1, 8, 2, 16)) for _ in range(3)]
    add("causal_attention", lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True), qkv, ("compare GE", "compare GT"),
        "JAX's causal dot_product_attention: calls, rank-5 values, iota, compare, and, select, dot_general with two free dimensions")
    # --- tanh, logistic, erf and power -------------------------------------------------------------------
    # logistic is 1 / (1 + exp(-x)) in XLA and here. tanh and erf are rationals XLA evaluates with fused
    # multiply-adds (tools/unary_compose_xla.py), which separate kernels cannot reproduce.
    urng = np.random.default_rng(20261120)
    ubits = lambda n: urng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
    usp = np.array([0.0, -0.0, 1e-45, -1e-45, 1e-39, np.inf, -np.inf, np.nan, 4e-4, -4e-4, 3.9e-4, 7.9988117, 7.9053111, -7.9988117, 20.0, -20.0,
                    19.99, 3.8325069, -3.8325069, 3.83, 88.7, -88.7, 100.0, -100.0, 1.0, -1.0, 0.5, 9.0], np.float32)
    def stress(*parts):
        v = np.concatenate([np.asarray(q, np.float32) for q in parts] + [usp]).astype(np.float32)
        return np.concatenate([v, urng.uniform(-3, 3, 4096 - len(v)).astype(np.float32)])[:4096]
    add("tanh_f32", lambda a: jnp.tanh(a), [rand(rng, (8, 16), 2.0)], ("stablehlo.tanh", "stablehlo.negate"), "tanh")
    add("tanh_stress", lambda a: jnp.tanh(a), [stress(ubits(900), urng.uniform(-10, 10, 1500), urng.uniform(-1e-3, 1e-3, 400), urng.uniform(-25, 25, 500))],
        ("stablehlo.tanh", "stablehlo.negate"), "tanh over every range of XLA's rational: the small-input pass-through, the clamp and the +-1 tail")
    add("logistic_f32", lambda a: jax.nn.sigmoid(a), [rand(rng, (8, 16), 3.0)], ("stablehlo.logistic", "stablehlo.negate"), "logistic")
    add("logistic_stress", lambda a: jax.nn.sigmoid(a), [stress(ubits(900), urng.uniform(-30, 30, 1500), urng.uniform(-100, 100, 600), urng.uniform(-1, 1, 300))],
        ("stablehlo.logistic", "stablehlo.negate"), "logistic over every input class: 1 / (1 + exp(-x)) with XLA's exp")
    add("silu_f32", lambda a: jax.nn.silu(a), [rand(rng, (8, 16), 3.0)], ("stablehlo.logistic", "stablehlo.negate"), "SiLU: x * logistic(x)")
    add("erf_f32", lambda a: jax_erf(a), [rand(rng, (8, 16), 1.5)], ("chlo.erf", "stablehlo.negate"), "erf (chlo.erf)")
    add("erf_stress", lambda a: jax_erf(a), [stress(ubits(900), urng.uniform(-5, 5, 1800), urng.uniform(-1e-3, 1e-3, 300), urng.uniform(3.7, 3.9, 300))],
        ("chlo.erf", "stablehlo.negate"), "erf over every range of XLA's rational and its +-1 tail")
    add("gelu_erf", lambda a: jax.nn.gelu(a, approximate=False), [rand(rng, (8, 16), 2.0)], ("chlo.erfc", "chlo.erf"), "exact GELU as JAX writes it: 0.5 * x * erfc(-x / sqrt 2)")
    add("gelu_tanh", lambda a: jax.nn.gelu(a, approximate=True), [rand(rng, (8, 16), 2.0)], ("stablehlo.tanh", "stablehlo.negate"), "tanh GELU: a cube by multiplies, tanh")
    pw = np.concatenate([usp, rand(rng, (100,), 3.0)]).astype(np.float32)
    add("power_square", lambda a: a ** 2.0, [pw], ("dense<2.000000e+00>", "dense<3.000000e+00>"), "power with a constant exponent of 2: x * x, as XLA rewrites it")
    add("power_cube", lambda a: a ** 3.0, [pw], ("dense<3.000000e+00>", "dense<2.000000e+00>"), "power with a constant exponent of 3: x * x * x")
    add("power_recip", lambda a: a ** -1.0, [pw], ("dense<-1.000000e+00>", "dense<2.000000e+00>"), "power with a constant exponent of -1: 1 / x")

    # --- i32 at run time and gather ----------------------------------------------------------------------
    # A run-time i32 is an f32 tensor of whole numbers (exact inside 2^24); an i32 argument is converted on the
    # host at Execute. A gather takes rows with ggml_get_rows, a 32-bit copy, after clamping the index as XLA does.
    table = rand(rng, (10, 4))
    idx8 = np.array([0, 9, -1, -10, 3, 12, -15, 5], np.int32)
    add("embedding_lookup", lambda t, i: t[i], [table, idx8], ("dense<10>", "dense<9>"),
        "table[i] with i32 indices at run time: the negative wrap (compare, add, select on i32) and a gather that clamps")
    add("take_clip", lambda t, i: jnp.take(t, i, axis=0, mode="clip"), [table, idx8], ("slice_sizes = array<i64: 1, 4>", "slice_sizes = array<i64: 1, 3>"),
        "jnp.take(mode='clip'): the gather alone, on an i32 argument")
    wide = rand(rng, (3, 10))
    add("take_axis1_fill", lambda t, i: jnp.take(t, i, axis=1), [wide, idx8], ("compare LE", "compare LT"),
        "jnp.take on axis 1 in its default mode: a gather, an i1 reduce and a NaN fill for the indices out of range")
    add("index_pairs", lambda t, i, j: t[i, j], [rand(rng, (5, 6)), np.array([0, 4, -1, 2], np.int32), np.array([5, 0, 3, -2], np.int32)],
        ("dense<6>", "dense<5>"), "t[i, j]: a gather by an index vector of two, as one linear row index")
    add("index_constant", lambda t: t[np.array([3, 1, 2, 9, 0])], [table], ("[3, 1, 2, 9, 0]", "[3, 1, 2, 8, 0]"), "table[constant indices]: the index folded on the host")
    ia = np.array([3, -7, 0, 11, 250, -1000, 4095, 2], np.int32)
    ib = np.array([5, 2, -3, 11, -17, 999, 1, -2], np.int32)
    add("i32_arith_runtime", lambda i, j: i * 3 - j + jnp.maximum(i, j) * jnp.minimum(i, 4) - jnp.abs(j), [ia, ib], ("stablehlo.subtract", "stablehlo.add"),
        "i32 add, subtract, multiply, maximum, minimum and abs at run time, returned as i32")
    add("i32_convert_runtime", lambda a, i: a.astype(jnp.int32).astype(jnp.float32) * 0.5 + i.astype(jnp.float32) * a + (i > 2).astype(jnp.int32).astype(jnp.float32),
        [(rand(rng, (8,), 6.0)), ia], ("compare GT", "compare LT"), "f32 to i32 (toward zero), i32 to f32 and i1 to i32 at run time")
    add("where_i32_runtime", lambda i, j: jnp.where(i > j, i, -j), [ia, ib], ("compare GT", "compare LT"), "select of i32 values by an i32 compare at run time")

    # --- convolution -------------------------------------------------------------------------------------
    # XLA CPU sums a convolution in its own order, like a dot; integer-valued operands make every partial sum exact.
    nhwc = ("NHWC", "HWIO", "NHWC")
    conv = lambda strides, padding, **kw: (lambda x, w: jax.lax.conv_general_dilated(x, w, strides, padding, dimension_numbers=nhwc, **kw))
    xi, wi = ints(rng, (1, 8, 8, 3)), ints(rng, (3, 3, 3, 5), -2, 3)
    add("conv_exact", conv((2, 2), ((1, 1), (1, 1))), [xi, wi], ("stride = [2, 2]", "stride = [2, 1]"), "3x3 convolution, stride 2, padding 1, integer-valued: every partial sum is exact")
    add("conv_random", conv((2, 2), ((1, 1), (1, 1))), [rand(rng, (1, 8, 8, 3)), rand(rng, (3, 3, 3, 5), 0.3)], ("stride = [2, 2]", "stride = [2, 1]"), "3x3 convolution, stride 2, padding 1, random")
    add("conv_asym_pad_exact", conv((1, 2), ((0, 2), (1, 0))), [xi, wi], ("pad = [[0, 2], [1, 0]]", "pad = [[2, 0], [1, 0]]"), "convolution with another padding on each edge and another stride on each axis")
    add("conv_dilated_exact", conv((1, 1), ((2, 2), (2, 2)), rhs_dilation=(2, 2)), [xi, wi], ("rhs_dilate = [2, 2]", "rhs_dilate = [2, 1]"), "convolution with a kernel dilation of 2")
    add("conv_groups_exact", conv((1, 1), ((1, 1), (1, 1)), feature_group_count=2), [ints(rng, (1, 6, 6, 4)), ints(rng, (3, 3, 2, 6), -2, 3)],
        ("feature_group_count = 2", "feature_group_count = 1"), "convolution with two feature groups")
    add("conv_depthwise_exact", conv((1, 1), ((1, 1), (1, 1)), feature_group_count=4), [ints(rng, (1, 6, 6, 4)), ints(rng, (3, 3, 1, 4), -2, 3)],
        ("stride = [1, 1]", "stride = [2, 1]"), "depthwise convolution: one feature group for each channel")
    add("conv_nchw_exact", lambda x, w: jax.lax.conv_general_dilated(x, w, (1, 1), ((1, 1), (1, 1)), dimension_numbers=("NCHW", "OIHW", "NCHW")),
        [ints(rng, (2, 3, 6, 6)), ints(rng, (5, 3, 3, 3), -2, 3)], ("pad = [[1, 1], [1, 1]]", "pad = [[1, 1], [0, 2]]"), "convolution in the NCHW and OIHW layout, batch of two")
    add("conv_1d_exact", lambda x, w: jax.lax.conv_general_dilated(x, w, (2,), ((1, 1),), dimension_numbers=("NWC", "WIO", "NWC")),
        [ints(rng, (1, 12, 3)), ints(rng, (3, 3, 4), -2, 3)], ("stride = [2]", "stride = [1]"), "convolution over one spatial dimension")
    tconv = lambda x, w: jax.lax.conv_transpose(x, w, (2, 2), "SAME", dimension_numbers=nhwc)
    add("conv_transpose_exact", tconv, [ints(rng, (1, 4, 4, 3)), ints(rng, (2, 2, 3, 5), -2, 3)], ("lhs_dilate = [2, 2]", "lhs_dilate = [2, 1]"),
        "jax.lax.conv_transpose, stride 2: a convolution of the input dilated by 2")
    add("conv_transpose_random", lambda x, w: jax.lax.conv_transpose(x, w, (2, 2), "SAME", dimension_numbers=nhwc), [rand(rng, (1, 4, 4, 3)), rand(rng, (3, 3, 3, 5), 0.3)],
        ("lhs_dilate = [2, 2]", "lhs_dilate = [2, 1]"), "jax.lax.conv_transpose with a 3x3 kernel, random")
    add("patch_embed", lambda x, w, b: jax.lax.conv_general_dilated(x, w, (4, 4), "VALID", dimension_numbers=nhwc).reshape(1, 16, 8) + b[None, None, :],
        [rand(rng, (1, 16, 16, 3)), rand(rng, (4, 4, 3, 8), 0.2), rand(rng, (8,), 0.1)], ("stride = [4, 4]", "stride = [4, 3]"), "a ViT patch embedding: a 4x4 convolution with stride 4, flattened, plus a bias")

    # --- windows, clamp, pad -----------------------------------------------------------------------------
    pool = lambda init, f, win, strides, padding: (lambda x: jax.lax.reduce_window(x, init, f, win, strides, padding))
    px = rand(rng, (1, 8, 8, 3))
    add("maxpool_2x2", pool(-jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID"), [px], ("window_strides = array<i64: 1, 2, 2, 1>", "window_strides = array<i64: 1, 2, 1, 1>"), "2x2 max pooling, stride 2")
    add("maxpool_3x3_same", pool(-jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), "SAME"), [px], ("window_dimensions = array<i64: 1, 3, 3, 1>", "window_dimensions = array<i64: 1, 3, 2, 1>"),
        "3x3 max pooling, stride 2, padded with -inf")
    add("minpool_2x2", pool(jnp.inf, jax.lax.min, (1, 2, 2, 1), (1, 1, 1, 1), "VALID"), [px], ("stablehlo.minimum", "stablehlo.maximum"), "2x2 min pooling, stride 1")
    add("sumpool_exact", pool(0.0, jax.lax.add, (1, 3, 3, 1), (1, 2, 2, 1), "SAME"), [ints(rng, (1, 8, 8, 3))], ("window_dimensions = array<i64: 1, 3, 3, 1>", "window_dimensions = array<i64: 1, 3, 2, 1>"),
        "3x3 sum pooling, stride 2, zero padded, integer-valued")
    add("avgpool_2x2", lambda x: jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 2, 2, 1), (1, 2, 2, 1), "VALID") / 4.0, [px], ("stablehlo.divide", "stablehlo.multiply"),
        "2x2 average pooling: a window sum of four, written out in the window's order, then a divide")
    add("avgpool_3x3", lambda x: jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 3, 3, 1), (1, 1, 1, 1), "VALID") / 9.0, [px], ("stablehlo.divide", "stablehlo.multiply"),
        "3x3 average pooling, stride 1: a window sum of nine")
    spv2 = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e-45, -1e-45, 1e-39, 1.17549435e-38, 2.0, -2.0, 3.4028235e38, 0.5, -0.5], np.float32)
    add("clamp_special", lambda a: jax.lax.clamp(-1.0, a, 1.0), [np.concatenate([spv2, rand(rng, (112,), 2.0)]).astype(np.float32)], ("stablehlo.clamp %0, %arg0, %1", "stablehlo.clamp %1, %arg0, %0"),
        "stablehlo.clamp against scalar bounds, special values included")
    add("clamp_tensor_bounds", lambda lo, a, hi: jax.lax.clamp(lo, a, hi), [rand(rng, (8, 16)) - 1.0, rand(rng, (8, 16), 2.0), rand(rng, (8, 16)) + 1.0],
        ("stablehlo.clamp %arg0, %arg1, %arg2", "stablehlo.clamp %arg2, %arg1, %arg0"), "stablehlo.clamp against tensor bounds")
    add("pad_value", lambda a: jnp.pad(a, ((1, 2), (0, 3)), constant_values=1.5), [rand(rng, (6, 8))], ("low = [1, 0], high = [2, 3]", "low = [2, 0], high = [1, 3]"), "pad with a constant that is not zero")
    add("pad_interior", lambda a: jax.lax.pad(a, np.float32(0), ((1, 2, 1), (0, 0, 2))), [rand(rng, (6, 8))], ("interior = [1, 2]", "interior = [2, 1]"), "pad with interior padding: zeros between the elements")
    add("reduce_min_last", lambda a: a.min(axis=-1), [rand(rng, (8, 8))], ("stablehlo.minimum", "stablehlo.maximum"), "reduce minimum")
    add("reduce_any_all", lambda a: (jnp.any(a > 1.0, axis=-1) | jnp.all(a > -1.5, axis=0)[:8]).astype(jnp.float32), [rand(rng, (8, 16))], ("stablehlo.or across", "stablehlo.and across"),
        "reduce or and reduce and of run-time predicates")

    # --- slices and updates by index ---------------------------------------------------------------------
    add("dynamic_slice_runtime", lambda a, i: jax.lax.dynamic_slice(a, (i,), (3,)), [rand(rng, (8,)), np.array(4, np.int32)], ("sizes = [3]", "sizes = [2]"),
        "dynamic_slice with a start computed at run time: a gather of start .. start + 2")
    add("dynamic_slice_clamped", lambda a, i, j: jax.lax.dynamic_slice(a, (i, j), (2, 3)), [rand(rng, (5, 6)), np.array(4, np.int32), np.array(-2, np.int32)], ("dense<6>", "dense<5>"),
        "dynamic_slice of two dimensions whose starts are clamped so that the slice fits")
    add("update_slice_set", lambda a, b: a.at[2:5].set(b), [rand(rng, (8,)), rand(rng, (3,))], ("dense<2>", "dense<3>"), "x.at[2:5].set(y): a scatter that writes one window")
    add("update_block_set", lambda a, b: a.at[1:3, 2:5].set(b), [rand(rng, (4, 6)), rand(rng, (2, 3))], ("dense<[1, 2]>", "dense<[2, 2]>"), "x.at[1:3, 2:5].set(y): a window of two dimensions")
    add("dynamic_update_slice", lambda a, b: jax.lax.dynamic_update_slice(a, b, (np.int32(1), np.int32(2))), [rand(rng, (4, 6)), rand(rng, (2, 3))], ("dense<1>", "dense<0>"),
        "dynamic_update_slice with a start known at compile time")

    # --- resize as jax.image.resize writes it ------------------------------------------------------------
    rx = rand(rng, (1, 4, 4, 3))
    add("resize_nearest_up", lambda a: jax.image.resize(a, (1, 8, 8, 3), "nearest"), [rx], ("dense<4.000000e+00>", "dense<3.000000e+00>"),
        "jax.image.resize nearest, 2x: an index from floats (floor, convert to i32) and two gathers")
    add("resize_bilinear_up", lambda a: jax.image.resize(a, (1, 8, 8, 3), "bilinear"), [rx], ("dense<3.500000e+00>", "dense<2.500000e+00>"),
        "jax.image.resize bilinear, 2x: weight matrices computed in the module and two dot_generals")
    add("resize_bilinear_exact", lambda a: jax.image.resize(a, (1, 8, 8, 3), "bilinear"), [ints(rng, (1, 4, 4, 3)) * 16.0], ("dense<3.500000e+00>", "dense<2.500000e+00>"),
        "jax.image.resize bilinear, 2x, on multiples of 16: every weight is a quarter, so every partial sum is exact")
    return out


def lower(fn, args):
    specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args]
    return jax.jit(fn).lower(*specs).as_text()


def mutated(text, rule):
    old, new = rule
    if old == "OPERANDS":
        for line in text.split("\n"):
            if "stablehlo.dot_general" in line:
                head, tail = line.split("stablehlo.dot_general ", 1)
                ops, rest = tail.split(", ", 2)[0:2], tail.split(", ", 2)[2]
                return text.replace(line, head + "stablehlo.dot_general " + ops[1] + ", " + ops[0] + ", " + rest, 1)
        raise SystemExit("no dot_general to mutate")
    if old == "DOT":
        return mutated(text, ("OPERANDS", ""))
    if old not in text:
        raise SystemExit("mutation target %r not in module:\n%s" % (old, text))
    return text.replace(old, new, 1)


def dtype_tag(a):
    return {np.dtype(np.float32): "f32", np.dtype(BF16): "bf16", np.dtype(np.int32): "i32"}[a.dtype]


def kept_references(path):
    """The reference bytes of the KEEP_REFERENCE cases, from the file that is being replaced."""
    if platform.system() == "Windows":
        return {}
    try:
        text = open(path).read()
    except OSError:
        raise SystemExit("%s must exist: it holds the references of %s, which only Windows can generate" % (path, ", ".join(KEEP_REFERENCE)))
    kept = {}
    for name in KEEP_REFERENCE:
        m = re.search(r"static const unsigned char %s_out\[\] = \{([^}]*)\};" % name, text)
        if m is None:
            raise SystemExit("%s has no reference for %s to keep" % (path, name))
        kept[name] = m.group(1)
    return kept


def c_bytes(a):
    raw = np.ascontiguousarray(a).tobytes()
    return ",".join("0x%02x" % v for v in raw)


def main():
    path = sys.argv[1]
    kept = kept_references(path)
    lines = ["// Generated by gen_stablehlo_cases.py with jax %s. Do not edit." % jax.__version__]
    if kept:
        lines.append("// The references of %s are kept from the Windows run: they are the C runtime's sinf and cosf." % ", ".join(KEEP_REFERENCE))
    lines.append("")
    names = []
    seen = set()
    for idx, (name, fn, args, rule, note) in enumerate(cases()):
        if name in seen:
            raise SystemExit("case %s is defined twice" % name)
        seen.add(name)
        text = lower(fn, args)
        res = np.asarray(jax.jit(fn)(*[jnp.asarray(a) for a in args]))
        bad = mutated(text, rule)
        if "stablehlo.custom_call" in text:
            raise SystemExit("%s lowered to something outside the slice:\n%s" % (name, text))
        if res.size == 0 or not np.any(np.asarray(res, np.float64) != 0):
            raise SystemExit("%s has a reference of zeros, which proves nothing" % name)
        lines.append("// %s: %s" % (name, note))
        for k, a in enumerate(args):
            lines.append("static const unsigned char %s_a%d[] = {%s};" % (name, k, c_bytes(a)))
        lines.append("static const unsigned char %s_out[] = {%s};" % (name, kept.get(name) or c_bytes(res)))
        lines.append('static const char %s_hlo[] = R"HLO(%s)HLO";' % (name, text))
        lines.append('static const char %s_bad[] = R"HLO(%s)HLO";' % (name, bad))
        argdefs = ",".join("{StablehloDtype::%s,%d,%s_a%d,sizeof(%s_a%d)}" % (dtype_tag(a).upper(), a.size, name, k, name, k) for k, a in enumerate(args))
        lines.append("static const StablehloArg %s_args[] = {%s};" % (name, argdefs))
        names.append((name, dtype_tag(res), res.size, len(args), note))
        lines.append("")
    lines.append("static const StablehloCase kStablehloCases[] = {")
    for unknown in sorted(set(XFAIL) - seen):
        raise SystemExit("XFAIL names %s, which is not a case" % unknown)
    for name, tag, count, nargs, note in names:
        xf = XFAIL.get(name)
        lines.append('\t{"%s", %s_hlo, %s_bad, %s_args, %d, %s_out, sizeof(%s_out), StablehloDtype::%s, "%s", %s},' % (
            name, name, name, name, nargs, name, name, tag.upper(), note, 'nullptr' if xf is None else '"%s"' % xf))
    lines.append("};")
    open(path, "w", newline="\n").write("\n".join(lines) + "\n")
    print("wrote %s: %d cases" % (path, len(names)))


if __name__ == "__main__":
    main()
