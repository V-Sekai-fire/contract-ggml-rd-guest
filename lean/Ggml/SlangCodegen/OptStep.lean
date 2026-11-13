import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Unary

/-!
# `Ggml.SlangCodegen.OptStep` — ggml OPT_STEP_SGD, f32

`ggml_opt_step_sgd(w, grad, params)`: `w' = w * keep - alpha * grad` with
`alpha = params[0]` and `keep = 1 - alpha * params[1]` (params[1] is the
weight decay). dst is `w` itself, so one thread per element reads its own
element of `w`, the matching element of `grad`, and writes the result back.

Both products and the subtraction are single f32 operations in the same
order as ggml-cpu's `ggml_compute_forward_opt_step_sgd_f32`, and `keep` is
computed per thread from the two parameters in f32, so the result is
bit-exact against ggml-cpu when neither side contracts the multiply-adds.
`w` and `grad` share a shape and may be strided views; `params` is a
2-element tensor read through its own block.
-/

namespace Ggml.SlangCodegen.OptStep

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Unary (fl fsub)

def threadgroup : Nat := 256

def flT : SlangType := .scalar .float
def ui : SlangType := .scalar .uint

def body : List SlangStmt :=
  let s2 := wSrc 2
  [ .decl ui "i0", .decl ui "i1", .decl ui "i2", .decl ui "i3"
  , .expr (.call "unravel4" [v "e", u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit ui "d" (.call "off4" [u wDst, v "i0", v "i1", v "i2", v "i3"])
  , .declInit ui "a" (.call "off4" [u (wSrc 0), v "i0", v "i1", v "i2", v "i3"])
  , .declInit ui "b" (.call "off4" [u (wSrc 1), v "i0", v "i1", v "i2", v "i3"])
  , .declInit flT "alpha" (.index (v "s2") (.call "off4" [u s2, u 0, u 0, u 0, u 0]))
  , .declInit flT "wd" (.index (v "s2") (.call "off4" [u s2, u 1, u 0, u 0, u 0]))
  , .declInit flT "keep" (fsub (fl 1.0) (mul (v "alpha") (v "wd")))
  , .assign (.index (v "dst") (v "d"))
      (fsub (mul (.index (v "s0") (v "a")) (v "keep"))
            (mul (v "alpha") (.index (v "s1") (v "b")))) ]

def sgdF32 : SlangShaderModule :=
  kernelModule .float .float .float .float [fnPw, fnUnravel4, fnOff4] (entry1D threadgroup body)

def kernels : List (String × SlangShaderModule) :=
  [ ("opt_step_sgd_f32", sgdF32) ]

end Ggml.SlangCodegen.OptStep
