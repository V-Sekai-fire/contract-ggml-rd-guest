import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Rows
import Ggml.SlangCodegen.Reduce

/-!
# `Ggml.SlangCodegen.Loss` — ggml CROSS_ENTROPY_LOSS, f32

`ggml_cross_entropy_loss(logits, labels)`: dst is one f32,

    loss = -(1 / nrows) · Σ_rows Σ_i (x_i - mx - log Σ_j exp(x_j - mx)) · y_i

over src0's rows (`ne1 * ne2 * ne3` of them, `ne0` columns each). One work
group of 256 threads, like `SUM`: thread `t` takes rows `t`, `t + 256`, ...
whole (max, then the exponential sum, then the weighted log-probabilities,
each serially over the row), keeps their sum as its partial, and `Rows`'
tree adds the partials. Thread 0 scales by `-1 / float(nrows)` and writes
`dst[offset]`.

ggml-cpu reads both sources by row through `nb1` only (row `r` of the
flattened `ne1 * ne2 * ne3` at `r * nb1`; its own TODO says permuted inputs
are not handled), so the kernel does the same through the word of `nb1`,
and the packer takes sources whose dims 1 to 3 are packed on `nb1`.
It sums in f32 where ggml-cpu sums rows in double per thread, so the result
matches within test-backend-ops' NMSE, not bit for bit.
-/

namespace Ggml.SlangCodegen.Loss

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Rows
open Ggml.SlangCodegen.Reduce (entryOneGroup)

def inc (i : String) : SlangStmt := .assign (v i) (add (v i) (u 1))

def xAt : SlangExpr := .index (v "s0") (add (v "b0") (v "i"))

/-- Thread `t`'s partial over its rows, into `sh[t]`. -/
def partialRows (tg : Nat) : List SlangStmt :=
  [ .declInit fl "acc" (fLit 0.0)
  , .declInit ui "r" (v "t")
  , .whileLoop (lt (v "r") (v "nr"))
      [ .declInit ui "b0" (add (v "o0") (mul (v "r") (v "k0")))
      , .declInit ui "b1" (add (v "o1") (mul (v "r") (v "k1")))
      , .declInit fl "mx" negInf
      , .declInit ui "i" (u 0)
      , .whileLoop (lt (v "i") (v "nc"))
          [ .assign (v "mx") (.call "max" [v "mx", xAt]), inc "i" ]
      , .declInit fl "se" (fLit 0.0)
      , .assign (v "i") (u 0)
      , .whileLoop (lt (v "i") (v "nc"))
          [ .assign (v "se") (add (v "se") (.call "exp" [sub xAt (v "mx")])), inc "i" ]
      , .declInit fl "lse" (.call "log" [v "se"])
      , .declInit fl "rs" (fLit 0.0)
      , .assign (v "i") (u 0)
      , .whileLoop (lt (v "i") (v "nc"))
          [ .assign (v "rs") (add (v "rs")
              (mul (sub (sub xAt (v "mx")) (v "lse")) (.index (v "s1") (add (v "b1") (v "i")))))
          , inc "i" ]
      , .assign (v "acc") (add (v "acc") (v "rs"))
      , .assign (v "r") (add (v "r") (u tg)) ]
  , .assign (sh (v "t")) (v "acc") ]

def total (tg : Nat) (var : Variant) : List SlangStmt :=
  let step (s : Nat) : SlangStmt := .assign (sh (v "t")) (add (sh (v "t")) (sh (add (v "t") (u s))))
  match var with
  | .par =>
      [ .declInit ui "t" (v "e") ] ++ partialRows tg ++ [barrier] ++
      ((treeSteps tg).map (fun s => [ .ifNoElse (lt (v "t") (u s)) [step s], barrier ])).flatten ++
      [ .declInit fl "total" (sh (u 0)) ]
  | .ser =>
      [ .declareArray fl "sh" tg
      , .forCount "t" (u 0) (u tg) (partialRows tg) ] ++
      (treeSteps tg).map (fun s => .forCount "t" (u 0) (u s) [step s]) ++
      [ .declInit fl "total" (sh (u 0)) ]

def body (tg : Nat) (var : Variant) : List SlangStmt :=
  [ .declInit ui "nc" (pwN (wSrc 0 + tNe))
  , .declInit ui "nr" (mul (mul (pwN (wSrc 0 + tNe + 1)) (pwN (wSrc 0 + tNe + 2))) (pwN (wSrc 0 + tNe + 3)))
  , .declInit ui "o0" (pwN (wSrc 0 + tOff))
  , .declInit ui "o1" (pwN (wSrc 1 + tOff))
  , .declInit ui "k0" (pwN (wSrc 0 + tNb + 1))
  , .declInit ui "k1" (pwN (wSrc 1 + tNb + 1)) ] ++
  total tg var ++
  (let st := SlangStmt.assign (.index (v "dst") (pwN (wDst + tOff)))
    (mul (v "total") (fdiv (fLit (-1.0)) (.call "float" [v "nr"])))
   match var with
   | .par => [ .ifNoElse (.bin "==" (v "e") (u 0)) [st] ]
   | .ser => [ st ])

def module (tg : Nat) (var : Variant) : SlangShaderModule :=
  let m := kernelModule .float .float .uint .float [fnPw, fnOff4]
    (match var with
     | .par => entryOneGroup tg (body tg var)
     | .ser => entry1D 1 (body tg var))
  match var with
  | .par => { m with groupShared := [ { name := "sh", elemType := fl, dims := [tg] } ] }
  | .ser => m

/-! ## CROSS_ENTROPY_LOSS_BACK: one thread per row -/

/-- `ggml_cross_entropy_loss_back(g, x, y)`: dst, shaped like `x`, is
    `(softmax(x) - y) · g / nr` per row, with `g` the one-element gradient in
    src0, `x` in src1 and `y` in src2. One thread per row, in ggml-cpu's order:
    `exp(x - mx)` scaled by the reciprocal of its sum, minus the label, times
    `g / float(nr)`. Rows are found through `nb1` as in the forward kernel. -/
def backBody : List SlangStmt :=
  let xAt1 : SlangExpr := .index (v "s1") (add (v "b1") (v "i"))
  [ .declInit ui "nc" (pwN (wSrc 1 + tNe))
  , .declInit ui "nr" (mul (mul (pwN (wSrc 1 + tNe + 1)) (pwN (wSrc 1 + tNe + 2))) (pwN (wSrc 1 + tNe + 3)))
  , .declInit ui "b1" (add (pwN (wSrc 1 + tOff)) (mul (v "e") (pwN (wSrc 1 + tNb + 1))))
  , .declInit ui "b2" (add (pwN (wSrc 2 + tOff)) (mul (v "e") (pwN (wSrc 2 + tNb + 1))))
  , .declInit ui "bd" (add (pwN (wDst + tOff)) (mul (v "e") (pwN (wDst + tNb + 1))))
  , .declInit fl "dn" (fdiv (.index (v "s0") (pwN (wSrc 0 + tOff))) (.call "float" [v "nr"]))
  , .declInit fl "mx" negInf
  , .declInit ui "i" (u 0)
  , .whileLoop (lt (v "i") (v "nc"))
      [ .assign (v "mx") (.call "max" [v "mx", xAt1]), inc "i" ]
  , .declInit fl "se" (fLit 0.0)
  , .assign (v "i") (u 0)
  , .whileLoop (lt (v "i") (v "nc"))
      [ .assign (v "se") (add (v "se") (.call "exp" [sub xAt1 (v "mx")])), inc "i" ]
  , .declInit fl "inv" (fdiv (fLit 1.0) (v "se"))
  , .assign (v "i") (u 0)
  , .whileLoop (lt (v "i") (v "nc"))
      [ .assign (.index (v "dst") (add (v "bd") (v "i")))
          (mul (sub (mul (.call "exp" [sub xAt1 (v "mx")]) (v "inv"))
                    (.index (v "s2") (add (v "b2") (v "i")))) (v "dn"))
      , inc "i" ] ]

def ceLossBackF32 : SlangShaderModule :=
  kernelModule .float .float .float .float [fnPw, fnOff4] (entry1D threadgroup backBody)

def ceLossF32 : SlangShaderModule := module threadgroup .par
def ceLossF32Serial : SlangShaderModule := module threadgroup .ser

def kernels : List (String × SlangShaderModule) :=
  [ ("ce_loss_f32", ceLossF32)
  , ("ce_loss_f32_serial", ceLossF32Serial)
  , ("ce_loss_back_f32", ceLossBackF32) ]

end Ggml.SlangCodegen.Loss
