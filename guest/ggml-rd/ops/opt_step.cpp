// ggml-rd op: OPT_STEP_SGD, f32 (ggml_opt_step_sgd): dst is src0, the weight,
// updated in place as w * keep - alpha * grad, with alpha = params[0] and
// keep = 1 - alpha * params[1].
//
// The kernel is lean/Ggml/SlangCodegen/OptStep.lean (opt_step_sgd_f32): one
// thread per element, every operand strided, no derived words.
//
// supports(): src0, src1 (the gradient) and dst f32 of one shape and one stride pattern, src2 the
// two-element f32 parameter tensor, everything in the params words.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_opt_step_sgd_f32(const ggml_tensor *op) {
	const ggml_tensor *w = op->src[0];
	const ggml_tensor *g = op->src[1];
	const ggml_tensor *p = op->src[2];
	if (w == nullptr || g == nullptr || p == nullptr || op->src[3] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32 || p->type != GGML_TYPE_F32) {
		return false;
	}
	// ggml-cpu reads the gradient at the weight's byte offsets, so a gradient
	// laid out differently has no defined result there: refuse it.
	if (w->nb[0] != g->nb[0] || w->nb[1] != g->nb[1] || w->nb[2] != g->nb[2] || w->nb[3] != g->nb[3]) {
		return false;
	}
	if (!ggml_are_same_shape(w, g) || !ggml_are_same_shape(w, op) || ggml_nelements(p) != 2 || !ggml_is_contiguous(p)) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(w) && tensor_fits(g) && tensor_fits(p);
}

bool pack_opt_step_sgd_f32(Pack &p) {
	p.kernel = kernel_index("opt_step_sgd_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(opt_step_sgd_f32, GGML_OP_OPT_STEP_SGD, -1, supports_opt_step_sgd_f32, pack_opt_step_sgd_f32);
