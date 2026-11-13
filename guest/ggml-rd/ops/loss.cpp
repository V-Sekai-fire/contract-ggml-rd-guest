// ggml-rd ops: CROSS_ENTROPY_LOSS and CROSS_ENTROPY_LOSS_BACK, f32.
//
// The kernel is lean/Ggml/SlangCodegen/Loss.lean (ce_loss_f32 and its
// _serial sibling for the host tests): one 256-thread work group, thread t
// takes rows t, t + 256, ..., and thread 0 writes -(1 / nrows) * sum. Word 53
// is the thread count, so the grid is that one group (grid_1d over tg threads).
//
// supports(): logits, labels and dst f32, the two sources one shape and
// contiguous (ggml-cpu reads row r at r * nb1 and says it does not handle
// permuted ones: columns contiguous, dims 2 and 3 packed on nb1), dst one element.
#include "rows.h"

namespace {

using namespace ggml_rd;

bool supports_ce_loss(const ggml_tensor *op) {
	const ggml_tensor *x = op->src[0];
	const ggml_tensor *y = op->src[1];
	if (x == nullptr || y == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32) {
		return false;
	}
	if (!ggml_are_same_shape(x, y) || ggml_nelements(op) != 1) {
		return false;
	}
	// ggml-cpu finds row r of the flattened dims 1 to 3 at r * nb1, so dims 2
	// and 3 must be packed on nb1 and the columns contiguous.
	for (const ggml_tensor *t : { x, y }) {
		if (t->nb[0] != sizeof(float) || t->nb[2] != t->nb[1] * size_t(t->ne[1]) || t->nb[3] != t->nb[2] * size_t(t->ne[2])) {
			return false;
		}
	}
	return ggml_nelements(x) <= 0xFFFFFFFFll && tensor_fits(op) && tensor_fits(x) && tensor_fits(y);
}

bool pack_ce_loss(Pack &p) {
	p.kernel = kernel_index("ce_loss_f32");
	if (p.kernel < 0) {
		return false;
	}
	return grid_1d(p, kernel_desc(p.kernel).threadgroup[0]);
}

// CROSS_ENTROPY_LOSS_BACK: src0 the one-element gradient, src1 the logits, src2 the labels, dst
// shaped like the logits. ggml-cpu asserts all four contiguous, so the packer does too. One thread
// per row of the flattened dims 1 to 3 (ce_loss_back_f32).
bool supports_ce_loss_back(const ggml_tensor *op) {
	const ggml_tensor *g = op->src[0];
	const ggml_tensor *x = op->src[1];
	const ggml_tensor *y = op->src[2];
	if (g == nullptr || x == nullptr || y == nullptr || op->src[3] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32) {
		return false;
	}
	if (ggml_nelements(g) != 1 || !ggml_are_same_shape(x, y) || !ggml_are_same_shape(x, op)) {
		return false;
	}
	if (!ggml_is_contiguous(x) || !ggml_is_contiguous(y) || !ggml_is_contiguous(op) || !ggml_is_contiguous(g)) {
		return false;
	}
	return ggml_nelements(x) <= 0xFFFFFFFFll && tensor_fits(op) && tensor_fits(g) && tensor_fits(x) && tensor_fits(y);
}

bool pack_ce_loss_back(Pack &p) {
	p.kernel = kernel_index("ce_loss_back_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nrows(p.node)));
}

} // namespace

GGML_RD_OP(ce_loss_f32, GGML_OP_CROSS_ENTROPY_LOSS, -1, supports_ce_loss, pack_ce_loss);
GGML_RD_OP(ce_loss_back_f32, GGML_OP_CROSS_ENTROPY_LOSS_BACK, -1, supports_ce_loss_back, pack_ce_loss_back);
