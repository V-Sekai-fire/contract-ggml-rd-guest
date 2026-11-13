// ggml-rd ops: CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_BACK, SILU_BACK and SOFT_MAX_BACK, f32.
//
// The kernel is lean/Ggml/SlangCodegen/Loss.lean (ce_loss_f32 and its
// _serial sibling for the host tests): one 256-thread work group, thread t
// takes rows t, t + 256, ..., and thread 0 writes -(1 / nrows) * sum. Word 53
// is the thread count, so the grid is that one group (grid_1d over tg threads).
//
// supports(): logits, labels and dst f32, the two sources one shape and
// contiguous (ggml-cpu reads row r at r * nb1 and says it does not handle
// permuted ones: columns contiguous, dims 2 and 3 packed on nb1), dst one element.
#include <cstring>

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

// SILU_BACK: src0 the incoming gradient, src1 the forward input, dst the same shape. ggml-cpu
// asserts src1 contiguous in dim 0 and the three one shape. Rows go through nb1, as ggml-cpu reads them.
bool supports_silu_back(const ggml_tensor *op) {
	const ggml_tensor *dy = op->src[0];
	const ggml_tensor *x = op->src[1];
	if (dy == nullptr || x == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || dy->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32) {
		return false;
	}
	if (!ggml_are_same_shape(x, dy) || !ggml_are_same_shape(x, op)) {
		return false;
	}
	// ggml-cpu finds row r of the flattened dims 1 to 3 at r * nb1, so a layout with dims 2 and 3 not
	// packed on nb1 (a permuted view) has no defined result there: refuse it.
	for (const ggml_tensor *t : { dy, x, op }) {
		if (t->nb[0] != sizeof(float) || t->nb[2] != t->nb[1] * size_t(t->ne[1]) || t->nb[3] != t->nb[2] * size_t(t->ne[2])) {
			return false;
		}
	}
	return tensor_fits(op) && tensor_fits(dy) && tensor_fits(x);
}

bool pack_silu_back(Pack &p) {
	p.kernel = kernel_index("silu_back_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

// SOFT_MAX_BACK: src0 the incoming gradient, src1 the forward output, scale = op_params[0],
// max_bias (op_params[1]) 0 as ggml-cpu asserts. All three contiguous, as it asserts. One thread per row.
bool supports_soft_max_back(const ggml_tensor *op) {
	const ggml_tensor *dy = op->src[0];
	const ggml_tensor *y = op->src[1];
	if (dy == nullptr || y == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || dy->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32) {
		return false;
	}
	if (!ggml_are_same_shape(dy, y) || !ggml_are_same_shape(dy, op)) {
		return false;
	}
	if (!ggml_is_contiguous(dy) || !ggml_is_contiguous(y) || !ggml_is_contiguous(op)) {
		return false;
	}
	float max_bias;
	std::memcpy(&max_bias, reinterpret_cast<const float *>(op->op_params) + 1, sizeof(float));
	return max_bias == 0.0f && tensor_fits(op) && tensor_fits(dy) && tensor_fits(y);
}

bool pack_soft_max_back(Pack &p) {
	p.kernel = kernel_index("soft_max_back_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nrows(p.node)));
}

} // namespace

GGML_RD_OP(ce_loss_f32, GGML_OP_CROSS_ENTROPY_LOSS, -1, supports_ce_loss, pack_ce_loss);
GGML_RD_OP(ce_loss_back_f32, GGML_OP_CROSS_ENTROPY_LOSS_BACK, -1, supports_ce_loss_back, pack_ce_loss_back);
GGML_RD_OP(silu_back_f32, GGML_OP_SILU_BACK, -1, supports_silu_back, pack_silu_back);
GGML_RD_OP(soft_max_back_f32, GGML_OP_SOFT_MAX_BACK, -1, supports_soft_max_back, pack_soft_max_back);
