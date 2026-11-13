// L2 cases for CROSS_ENTROPY_LOSS (f32), against ggml-cpu.
//
// The labels are a distribution per row in the training use, but the loss is
// defined for any labels, so the uniform fill in [0, 1) is a valid input and
// exercises every term. Logits are in [-4, 4) so the row max matters. The
// sum is f32 in a tree where ggml-cpu sums rows in double, so the bar is
// test-backend-ops' NMSE (1e-7), not bit equality.
#include <cstdio>
#include "../l2.h"
namespace {
// pad > 0: both sources are views of rows `pad` columns wider, so nb1 is not ne0 * 4.
void add(std::vector<L2Case> &out, int64_t nc, int64_t nr, int64_t n2, int64_t pad = 0) {
	L2Case c;
	char b[96];
	std::snprintf(b, sizeof b, "CROSS_ENTROPY_LOSS nc=%lld nr=%lld n2=%lld pad=%lld", (long long)nc, (long long)nr, (long long)n2, (long long)pad);
	c.name = b;
	c.value_blind = nc == 1; // one column: log-softmax is 0, so the loss is 0 for any strides
	c.lo = -4.0f;
	c.hi = 4.0f;
	c.build = [=](ggml_context *ctx) {
		ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nc + pad, nr, n2);
		ggml_tensor *y = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nc + pad, nr, n2);
		if (pad > 0) {
			x = ggml_view_3d(ctx, x, nc, nr, n2, x->nb[1], x->nb[2], 0);
			y = ggml_view_3d(ctx, y, nc, nr, n2, y->nb[1], y->nb[2], 0);
		}
		return ggml_cross_entropy_loss(ctx, x, y);
	};
	out.push_back(c);
}
} // namespace
L2_CASES(loss) {
	add(out, 10, 1, 1);
	add(out, 10, 5, 1);
	add(out, 32, 8, 2);
	add(out, 1000, 4, 1);
	add(out, 7, 300, 1); // more rows than threads: a thread takes two
	add(out, 5, 257, 3);
	add(out, 1, 17, 1);
	add(out, 10, 6, 2, 3);
	add(out, 33, 300, 1, 7);
}
