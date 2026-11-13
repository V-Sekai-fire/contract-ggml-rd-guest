// L2 cases for OPT_STEP_SGD (f32): the update in place on a weight, against ggml-cpu.
//
// Shapes: test-backend-ops' [10,5,4,2], odd sizes and a one-row tensor. A gradient
// with other strides than the weight is not tested: ggml-cpu reads it at the weight's
// offsets, so it has no reference, and supports() refuses it. The parameters are alpha = 0.1 and a weight decay
// of 0.01 as test-backend-ops uses. The update is two multiplies and a subtract
// per element in ggml-cpu's order, so every case is expected bit-exact.
#include <array>
#include <cstdio>
#include <string>
#include "../l2.h"
namespace {
using Ne = std::array<int64_t, 4>;

// The parameter tensor is a leaf whose two elements init() pins to alpha and wd.
ggml_tensor *step(ggml_context *ctx, Ne ne, bool permuted) {
	ggml_tensor *w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
	ggml_tensor *g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
	if (permuted) {
		// ggml_set_param asserts on a view, so only the gradient is strided.
		g = ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[2], ne[1], ne[3]), 0, 2, 1, 3);
	}
	ggml_set_param(w); // ggml_opt_step_sgd asserts its weight is a parameter
	ggml_tensor *params = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
	ggml_set_name(params, "sgd_params");
	return ggml_opt_step_sgd(ctx, w, g, params);
}

void add(std::vector<L2Case> &out, Ne ne, bool permuted) {
	L2Case c;
	char b[120];
	std::snprintf(b, sizeof b, "OPT_STEP_SGD ne=[%lld,%lld,%lld,%lld]%s", (long long)ne[0], (long long)ne[1], (long long)ne[2],
			(long long)ne[3], permuted ? " grad permuted" : "");
	c.name = b;
	c.build = [=](ggml_context *ctx) { return step(ctx, ne, permuted); };
	c.init = [](ggml_tensor *t, std::mt19937 &) {
		if (t->name[0] == 's' && std::string(t->name) == "sgd_params") {
			float *d = reinterpret_cast<float *>(t->data);
			d[0] = 0.1f;
			d[1] = 0.01f;
			return true;
		}
		return false;
	};
	out.push_back(c);
}
} // namespace
L2_CASES(opt_step) {
	add(out, { 10, 5, 4, 2 }, false);
	add(out, { 7, 3, 5, 2 }, false);
	add(out, { 64, 1, 1, 1 }, false);
	add(out, { 1, 1, 1, 1 }, false);
	add(out, { 6, 4, 3, 2 }, false);
	add(out, { 33, 1, 7, 1 }, false);
	add(out, { 1280, 16, 1, 1 }, false);
}
