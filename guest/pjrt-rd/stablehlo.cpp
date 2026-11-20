#include "stablehlo.h"
#include "../ggml-rd/rsqrt_zen2.h"
#include "../ggml-rd/xla_compare.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "ggml.h"

namespace pjrt_rd {

namespace {

// Element types. An i32 or i1 value known at compile time (iota, constants, compares of them, masks) is folded
// on the host and has no tensor. One computed at run time is an f32 tensor of whole numbers (0 and 1 for an i1)
// that carries the label i32 or i1: exact while every value stays inside 2^24. GGML_TYPE_I8 stands for i1.
const ggml_type kPred = GGML_TYPE_I8;

bool is_float(ggml_type t) {
	return t == GGML_TYPE_F32 || t == GGML_TYPE_BF16;
}

struct Val {
	ggml_tensor *t = nullptr; // null for an i32 or i1 value known at compile time; f32 for one computed at run time
	std::vector<int64_t> shape; // row-major, as StableHLO writes it
	ggml_type type = GGML_TYPE_F32;
	bool neg_inf = false; // a constant -inf, or a reshape or broadcast of one
	ggml_tensor *pre = nullptr; // for a broadcast: the operand before it, shaped like the broadcast with 1s
	std::vector<int64_t> pre_shape;
	bool known = false; // the value is known at compile time: hv holds it, row-major
	std::vector<double> hv; // exact for f32, bf16, i32 and i1
};

struct Ty {
	std::vector<int64_t> shape;
	ggml_type type = GGML_TYPE_F32;
};

bool starts_with_unsupported(const std::string &e) {
	return e.compare(0, 11, "unsupported") == 0;
}

struct Lowering {
	ggml_context *ctx = nullptr;
	std::map<std::string, Val> vals;
	std::vector<ggml_tensor *> params;
	std::unordered_map<const ggml_tensor *, std::vector<uint8_t>> consts;
	std::map<std::pair<std::vector<int64_t>, uint32_t>, ggml_tensor *> fills; // constants of one value, by ne and bits
	std::string err;

	bool fail(const std::string &e) {
		if (err.empty()) {
			err = e;
		}
		return false;
	}
	bool unsupported(const std::string &e) {
		return fail("unsupported: " + e);
	}
	// A type error keeps its "unsupported" marker, which picks the error code, outside the context.
	bool type_error(const std::string &context, const std::string &e) {
		return starts_with_unsupported(e) ? fail(e) : fail(context + ": " + e);
	}
};

std::string trim(const std::string &s) {
	size_t a = 0;
	size_t b = s.size();
	while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) {
		++a;
	}
	while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) {
		--b;
	}
	return s.substr(a, b - a);
}

bool starts_with(const std::string &s, const char *p) {
	return s.compare(0, std::strlen(p), p) == 0;
}

bool parse_type_at(const std::string &s, size_t at, Ty &out, size_t &end, std::string &err) {
	const size_t lt = at + 7; // after "tensor<"
	const size_t gt = s.find('>', lt);
	if (s.compare(at, 7, "tensor<") != 0 || gt == std::string::npos) {
		err = "expected a tensor type at '" + s.substr(at, 24) + "'";
		return false;
	}
	std::string body = s.substr(lt, gt - lt);
	end = gt + 1;
	out.shape.clear();
	std::vector<std::string> parts;
	size_t p = 0;
	while (true) {
		const size_t x = body.find('x', p);
		if (x == std::string::npos) {
			parts.push_back(body.substr(p));
			break;
		}
		parts.push_back(body.substr(p, x - p));
		p = x + 1;
	}
	const std::string elem = parts.back();
	if (elem == "f32") {
		out.type = GGML_TYPE_F32;
	} else if (elem == "bf16") {
		out.type = GGML_TYPE_BF16;
	} else if (elem == "i32") {
		out.type = GGML_TYPE_I32;
	} else if (elem == "i1") {
		out.type = kPred;
	} else {
		err = "unsupported: element type '" + elem + "' (f32, bf16, and i32 and i1 known at compile time)";
		return false;
	}
	for (size_t i = 0; i + 1 < parts.size(); ++i) {
		char *e = nullptr;
		const long long d = std::strtoll(parts[i].c_str(), &e, 10);
		if (parts[i].empty() || *e != '\0' || d <= 0) {
			err = "unsupported: dimension '" + parts[i] + "' (static shapes only)";
			return false;
		}
		out.shape.push_back(d);
	}
	// A ggml tensor has four dimensions. A higher rank is kept when its dimensions of size one can be dropped
	// (ne_of), which is what attention's rank-5 intermediates need.
	size_t wide = 0;
	for (int64_t d : out.shape) {
		wide += d > 1 ? 1 : 0;
	}
	if (wide > GGML_MAX_DIMS) {
		err = "unsupported: more than 4 dimensions of size above one";
		return false;
	}
	return true;
}

int64_t count_of(const std::vector<int64_t> &shape) {
	int64_t n = 1;
	for (int64_t d : shape) {
		n *= d;
	}
	return n;
}

// ggml ne of a row-major shape: the dimensions reversed. A shape of rank above 4 drops its dimensions of size
// one (parse_type_at has checked that at most four are left); every tensor of such a value is stored that way.
void ne_of(const std::vector<int64_t> &shape, int64_t ne[GGML_MAX_DIMS]) {
	for (int i = 0; i < GGML_MAX_DIMS; ++i) {
		ne[i] = 1;
	}
	std::vector<int64_t> s;
	for (int64_t d : shape) {
		if (shape.size() <= GGML_MAX_DIMS || d != 1) {
			s.push_back(d);
		}
	}
	const size_t r = s.size();
	for (size_t i = 0; i < r && i < GGML_MAX_DIMS; ++i) {
		ne[i] = s[r - 1 - i];
	}
}

// The shape without the dimensions that keep[i] does not name.
std::vector<int64_t> kept_dims(const std::vector<int64_t> &shape, const std::vector<bool> &keep) {
	std::vector<int64_t> o;
	for (size_t i = 0; i < shape.size(); ++i) {
		if (keep[i]) {
			o.push_back(shape[i]);
		}
	}
	return o;
}

ggml_tensor *new_tensor(Lowering &L, ggml_type type, const std::vector<int64_t> &shape) {
	int64_t ne[GGML_MAX_DIMS];
	ne_of(shape, ne);
	return ggml_new_tensor_4d(L.ctx, type, ne[0], ne[1], ne[2], ne[3]);
}

// Host arrays, row-major. Calls f(index, flat) for every element of shape, in order.
template <typename F>
void each_index(const std::vector<int64_t> &shape, F f) {
	const int64_t n = count_of(shape);
	std::vector<int64_t> idx(shape.size(), 0);
	for (int64_t flat = 0; flat < n; ++flat) {
		f(idx, size_t(flat));
		for (size_t k = shape.size(); k-- > 0;) {
			if (++idx[k] < shape[k]) {
				break;
			}
			idx[k] = 0;
		}
	}
}

std::vector<int64_t> strides_of(const std::vector<int64_t> &shape) {
	std::vector<int64_t> s(shape.size(), 1);
	for (size_t k = shape.size(); k-- > 1;) {
		s[k - 1] = s[k] * shape[k];
	}
	return s;
}

// out dimension i is input dimension perm[i].
std::vector<double> h_transpose(const std::vector<double> &in, const std::vector<int64_t> &shape, const std::vector<int64_t> &perm) {
	std::vector<int64_t> os;
	for (int64_t p : perm) {
		os.push_back(shape[size_t(p)]);
	}
	const std::vector<int64_t> is = strides_of(shape);
	std::vector<double> out(in.size());
	each_index(os, [&](const std::vector<int64_t> &o, size_t flat) {
		int64_t at = 0;
		for (size_t i = 0; i < o.size(); ++i) {
			at += o[i] * is[size_t(perm[i])];
		}
		out[flat] = in[size_t(at)];
	});
	return out;
}

std::vector<double> h_broadcast(const std::vector<double> &in, const std::vector<int64_t> &shape, const std::vector<int64_t> &dims, const std::vector<int64_t> &to) {
	const std::vector<int64_t> is = strides_of(shape);
	std::vector<double> out(size_t(count_of(to)));
	each_index(to, [&](const std::vector<int64_t> &o, size_t flat) {
		int64_t at = 0;
		for (size_t k = 0; k < dims.size(); ++k) {
			at += shape[k] == 1 ? 0 : o[size_t(dims[k])] * is[k];
		}
		out[flat] = in[size_t(at)];
	});
	return out;
}

std::vector<double> h_slice(const std::vector<double> &in, const std::vector<int64_t> &shape, const std::vector<int64_t> &start,
		const std::vector<int64_t> &stride, const std::vector<int64_t> &count) {
	const std::vector<int64_t> is = strides_of(shape);
	std::vector<double> out(size_t(count_of(count)));
	each_index(count, [&](const std::vector<int64_t> &o, size_t flat) {
		int64_t at = 0;
		for (size_t i = 0; i < o.size(); ++i) {
			at += (start[i] + o[i] * stride[i]) * is[i];
		}
		out[flat] = in[size_t(at)];
	});
	return out;
}

// Rounds a value to what the element type can hold, the way a store of it would.
double h_round(ggml_type type, double v) {
	if (type == GGML_TYPE_F32) {
		return double(float(v));
	}
	if (type == GGML_TYPE_BF16) {
		return double(ggml_bf16_to_fp32(ggml_fp32_to_bf16(float(v))));
	}
	return v;
}

// An i32 operation's result, wrapped to 32 bits as two's complement.
double h_wrap32(int64_t v) {
	return double(int32_t(uint32_t(uint64_t(v))));
}

ggml_tensor *contiguous(Lowering &L, ggml_tensor *t) {
	return ggml_is_contiguous(t) ? t : ggml_cont(L.ctx, t);
}

Val reshaped(Lowering &L, const Val &v, const std::vector<int64_t> &shape) {
	Val o;
	if (v.t != nullptr) {
		int64_t ne[GGML_MAX_DIMS];
		ne_of(shape, ne);
		o.t = ggml_reshape_4d(L.ctx, contiguous(L, v.t), ne[0], ne[1], ne[2], ne[3]);
	}
	o.shape = shape;
	o.type = v.type;
	o.neg_inf = v.neg_inf;
	o.known = v.known;
	o.hv = v.hv;
	return o;
}

// A value known at compile time. A float one also gets a constant tensor; an i32 or i1 one stays on the host.
Val known_val(Lowering &L, ggml_type type, const std::vector<int64_t> &shape, std::vector<double> hv) {
	Val v;
	v.shape = shape;
	v.type = type;
	v.known = true;
	v.hv = std::move(hv);
	if (is_float(type)) {
		v.t = new_tensor(L, type, shape);
		const size_t esz = type == GGML_TYPE_F32 ? 4 : 2;
		std::vector<uint8_t> bytes(v.hv.size() * esz);
		bool all_neg_inf = !v.hv.empty();
		for (size_t i = 0; i < v.hv.size(); ++i) {
			const float f = float(v.hv[i]);
			all_neg_inf = all_neg_inf && std::isinf(f) && f < 0.0f;
			if (type == GGML_TYPE_F32) {
				std::memcpy(bytes.data() + i * 4, &f, 4);
			} else {
				const ggml_bf16_t b = ggml_fp32_to_bf16(f);
				std::memcpy(bytes.data() + i * 2, &b, 2);
			}
		}
		L.consts[v.t] = std::move(bytes);
		v.neg_inf = all_neg_inf;
	}
	return v;
}

Val as_type(Lowering &L, const Val &v, ggml_type type) {
	if (v.type == type) {
		return v;
	}
	const bool from_whole = v.type == kPred || v.type == GGML_TYPE_I32;
	const bool to_whole = type == kPred || type == GGML_TYPE_I32;
	if ((from_whole && type == GGML_TYPE_F32) || (v.type == GGML_TYPE_F32 && to_whole)) {
		// An i1 or i32 value computed at run time is an f32 tensor of whole numbers: only the label changes.
		Val o = v;
		o.type = type;
		return o;
	}
	Val o = v;
	o.t = ggml_cast(L.ctx, v.t, type);
	o.type = type;
	if (o.known) {
		for (double &x : o.hv) {
			x = h_round(type, x);
		}
	}
	return o;
}

// out dimension i is input dimension perm[i], as stablehlo.transpose says.
Val transposed(Lowering &L, const Val &v, const std::vector<int64_t> &perm) {
	const int r = int(v.shape.size());
	bool identity = true;
	for (int i = 0; i < r; ++i) {
		identity = identity && perm[size_t(i)] == i;
	}
	if (identity) {
		return v;
	}
	std::vector<int64_t> out_shape;
	for (int i = 0; i < r; ++i) {
		out_shape.push_back(v.shape[size_t(perm[size_t(i)])]);
	}
	if (v.t == nullptr || r > GGML_MAX_DIMS) {
		Val o;
		if (v.t != nullptr) {
			// Above rank 4 the tensor holds only the dimensions of size above one (ne_of); permute those.
			std::vector<int64_t> at(size_t(r), -1);
			Val s = v;
			s.shape.clear();
			s.known = false;
			for (int i = 0; i < r; ++i) {
				if (v.shape[size_t(i)] != 1) {
					at[size_t(i)] = int64_t(s.shape.size());
					s.shape.push_back(v.shape[size_t(i)]);
				}
			}
			std::vector<int64_t> sp;
			for (int i = 0; i < r; ++i) {
				if (at[size_t(perm[size_t(i)])] >= 0) {
					sp.push_back(at[size_t(perm[size_t(i)])]);
				}
			}
			o = transposed(L, s, sp);
		}
		o.shape = out_shape;
		o.type = v.type;
		o.neg_inf = v.neg_inf;
		if (v.known) {
			o.known = true;
			o.hv = h_transpose(v.hv, v.shape, perm);
		}
		return o;
	}
	int axis[GGML_MAX_DIMS] = { 0, 1, 2, 3 };
	for (int k = 0; k < r; ++k) {
		const int hlo_in = r - 1 - k;
		for (int i = 0; i < r; ++i) {
			if (perm[size_t(i)] == hlo_in) {
				axis[k] = r - 1 - i;
			}
		}
	}
	Val o;
	o.t = ggml_cont(L.ctx, ggml_permute(L.ctx, v.t, axis[0], axis[1], axis[2], axis[3]));
	o.type = v.type;
	o.shape = out_shape;
	o.neg_inf = v.neg_inf;
	if (v.known) {
		o.known = true;
		o.hv = h_transpose(v.hv, v.shape, perm);
	}
	return o;
}

bool int_list(const std::string &s, const char *key, size_t from, std::vector<int64_t> &out, size_t *after) {
	const size_t k = s.find(key, from);
	if (k == std::string::npos) {
		return false;
	}
	const size_t lb = s.find('[', k);
	const size_t rb = s.find(']', lb);
	if (lb == std::string::npos || rb == std::string::npos) {
		return false;
	}
	out.clear();
	size_t p = lb + 1;
	while (p < rb) {
		while (p < rb && (s[p] == ' ' || s[p] == ',')) {
			++p;
		}
		if (p >= rb) {
			break;
		}
		char *e = nullptr;
		out.push_back(std::strtoll(s.c_str() + p, &e, 10));
		p = size_t(e - s.c_str());
	}
	if (after != nullptr) {
		*after = rb + 1;
	}
	return true;
}

// "name = [a, b] x [c, d]"
bool int_list_pair(const std::string &s, const char *key, std::vector<int64_t> &lhs, std::vector<int64_t> &rhs) {
	size_t after = 0;
	if (!int_list(s, key, 0, lhs, &after)) {
		return false;
	}
	const size_t x = s.find('x', after);
	const size_t lb = s.find('[', after);
	if (x == std::string::npos || lb == std::string::npos || x > lb) {
		return false;
	}
	return int_list(s, "[", after, rhs, nullptr);
}

// An f32 constant of the given shape with every element v.
Val filled(Lowering &L, const std::vector<int64_t> &shape, float v) {
	Val o;
	o.t = new_tensor(L, GGML_TYPE_F32, shape);
	o.shape = shape;
	o.type = GGML_TYPE_F32;
	std::vector<uint8_t> bytes(size_t(count_of(shape)) * 4);
	for (size_t i = 0; i < bytes.size() / 4; ++i) {
		std::memcpy(bytes.data() + i * 4, &v, 4);
	}
	L.consts[o.t] = std::move(bytes);
	return o;
}

// An f32 constant tensor with like's ne, every element v.
ggml_tensor *filled_like(Lowering &L, const ggml_tensor *like, float v) {
	uint32_t bits = 0;
	std::memcpy(&bits, &v, 4);
	const std::pair<std::vector<int64_t>, uint32_t> key(std::vector<int64_t>(like->ne, like->ne + GGML_MAX_DIMS), bits);
	const std::map<std::pair<std::vector<int64_t>, uint32_t>, ggml_tensor *>::const_iterator have = L.fills.find(key);
	if (have != L.fills.end()) {
		return have->second;
	}
	ggml_tensor *t = ggml_new_tensor_4d(L.ctx, GGML_TYPE_F32, like->ne[0], like->ne[1], like->ne[2], like->ne[3]);
	L.fills[key] = t;
	std::vector<uint8_t> bytes(size_t(ggml_nelements(t)) * 4);
	for (size_t i = 0; i < bytes.size() / 4; ++i) {
		std::memcpy(bytes.data() + i * 4, &v, 4);
	}
	L.consts[t] = std::move(bytes);
	return t;
}

// XLA CPU's maximum or minimum against one constant (a tensor of one element, repeated by the kernel).
ggml_tensor *xla_against(Lowering &L, ggml_rd_xla_binary_kind kind, ggml_tensor *a, float c) {
	const Val k = filled(L, {}, c);
	return ggml_rd_xla_binary_op(L.ctx, a->view_src != nullptr || !ggml_is_contiguous(a) ? ggml_cont(L.ctx, a) : a, k.t, kind);
}

// c[0] * z^(n-1) + ... + c[n-1], as multiplies and adds that are not fused.
ggml_tensor *horner(Lowering &L, ggml_tensor *z, const float *c, int n) {
	ggml_tensor *acc = ggml_scale_bias(L.ctx, z, c[0], c[1]);
	for (int i = 2; i < n; ++i) {
		acc = ggml_scale_bias(L.ctx, ggml_mul(L.ctx, z, acc), 1.0f, c[i]);
	}
	return acc;
}

// erf as XLA CPU computes it for f32 (xla/codegen/intrinsic/erf.cc at d867969800d4): the rational
// x * P(x^2) / Q(x^2) of Eigen, and +-1 from |x| = 3.8325069 on. XLA evaluates P and Q with fused multiply-adds;
// here each multiply and add is its own kernel, so the result is within a few ulp of XLA's and not bit-equal.
ggml_tensor *erf_of(Lowering &L, ggml_tensor *x) {
	static const float kAlpha[] = { 0.00022905065861350646f, 0.0034082910107109506f, 0.050955695062380861f, 0.18520832239976145f, 1.128379143519084f };
	static const float kBeta[] = { -1.1791602954361697e-7f, 0.000023547966471313185f, 0.0010179625278914885f, 0.014070470171167667f, 0.11098505178285362f,
		0.49746925110067538f, 1.0f };
	const float lim = 3.832506856900711f;
	ggml_tensor *xc = xla_against(L, GGML_RD_XLA_MIN, xla_against(L, GGML_RD_XLA_MAX, x, -lim), lim);
	ggml_tensor *x2 = ggml_mul(L.ctx, xc, xc);
	ggml_tensor *p = ggml_mul(L.ctx, xc, horner(L, x2, kAlpha, 5));
	ggml_tensor *q = horner(L, x2, kBeta, 7);
	return xla_against(L, GGML_RD_XLA_MIN, xla_against(L, GGML_RD_XLA_MAX, ggml_div(L.ctx, p, q), -1.0f), 1.0f);
}

bool op_unary(Lowering &L, const std::string &op, const Val &a, Val &out) {
	Val f = as_type(L, a, GGML_TYPE_F32);
	Val o = f;
	if (op == "negate") {
		o.t = ggml_neg(L.ctx, f.t);
	} else if (op == "sine") {
		o.t = ggml_sin(L.ctx, f.t);
	} else if (op == "cosine") {
		o.t = ggml_cos(L.ctx, f.t);
	} else if (op == "exponential") {
		o.t = ggml_exp(L.ctx, f.t);
	} else if (op == "log") {
		o.t = ggml_log(L.ctx, f.t);
	} else if (op == "sqrt") {
		o.t = ggml_sqrt(L.ctx, f.t);
	} else if (op == "abs") {
		o.t = ggml_abs(L.ctx, f.t);
	} else if (op == "floor") {
		o.t = ggml_floor(L.ctx, f.t);
	} else if (op == "sign") {
		// SGN gives a zero for a zero, but not a signed one: ggml-cpu returns +0 and ggml-rd -0, and XLA returns the
		// zero it was given. Rebuild the sign with exact arithmetic that does not depend on it:
		//   s = sgn(x), ss = s*s (1 for a nonzero, +0 for a zero), c = s + sgn(1/x) * (1 - ss), result = c * ss.
		// c is +-1 with the sign of x everywhere (1/+-0 is +-inf), so c * (+0) at a zero is a zero with x's sign.
		// A NaN gives NaN (see below).
		const Val ones = filled(L, f.shape, 1.0f);
		ggml_tensor *sg = ggml_sgn(L.ctx, f.t);
		ggml_tensor *ss = ggml_mul(L.ctx, sg, sg);
		ggml_tensor *q = ggml_sgn(L.ctx, ggml_div(L.ctx, ones.t, f.t));
		ggml_tensor *c = ggml_add(L.ctx, sg, ggml_mul(L.ctx, q, ggml_sub(L.ctx, ones.t, ss)));
		// A NaN is the one input where ss = 0 and sgn(1/x) = 0 (a zero has sgn(1/x) = +-1, an infinity has ss = 1).
		// ind = (1 - ss) * (1 - e*e) is 1 only there, and 1 + 0/(1 - ind) is NaN there (0/0) and 1 elsewhere.
		const Val zeros = filled(L, f.shape, 0.0f);
		ggml_tensor *e2 = ggml_mul(L.ctx, q, q);
		ggml_tensor *ind = ggml_mul(L.ctx, ggml_sub(L.ctx, ones.t, ss), ggml_sub(L.ctx, ones.t, e2));
		ggml_tensor *nan_or_one = ggml_add(L.ctx, ones.t, ggml_div(L.ctx, zeros.t, ggml_sub(L.ctx, ones.t, ind)));
		o.t = ggml_mul(L.ctx, ggml_mul(L.ctx, c, ss), nan_or_one);
	} else if (op == "rsqrt") {
		// XLA CPU's rsqrt is the CPU's estimate instruction and one Newton step; this is that on a Zen 2 (rsqrt_zen2.h).
		o.t = ggml_rd_rsqrt_zen2_op(L.ctx, contiguous(L, f.t));
	} else if (op == "logistic") {
		// XLA expands logistic to 1 / (1 + exp(-x)) before it emits code, and each of those is bit-equal here
		// (tools/unary_compose_xla.py: 0 of 920,015 inputs differ).
		ggml_tensor *e = ggml_exp(L.ctx, ggml_neg(L.ctx, f.t));
		ggml_tensor *d = ggml_scale_bias(L.ctx, e, 1.0f, 1.0f);
		o.t = ggml_div(L.ctx, filled_like(L, d, 1.0f), d);
	} else if (op == "tanh") {
		// ggml-rd's TANH kernel. XLA CPU's tanh is a clamped rational evaluated with fused multiply-adds
		// (xla/codegen/intrinsic/tanh.cc), so this is close to it and not bit-equal.
		o.t = ggml_tanh(L.ctx, f.t);
	} else if (op == "erf") {
		o.t = erf_of(L, contiguous(L, f.t));
	} else if (op == "erfc") {
		// 1 - erf(x): exact to an ulp of 1, so the tail (erfc below about 1e-7) has no relative precision.
		o.t = ggml_scale_bias(L.ctx, erf_of(L, contiguous(L, f.t)), -1.0f, 1.0f);
	} else {
		return L.unsupported("stablehlo." + op);
	}
	out = as_type(L, o, a.type);
	return true;
}

bool op_binary(Lowering &L, const std::string &op, const Val &a, const Val &b, Val &out) {
	if (a.shape != b.shape) {
		return L.fail("stablehlo." + op + ": operand shapes differ");
	}
	if (op == "maximum") {
		// Softmax writes max(-inf, row max); a maximum against a constant -inf returns the other operand, NaN included.
		if (a.neg_inf || b.neg_inf) {
			out = a.neg_inf ? b : a;
			return true;
		}
		return L.unsupported("stablehlo.maximum of two values (only against a constant -inf)");
	}
	Val fa = as_type(L, a, GGML_TYPE_F32);
	Val fb = as_type(L, b, GGML_TYPE_F32);
	Val o = fa;
	if (op == "add") {
		o.t = ggml_add(L.ctx, fa.t, fb.t);
	} else if (op == "subtract") {
		o.t = ggml_sub(L.ctx, fa.t, fb.t);
	} else if (op == "multiply") {
		o.t = ggml_mul(L.ctx, fa.t, fb.t);
	} else if (op == "power") {
		// XLA CPU's power is the C library's powf, except for the exponents its simplifier rewrites. Those measured
		// equal to a composition here (tools/unary_compose_xla.py) are lowered; the rest is refused.
		bool same = b.known && !b.hv.empty();
		for (size_t i = 1; same && i < b.hv.size(); ++i) {
			same = b.hv[i] == b.hv[0];
		}
		const double e = same ? b.hv[0] : 0.0;
		if (same && e == 2.0) {
			o.t = ggml_mul(L.ctx, fa.t, fa.t);
		} else if (same && e == 3.0) {
			o.t = ggml_mul(L.ctx, ggml_mul(L.ctx, fa.t, fa.t), fa.t);
		} else if (same && e == 1.0) {
			o.t = fa.t;
		} else if (same && e == -1.0) {
			o.t = ggml_div(L.ctx, filled_like(L, fa.t, 1.0f), fa.t);
		} else {
			return L.unsupported("stablehlo.power with an exponent other than a constant 2, 3, 1 or -1 (the general case is the C library's powf)");
		}
	} else if (op == "divide") {
		if (fb.pre != nullptr && fb.type == GGML_TYPE_F32 && fb.pre->type == GGML_TYPE_F32) {
			// XLA's algebraic simplifier rewrites a / broadcast(b) as a * broadcast(1 / b), which rounds twice.
			// Doing the same keeps the bits: for a divisor of 3 the two results differ in the last place.
			ggml_tensor *rcp = ggml_div(L.ctx, filled_like(L, fb.pre, 1.0f), fb.pre);
			o.t = ggml_mul(L.ctx, fa.t, ggml_repeat(L.ctx, rcp, fa.t));
		} else {
			o.t = ggml_div(L.ctx, fa.t, fb.t);
		}
	} else {
		return L.unsupported("stablehlo." + op);
	}
	out = as_type(L, o, a.type);
	return true;
}

float element_f32(const Val &c, const std::vector<uint8_t> &bytes) {
	if (c.type == GGML_TYPE_F32) {
		float f = 0.0f;
		std::memcpy(&f, bytes.data(), 4);
		return f;
	}
	ggml_bf16_t b;
	std::memcpy(&b, bytes.data(), 2);
	return ggml_bf16_to_fp32(b);
}

// stablehlo.reduce of add, maximum, minimum, and, or over some dimensions, with the operation's identity as init.
bool op_reduce(Lowering &L, const std::string &attrs, const Val &x, const Val &init, const Ty &to, Val &out) {
	const size_t ap = attrs.find("applies stablehlo.");
	if (ap == std::string::npos) {
		return L.unsupported("stablehlo.reduce with a body other than a named stablehlo op");
	}
	const size_t ks = ap + 18;
	const size_t ke = attrs.find(' ', ks);
	const std::string kind = attrs.substr(ks, ke == std::string::npos ? std::string::npos : ke - ks);
	const bool pred = x.type == kPred;
	const bool whole = pred || x.type == GGML_TYPE_I32;
	if (kind != "add" && kind != "maximum" && kind != "minimum" && kind != "and" && kind != "or") {
		return L.unsupported("stablehlo.reduce applying stablehlo." + kind);
	}
	if (((kind == "and" || kind == "or") && !pred) || (kind == "add" && pred)) {
		return L.unsupported("stablehlo.reduce applying stablehlo." + kind + " to this element type");
	}
	std::vector<int64_t> dims;
	if (!int_list(attrs, "dimensions", 0, dims, nullptr)) {
		return L.fail("stablehlo.reduce: no dimensions");
	}
	const int64_t r = int64_t(x.shape.size());
	std::vector<bool> reduced(size_t(r), false);
	for (int64_t d : dims) {
		if (d < 0 || d >= r || reduced[size_t(d)]) {
			return L.fail("stablehlo.reduce: dimension " + std::to_string(d) + " is out of range or repeated");
		}
		reduced[size_t(d)] = true;
	}
	if (!init.known || !init.shape.empty() || init.hv.size() != 1) {
		return L.unsupported("stablehlo.reduce with an init that is not a scalar constant");
	}
	const double iv = init.hv[0];
	const bool is_max = kind == "maximum" || kind == "or";
	const bool is_min = kind == "minimum" || kind == "and";
	bool identity = false;
	if (kind == "add") {
		identity = iv == 0.0 && !std::signbit(iv);
	} else if (pred) {
		identity = iv == (is_max ? 0.0 : 1.0);
	} else if (x.type == GGML_TYPE_I32) {
		identity = iv == (is_max ? -2147483648.0 : 2147483647.0);
	} else {
		identity = std::isinf(iv) && (is_max ? iv < 0.0 : iv > 0.0);
	}
	if (!identity) {
		return L.unsupported("stablehlo.reduce whose init is not the identity of " + kind);
	}
	std::vector<int64_t> kept, perm, flat;
	int64_t n = 1;
	for (int64_t i = 0; i < r; ++i) {
		if (!reduced[size_t(i)]) {
			kept.push_back(x.shape[size_t(i)]);
			perm.push_back(i);
		}
	}
	for (int64_t i = 0; i < r; ++i) {
		if (reduced[size_t(i)]) {
			perm.push_back(i);
			n *= x.shape[size_t(i)];
		}
	}
	if (kept != to.shape) {
		return L.fail("stablehlo.reduce: the result shape does not match the kept dimensions");
	}
	if (x.t == nullptr) {
		// An i32 or i1 value known at compile time: folded on the host.
		const std::vector<double> t = h_transpose(x.hv, x.shape, perm);
		std::vector<double> hv(size_t(count_of(kept)));
		for (size_t i = 0; i < hv.size(); ++i) {
			double acc = iv;
			for (int64_t k = 0; k < n; ++k) {
				const double v = t[i * size_t(n) + size_t(k)];
				acc = kind == "add" ? h_wrap32(int64_t(acc) + int64_t(v)) : is_max ? std::max(acc, v) : std::min(acc, v);
			}
			hv[i] = acc;
		}
		out = known_val(L, x.type, kept, std::move(hv));
		return true;
	}
	if (n == 1 && kept.size() + 1 > GGML_MAX_DIMS) {
		return L.unsupported("stablehlo.reduce over dimensions of size one above rank 4");
	}
	Val v = as_type(L, x, GGML_TYPE_F32);
	v = transposed(L, v, perm);
	flat = kept;
	flat.push_back(n);
	v = reshaped(L, v, flat);
	Val o;
	o.type = GGML_TYPE_F32;
	if (kind == "add") {
		o.t = ggml_sum_rows(L.ctx, v.t);
	} else if (is_max) {
		o.t = ggml_pool_1d(L.ctx, v.t, GGML_OP_POOL_MAX, int(n), int(n), 0);
	} else {
		// The minimum is the negated maximum of the negated values; a negation is exact.
		o.t = ggml_neg(L.ctx, ggml_pool_1d(L.ctx, ggml_neg(L.ctx, v.t), GGML_OP_POOL_MAX, int(n), int(n), 0));
		if (whole) {
			// -0 for a whole number is 0; adding 0 writes it as +0.
			o.t = ggml_scale_bias(L.ctx, o.t, 1.0f, 0.0f);
		}
	}
	std::vector<int64_t> rows = kept;
	rows.push_back(1);
	o.shape = rows;
	o = reshaped(L, o, kept);
	out = as_type(L, o, x.type);
	return true;
}

// stablehlo.slice %x [start:limit(:stride), ...]: a strided view, copied out.
bool slice_core(Lowering &L, const Val &x, const std::vector<int64_t> &start, const std::vector<int64_t> &stride,
		const std::vector<int64_t> &span, const std::vector<int64_t> &count, Val &out);

bool op_slice(Lowering &L, const std::string &attrs, const Val &x, const Ty &to, Val &out) {
	const size_t lb = attrs.find('[');
	const size_t rb = attrs.find(']', lb);
	if (lb == std::string::npos || rb == std::string::npos) {
		return L.fail("stablehlo.slice: no index ranges");
	}
	std::vector<int64_t> start, count, stride, span;
	size_t p = lb + 1;
	while (p < rb) {
		while (p < rb && (attrs[p] == ' ' || attrs[p] == ',')) {
			++p;
		}
		if (p >= rb) {
			break;
		}
		char *e = nullptr;
		const int64_t a = std::strtoll(attrs.c_str() + p, &e, 10);
		if (*e != ':') {
			return L.fail("stablehlo.slice: cannot read a start:limit range");
		}
		const int64_t b = std::strtoll(e + 1, &e, 10);
		int64_t st = 1;
		if (*e == ':') {
			st = std::strtoll(e + 1, &e, 10);
		}
		if (st < 1 || a < 0 || b < a) {
			return L.fail("stablehlo.slice: a range is empty, negative or has a stride below one");
		}
		start.push_back(a);
		stride.push_back(st);
		span.push_back(b - a);
		count.push_back((b - a + st - 1) / st);
		p = size_t(e - attrs.c_str());
	}
	const size_t r = x.shape.size();
	if (start.size() != r) {
		return L.fail("stablehlo.slice: " + std::to_string(start.size()) + " ranges for rank " + std::to_string(r));
	}
	for (size_t i = 0; i < r; ++i) {
		if (start[i] + (count[i] - 1) * stride[i] >= x.shape[i] && count[i] > 0) {
			return L.fail("stablehlo.slice: a range is past the end of its dimension");
		}
	}
	(void)to;
	if (x.t == nullptr) {
		out = Val();
		out.shape = count;
		out.type = x.type;
		out.known = true;
		out.hv = h_slice(x.hv, x.shape, start, stride, count);
		return true;
	}
	Val c = x;
	if (r > GGML_MAX_DIMS) {
		// Above rank 4 a dimension of size one can only be sliced [0:1]; slice the others.
		std::vector<bool> keep(r);
		for (size_t i = 0; i < r; ++i) {
			keep[i] = x.shape[i] != 1;
		}
		c.shape = kept_dims(x.shape, keep);
		c.known = false;
		Val o;
		if (!slice_core(L, c, kept_dims(start, keep), kept_dims(stride, keep), kept_dims(span, keep), kept_dims(count, keep), o)) {
			return false;
		}
		o = reshaped(L, o, count);
		if (x.known) {
			o.known = true;
			o.hv = h_slice(x.hv, x.shape, start, stride, count);
		}
		out = o;
		return true;
	}
	if (!slice_core(L, c, start, stride, span, count, out)) {
		return false;
	}
	if (x.known) {
		out.known = true;
		out.hv = h_slice(x.hv, x.shape, start, stride, count);
	}
	return true;
}

bool slice_core(Lowering &L, const Val &x, const std::vector<int64_t> &start, const std::vector<int64_t> &stride,
		const std::vector<int64_t> &span, const std::vector<int64_t> &count, Val &out) {
	const size_t r = x.shape.size();
	Val c = x;
	c.t = contiguous(L, x.t);
	// A ggml view cannot step along its first dimension (nb0 is the element size), so a stride on the
	// last StableHLO dimension is applied afterwards, on a transposed copy.
	const bool last_strided = r > 0 && stride[r - 1] > 1;
	int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
	size_t nb[GGML_MAX_DIMS] = { c.t->nb[0], c.t->nb[1], c.t->nb[2], c.t->nb[3] };
	size_t offset = 0;
	for (size_t i = 0; i < r; ++i) {
		const size_t k = r - 1 - i; // ggml dimension of StableHLO dimension i
		const bool defer = last_strided && i == r - 1;
		ne[k] = defer ? span[i] : count[i];
		nb[k] = c.t->nb[k] * (defer ? size_t(1) : size_t(stride[i]));
		offset += size_t(start[i]) * c.t->nb[k];
	}
	Val o;
	o.t = ggml_cont(L.ctx, ggml_view_4d(L.ctx, c.t, ne[0], ne[1], ne[2], ne[3], nb[1], nb[2], nb[3], offset));
	if (last_strided) {
		ggml_tensor *t = ggml_cont(L.ctx, ggml_permute(L.ctx, o.t, 1, 0, 2, 3));
		ggml_tensor *v = ggml_view_4d(L.ctx, t, t->ne[0], count[r - 1], t->ne[2], t->ne[3], t->nb[1] * size_t(stride[r - 1]), t->nb[2], t->nb[3], 0);
		o.t = ggml_cont(L.ctx, ggml_permute(L.ctx, ggml_cont(L.ctx, v), 1, 0, 2, 3));
	}
	o.shape = count;
	o.type = x.type;
	out = o;
	return true;
}

bool op_concatenate(Lowering &L, const std::string &attrs, const std::vector<Val> &in, Val &out) {
	const size_t d = attrs.find("dim = ");
	if (d == std::string::npos || in.empty()) {
		return L.fail("stablehlo.concatenate: no dim");
	}
	const int64_t dim = std::strtoll(attrs.c_str() + d + 6, nullptr, 10);
	const int64_t r = int64_t(in[0].shape.size());
	if (dim < 0 || dim >= r) {
		return L.fail("stablehlo.concatenate: dim is out of range");
	}
	std::vector<int64_t> out_shape = in[0].shape;
	bool all_known = in[0].known;
	for (size_t k = 1; k < in.size(); ++k) {
		if (in[k].type != in[0].type || in[k].shape.size() != in[0].shape.size()) {
			return L.fail("stablehlo.concatenate: operands differ in rank or element type");
		}
		for (int64_t i = 0; i < r; ++i) {
			if (i != dim && in[k].shape[size_t(i)] != in[0].shape[size_t(i)]) {
				return L.fail("stablehlo.concatenate: operands differ outside the concatenated dimension");
			}
		}
		out_shape[size_t(dim)] += in[k].shape[size_t(dim)];
		all_known = all_known && in[k].known;
	}
	std::vector<double> hv;
	if (all_known) {
		// Row-major: the outer dimensions in turn, each operand's block of the inner ones.
		int64_t outer = 1;
		for (int64_t i = 0; i < dim; ++i) {
			outer *= out_shape[size_t(i)];
		}
		for (int64_t o = 0; o < outer; ++o) {
			for (const Val &v : in) {
				const size_t block = size_t(count_of(v.shape) / outer);
				hv.insert(hv.end(), v.hv.begin() + std::ptrdiff_t(size_t(o) * block), v.hv.begin() + std::ptrdiff_t(size_t(o + 1) * block));
			}
		}
	}
	if (in[0].t == nullptr) {
		out = Val();
		out.shape = out_shape;
		out.type = in[0].type;
		out.known = true;
		out.hv = std::move(hv);
		return true;
	}
	// Above rank 4, concatenate on the dimensions of size above one and the concatenated one.
	std::vector<bool> keep(size_t(r), true);
	int64_t gdim = dim;
	if (r > GGML_MAX_DIMS) {
		gdim = 0;
		for (int64_t i = 0; i < r; ++i) {
			keep[size_t(i)] = i == dim || out_shape[size_t(i)] != 1;
			gdim += i < dim && keep[size_t(i)] ? 1 : 0;
		}
		if (kept_dims(out_shape, keep).size() > GGML_MAX_DIMS) {
			return L.unsupported("stablehlo.concatenate of more than 4 dimensions of size above one");
		}
	}
	const int64_t gr = int64_t(kept_dims(out_shape, keep).size());
	const bool squeeze = r > GGML_MAX_DIMS;
	Val acc = squeeze ? reshaped(L, in[0], kept_dims(in[0].shape, keep)) : in[0];
	for (size_t k = 1; k < in.size(); ++k) {
		const Val next = squeeze ? reshaped(L, in[k], kept_dims(in[k].shape, keep)) : in[k];
		Val f0 = as_type(L, acc, GGML_TYPE_F32);
		Val f1 = as_type(L, next, GGML_TYPE_F32);
		Val o = acc;
		o.t = as_type(L, Val{ ggml_concat(L.ctx, contiguous(L, f0.t), contiguous(L, f1.t), int(gr - 1 - gdim)), f0.shape, GGML_TYPE_F32 }, acc.type).t;
		o.shape[size_t(gdim)] += next.shape[size_t(gdim)];
		o.known = false;
		acc = o;
	}
	out = squeeze ? reshaped(L, acc, out_shape) : acc;
	out.neg_inf = false;
	if (all_known) {
		out.known = true;
		out.hv = std::move(hv);
	}
	return true;
}

bool op_concatenate(Lowering &L, const std::string &attrs, const std::vector<Val> &in, Val &out);

// The elements [start, start + count) of dimension dim.
bool slice_dim(Lowering &L, const Val &x, size_t dim, int64_t start, int64_t count, Val &out) {
	std::string a = "[";
	for (size_t i = 0; i < x.shape.size(); ++i) {
		a += (i == 0 ? "" : ", ") + (i == dim ? std::to_string(start) + ":" + std::to_string(start + count) : "0:" + std::to_string(x.shape[i]));
	}
	return op_slice(L, a + "]", x, Ty(), out);
}

bool concat_dim(Lowering &L, const std::vector<Val> &parts, size_t dim, Val &out) {
	if (parts.size() == 1) {
		out = parts[0];
		return true;
	}
	return op_concatenate(L, "dim = " + std::to_string(dim), parts, out);
}

// `gap` zeros after every element of dimension dim but the last: n elements become n + (n - 1) * gap.
bool dilate_dim(Lowering &L, const Val &x, size_t dim, int64_t gap, Val &out) {
	const size_t r = x.shape.size();
	const int64_t n = x.shape[dim];
	if (gap == 0 || n == 1) {
		out = x;
		return true;
	}
	std::vector<int64_t> perm, back(r), rest;
	for (size_t i = 0; i < r; ++i) {
		if (i != dim) {
			perm.push_back(int64_t(i));
			rest.push_back(x.shape[i]);
		}
	}
	perm.push_back(int64_t(dim));
	for (size_t i = 0; i < r; ++i) {
		back[size_t(perm[i])] = int64_t(i);
	}
	const int64_t others = count_of(rest);
	Val v = reshaped(L, transposed(L, as_type(L, x, GGML_TYPE_F32), perm), { others, n, 1 });
	Val padded = v;
	padded.t = ggml_pad_ext(L.ctx, contiguous(L, v.t), 0, int(gap), 0, 0, 0, 0, 0, 0);
	padded.shape = { others, n, 1 + gap };
	Val flat = reshaped(L, padded, { others, n * (1 + gap) });
	Val cut;
	if (!slice_dim(L, flat, 1, 0, n + (n - 1) * gap, cut)) {
		return false;
	}
	std::vector<int64_t> shape = rest;
	shape.push_back(n + (n - 1) * gap);
	out = as_type(L, transposed(L, reshaped(L, cut, shape), back), x.type);
	out.known = false;
	out.hv.clear();
	return true;
}

// Edge padding of a float or run-time whole-number value with one value. Zero is ggml's PAD; another value is
// a concatenation with constants, so the padded elements keep their bits.
bool pad_edges(Lowering &L, const Val &x, const std::vector<int64_t> &low, const std::vector<int64_t> &high, float value, Val &out) {
	const size_t r = x.shape.size();
	bool any = false;
	for (size_t i = 0; i < r; ++i) {
		any = any || low[i] != 0 || high[i] != 0;
	}
	if (!any) {
		out = x;
		return true;
	}
	Val f = as_type(L, x, GGML_TYPE_F32);
	if (value == 0.0f && !std::signbit(value) && r <= GGML_MAX_DIMS) {
		int lp[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
		int rp[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
		for (size_t i = 0; i < r; ++i) {
			lp[r - 1 - i] = int(low[i]);
			rp[r - 1 - i] = int(high[i]);
		}
		Val o = f;
		o.t = ggml_pad_ext(L.ctx, contiguous(L, f.t), lp[0], rp[0], lp[1], rp[1], lp[2], rp[2], lp[3], rp[3]);
		for (size_t i = 0; i < r; ++i) {
			o.shape[i] += low[i] + high[i];
		}
		out = as_type(L, o, x.type);
	} else {
		Val acc = f;
		for (size_t i = 0; i < r; ++i) {
			std::vector<Val> parts;
			std::vector<int64_t> shape = acc.shape;
			if (low[i] > 0) {
				shape[i] = low[i];
				parts.push_back(filled(L, shape, value));
			}
			parts.push_back(acc);
			if (high[i] > 0) {
				shape[i] = high[i];
				parts.push_back(filled(L, shape, value));
			}
			Val next;
			if (!concat_dim(L, parts, i, next)) {
				return false;
			}
			acc = next;
		}
		out = as_type(L, acc, x.type);
	}
	out.known = false;
	out.hv.clear();
	out.neg_inf = false;
	out.pre = nullptr;
	return true;
}

// stablehlo.pad with a constant padding value, non-negative edges and interior padding.
bool op_pad(Lowering &L, const std::string &attrs, const Val &x, const Val &pad_value, Val &out) {
	std::vector<int64_t> low, high, interior;
	if (!int_list(attrs, "low", 0, low, nullptr) || !int_list(attrs, "high", 0, high, nullptr) || !int_list(attrs, "interior", 0, interior, nullptr)) {
		return L.fail("stablehlo.pad: low, high and interior are all needed");
	}
	const size_t r = x.shape.size();
	if (low.size() != r || high.size() != r || interior.size() != r) {
		return L.fail("stablehlo.pad: the padding lists do not match the rank");
	}
	if (x.t == nullptr || r > GGML_MAX_DIMS) {
		return L.unsupported("stablehlo.pad of i32 or i1 values known at compile time, or above rank 4");
	}
	for (size_t i = 0; i < r; ++i) {
		if (interior[i] < 0) {
			return L.fail("stablehlo.pad: negative interior padding");
		}
		if (low[i] < 0 || high[i] < 0) {
			return L.unsupported("stablehlo.pad with negative edge padding");
		}
	}
	if (!pad_value.known || !pad_value.shape.empty() || pad_value.hv.size() != 1) {
		return L.unsupported("stablehlo.pad with a padding value that is not a scalar constant");
	}
	const float value = float(pad_value.hv[0]);
	Val acc = x;
	for (size_t i = 0; i < r; ++i) {
		if (interior[i] != 0) {
			if (value != 0.0f || std::signbit(value)) {
				return L.unsupported("stablehlo.pad with interior padding of a value other than zero");
			}
			Val next;
			if (!dilate_dim(L, acc, i, interior[i], next)) {
				return false;
			}
			acc = next;
		}
	}
	return pad_edges(L, acc, low, high, value, out);
}

bool op_broadcast(Lowering &L, const Val &x, const std::vector<int64_t> &dims, const Ty &to, Val &out) {
	if (dims.size() != x.shape.size()) {
		return L.fail("stablehlo.broadcast_in_dim: dims has " + std::to_string(dims.size()) + " entries for rank " + std::to_string(x.shape.size()));
	}
	std::vector<int64_t> mid(to.shape.size(), 1);
	int64_t last = -1;
	for (size_t i = 0; i < dims.size(); ++i) {
		if (dims[i] <= last || dims[i] >= int64_t(to.shape.size())) {
			return L.unsupported("stablehlo.broadcast_in_dim with dims that are not strictly increasing");
		}
		last = dims[i];
		if (x.shape[i] != to.shape[size_t(dims[i])] && x.shape[i] != 1) {
			return L.fail("stablehlo.broadcast_in_dim: dimension " + std::to_string(i) + " does not fit");
		}
		mid[size_t(dims[i])] = x.shape[i];
	}
	if (x.t == nullptr) {
		out = Val();
		out.shape = to.shape;
		out.type = x.type;
		out.known = true;
		out.hv = h_broadcast(x.hv, x.shape, dims, to.shape);
		return true;
	}
	Val r;
	if (to.shape.size() > GGML_MAX_DIMS) {
		// Above rank 4 the result drops its dimensions of size one (ne_of); drop the same ones from mid, so the
		// repeat lines up.
		std::vector<int64_t> m;
		for (size_t i = 0; i < mid.size(); ++i) {
			if (to.shape[i] != 1) {
				m.push_back(mid[i]);
			}
		}
		const Val sq = reshaped(L, x, m);
		r = sq;
		r.shape = mid;
	} else {
		r = reshaped(L, x, mid);
	}
	if (mid == to.shape) {
		out = r;
		if (x.known) {
			out.hv = x.hv;
		}
		return true;
	}
	Val o;
	int64_t ne[GGML_MAX_DIMS];
	ne_of(to.shape, ne);
	o.t = ggml_repeat_4d(L.ctx, r.t, ne[0], ne[1], ne[2], ne[3]);
	o.shape = to.shape;
	o.type = x.type;
	o.neg_inf = x.neg_inf;
	o.pre = r.t;
	o.pre_shape = mid;
	if (x.known) {
		o.known = true;
		o.hv = h_broadcast(x.hv, x.shape, dims, to.shape);
	}
	out = o;
	return true;
}

// Reshapes only when the ggml ne changes: a reshape to the same ne would add a node and change nothing.
Val reshaped_ne(Lowering &L, const Val &v, const std::vector<int64_t> &shape) {
	int64_t a[GGML_MAX_DIMS], b[GGML_MAX_DIMS];
	ne_of(v.shape, a);
	ne_of(shape, b);
	if (v.t != nullptr && ggml_is_contiguous(v.t) && std::equal(a, a + GGML_MAX_DIMS, b)) {
		Val o = v;
		o.shape = shape;
		return o;
	}
	return reshaped(L, v, shape);
}

// dot_general: the batch, free and contracting dimensions of each side are each flattened into one, so the
// product is one ggml_mul_mat of [B, M, K] by [B, N, K]; the result [B, M, N] is reshaped to the batch dims,
// the lhs free dims and the rhs free dims, the order StableHLO gives them.
bool op_dot(Lowering &L, const std::string &attrs, const Val &lhs, const Val &rhs, const Ty &to, Val &out) {
	std::vector<int64_t> lb, rb, lc, rc;
	int_list_pair(attrs, "batching_dims", lb, rb);
	if (!int_list_pair(attrs, "contracting_dims", lc, rc) || lc.empty() || lc.size() != rc.size() || lb.size() != rb.size()) {
		return L.unsupported("stablehlo.dot_general without contracting dimensions paired on both sides");
	}
	if (lhs.t == nullptr || rhs.t == nullptr) {
		return L.unsupported("stablehlo.dot_general of i32 or i1 values");
	}
	const int64_t lr = int64_t(lhs.shape.size());
	const int64_t rr = int64_t(rhs.shape.size());
	std::vector<int> luse(size_t(lr), 0), ruse(size_t(rr), 0);
	for (size_t i = 0; i < lc.size(); ++i) {
		if (lc[i] < 0 || lc[i] >= lr || rc[i] < 0 || rc[i] >= rr || luse[size_t(lc[i])]++ != 0 || ruse[size_t(rc[i])]++ != 0) {
			return L.fail("stablehlo.dot_general: a contracting dimension is out of range or repeated");
		}
		if (lhs.shape[size_t(lc[i])] != rhs.shape[size_t(rc[i])]) {
			return L.fail("stablehlo.dot_general: contracted dimensions differ");
		}
	}
	for (size_t i = 0; i < lb.size(); ++i) {
		if (lb[i] < 0 || lb[i] >= lr || rb[i] < 0 || rb[i] >= rr || luse[size_t(lb[i])]++ != 0 || ruse[size_t(rb[i])]++ != 0) {
			return L.fail("stablehlo.dot_general: a batch dimension is out of range or repeated");
		}
		if (lhs.shape[size_t(lb[i])] != rhs.shape[size_t(rb[i])]) {
			return L.fail("stablehlo.dot_general: batch dimensions differ");
		}
	}
	std::vector<int64_t> lfree, rfree;
	for (int64_t d = 0; d < lr; ++d) {
		if (luse[size_t(d)] == 0) {
			lfree.push_back(d);
		}
	}
	for (int64_t d = 0; d < rr; ++d) {
		if (ruse[size_t(d)] == 0) {
			rfree.push_back(d);
		}
	}
	int64_t B = 1, M = 1, N = 1, K = 1;
	std::vector<int64_t> want;
	for (int64_t d : lb) {
		B *= lhs.shape[size_t(d)];
		want.push_back(lhs.shape[size_t(d)]);
	}
	for (int64_t d : lfree) {
		M *= lhs.shape[size_t(d)];
		want.push_back(lhs.shape[size_t(d)]);
	}
	for (int64_t d : rfree) {
		N *= rhs.shape[size_t(d)];
		want.push_back(rhs.shape[size_t(d)]);
	}
	for (int64_t d : lc) {
		K *= lhs.shape[size_t(d)];
	}
	std::vector<int64_t> lperm = lb;
	lperm.insert(lperm.end(), lfree.begin(), lfree.end());
	lperm.insert(lperm.end(), lc.begin(), lc.end());
	std::vector<int64_t> rperm = rb;
	rperm.insert(rperm.end(), rfree.begin(), rfree.end());
	rperm.insert(rperm.end(), rc.begin(), rc.end());
	Val l = reshaped_ne(L, transposed(L, lhs, lperm), { B, M, K });
	Val r = reshaped_ne(L, transposed(L, rhs, rperm), { B, N, K });
	l = as_type(L, l, GGML_TYPE_F32);
	Val o;
	o.t = ggml_mul_mat(L.ctx, contiguous(L, r.t), contiguous(L, l.t));
	o.type = GGML_TYPE_F32;
	o.shape = { B, M, N };
	out = as_type(L, reshaped(L, o, want), to.type);
	return true;
}

bool op_constant(Lowering &L, const std::string &line, const Ty &ty, Val &out) {
	const size_t d = line.find("dense<");
	size_t e = std::string::npos;
	if (d != std::string::npos) {
		e = line.find('>', d);
	}
	if (d == std::string::npos || e == std::string::npos) {
		return L.fail("stablehlo.constant without dense<...>");
	}
	const std::string body = line.substr(d + 6, e - d - 6);
	const int64_t n = count_of(ty.shape);
	std::vector<double> values;
	if (body.compare(0, 3, "\"0x") == 0) {
		// A large constant is printed as its raw bytes, little-endian, in hex: every element, or one to splat.
		// An i1 element is one byte.
		const size_t close = body.find('"', 3);
		const size_t esz = ty.type == GGML_TYPE_F32 || ty.type == GGML_TYPE_I32 ? 4 : ty.type == GGML_TYPE_BF16 ? 2 : 1;
		if (close == std::string::npos || (close - 3) % 2 != 0) {
			return L.fail("stablehlo.constant: a hex string with an odd number of digits");
		}
		std::vector<uint8_t> raw((close - 3) / 2);
		for (size_t i = 0; i < raw.size(); ++i) {
			char hx[3] = { body[3 + 2 * i], body[4 + 2 * i], 0 };
			char *x = nullptr;
			raw[i] = uint8_t(std::strtoul(hx, &x, 16));
			if (*x != '\0') {
				return L.fail("stablehlo.constant: a hex string with a digit that is not hex");
			}
		}
		if (raw.size() != esz && raw.size() != size_t(n) * esz) {
			return L.fail("stablehlo.constant: " + std::to_string(raw.size()) + " hex bytes for " + std::to_string(n) + " elements");
		}
		for (size_t i = 0; i < raw.size(); i += esz) {
			if (ty.type == GGML_TYPE_F32) {
				float f = 0.0f;
				std::memcpy(&f, raw.data() + i, 4);
				values.push_back(double(f));
			} else if (ty.type == GGML_TYPE_I32) {
				int32_t v = 0;
				std::memcpy(&v, raw.data() + i, 4);
				values.push_back(double(v));
			} else if (ty.type == GGML_TYPE_BF16) {
				ggml_bf16_t b;
				std::memcpy(&b, raw.data() + i, 2);
				values.push_back(double(ggml_bf16_to_fp32(b)));
			} else {
				values.push_back(raw[i] != 0 ? 1.0 : 0.0);
			}
		}
	} else if (body.compare(0, 2, "0x") == 0) {
		// StableHLO writes an infinity or a NaN as its hex bit pattern: one value, for every element.
		const size_t digits = ty.type == GGML_TYPE_F32 ? 8 : 4;
		if (!is_float(ty.type) || body.size() != 2 + digits) {
			return L.unsupported("stablehlo.constant with a hex literal that is not one float");
		}
		const uint32_t bits = uint32_t(std::strtoul(body.c_str() + 2, nullptr, 16));
		const uint32_t f32_bits = ty.type == GGML_TYPE_F32 ? bits : bits << 16;
		float f = 0.0f;
		std::memcpy(&f, &f32_bits, 4);
		values.push_back(double(f));
	} else {
		if (body.find('"') != std::string::npos || body.find("0x") != std::string::npos || body.find("inf") != std::string::npos || body.find("nan") != std::string::npos) {
			return L.unsupported("stablehlo.constant with a hex, string, inf or nan literal");
		}
		size_t p = 0;
		while (p < body.size()) {
			const char c = body[p];
			if (c == '[' || c == ']' || c == ',' || c == ' ') {
				++p;
				continue;
			}
			if (ty.type == kPred && (body.compare(p, 4, "true") == 0 || body.compare(p, 5, "false") == 0)) {
				const bool t = body[p] == 't';
				values.push_back(t ? 1.0 : 0.0);
				p += t ? 4 : 5;
				continue;
			}
			char *end = nullptr;
			const double v = std::strtod(body.c_str() + p, &end);
			if (end == body.c_str() + p || ty.type == kPred) {
				return L.fail("stablehlo.constant: cannot read '" + body.substr(p, 12) + "'");
			}
			if (ty.type == GGML_TYPE_I32 && (v != std::floor(v) || v < -2147483648.0 || v > 2147483647.0)) {
				return L.fail("stablehlo.constant: '" + body.substr(p, 12) + "' is not an i32");
			}
			values.push_back(h_round(ty.type, v));
			p = size_t(end - body.c_str());
		}
	}
	if (values.size() != 1 && int64_t(values.size()) != n) {
		return L.fail("stablehlo.constant: " + std::to_string(values.size()) + " values for " + std::to_string(n) + " elements");
	}
	std::vector<double> hv(static_cast<size_t>(n));
	for (int64_t i = 0; i < n; ++i) {
		hv[size_t(i)] = values.size() == 1 ? values[0] : values[size_t(i)];
	}
	out = known_val(L, ty.type, ty.shape, std::move(hv));
	return true;
}

// Operands "%a, %b" at the start of s; returns the text after them.
std::vector<std::string> operands(const std::string &s, size_t &end) {
	std::vector<std::string> o;
	size_t p = 0;
	while (p < s.size() && s[p] == '%') {
		size_t q = p + 1;
		while (q < s.size() && (std::isalnum(static_cast<unsigned char>(s[q])) || s[q] == '_' || s[q] == '.' || s[q] == '-')) {
			++q;
		}
		o.push_back(s.substr(p, q - p));
		p = q;
		if (p + 1 < s.size() && s[p] == ',' && s[p + 1] == ' ' && p + 2 < s.size() && s[p + 2] == '%') {
			p += 2;
		} else {
			break;
		}
	}
	end = p;
	return o;
}

// A value the device can hold: a float, or an i1 as an f32 tensor of 0 and 1. A known i1 gets its tensor here.
bool on_device(const Val &v) {
	return is_float(v.type) || v.type == kPred || v.type == GGML_TYPE_I32;
}

Val device_f32(Lowering &L, const Val &v) {
	if ((v.type == kPred || v.type == GGML_TYPE_I32) && v.t == nullptr) {
		// A whole number known at compile time, wanted on the device: its f32 constant.
		Val o = known_val(L, GGML_TYPE_F32, v.shape, v.hv);
		o.known = false;
		o.hv.clear();
		return o;
	}
	return as_type(L, v, GGML_TYPE_F32);
}

// XLA CPU's select on the reference desk (x86) returns a denormal operand as the zero of its sign (the Mac's XLA
// keeps it; where_runtime measures it). The gather copies bits, so its result is flushed: xla_max(x, x) is x with
// a denormal read as zero.
ggml_tensor *select_flush(Lowering &L, ggml_tensor *t) {
	return ggml_rd_xla_binary_op(L.ctx, t, t, GGML_RD_XLA_MAX);
}

// XLA CPU's maximum, minimum or a compare of two values computed at run time (xla_compare.h).
Val xla_binary(Lowering &L, ggml_rd_xla_binary_kind kind, const Val &a, const Val &b, ggml_type result) {
	const Val fa = device_f32(L, a);
	const Val fb = device_f32(L, b);
	Val o;
	o.t = ggml_rd_xla_binary_op(L.ctx, contiguous(L, fa.t), contiguous(L, fb.t), kind);
	o.shape = a.shape;
	o.type = GGML_TYPE_F32;
	return as_type(L, o, result);
}

// select with a predicate computed at run time: the same gather as a known one, with the row index computed on
// the device as iota + N * (1 - p), exact for N up to 2^24, and converted to i32.
bool select_runtime(Lowering &L, const Val &p, const Val &a, const Val &b, Val &out) {
	const int64_t n = count_of(a.shape);
	if (n > (int64_t(1) << 23)) {
		return L.unsupported("stablehlo.select with a run-time predicate over more than 2^23 elements");
	}
	Val pv = device_f32(L, p);
	ggml_tensor *pt = contiguous(L, pv.t);
	if (p.shape.empty() && !a.shape.empty()) {
		pt = ggml_repeat_4d(L.ctx, ggml_reshape_1d(L.ctx, pt, 1), n, 1, 1, 1);
	}
	pt = ggml_reshape_1d(L.ctx, pt, n);
	std::vector<double> iota(static_cast<size_t>(n));
	for (int64_t i = 0; i < n; ++i) {
		iota[size_t(i)] = double(i);
	}
	const Val base = known_val(L, GGML_TYPE_F32, { n }, iota);
	ggml_tensor *offs = ggml_scale(L.ctx, ggml_sub(L.ctx, filled_like(L, pt, 1.0f), pt), float(n));
	ggml_tensor *idx = ggml_cast(L.ctx, ggml_add(L.ctx, base.t, offs), GGML_TYPE_I32);
	Val fa = device_f32(L, a);
	Val fb = device_f32(L, b);
	ggml_tensor *ra = ggml_reshape_2d(L.ctx, contiguous(L, fa.t), 1, n);
	ggml_tensor *rb = ggml_reshape_2d(L.ctx, contiguous(L, fb.t), 1, n);
	Val o;
	o.t = select_flush(L, ggml_get_rows(L.ctx, ggml_concat(L.ctx, ra, rb, 1), idx));
	o.shape = { n, 1 };
	o.type = GGML_TYPE_F32;
	out = as_type(L, reshaped(L, o, a.shape), a.type);
	return true;
}

// Element-wise ops on values known at compile time that are i32 or i1. Float arithmetic is never folded: it
// runs on the device, so its rounding is the device's.
bool host_binary(Lowering &L, const std::string &op, const Val &a, const Val &b, Val &out) {
	if (!a.known || !b.known) {
		return L.unsupported("stablehlo." + op + " of i32 or i1 values computed at run time");
	}
	if (a.type != b.type) {
		return L.fail("stablehlo." + op + ": operand types differ");
	}
	const bool pred = a.type == kPred;
	std::vector<double> hv(a.hv.size());
	for (size_t i = 0; i < hv.size(); ++i) {
		const int64_t x = int64_t(a.hv[i]);
		const int64_t y = int64_t(b.hv[i]);
		int64_t r = 0;
		if (op == "add" && !pred) {
			r = x + y;
		} else if (op == "subtract" && !pred) {
			r = x - y;
		} else if (op == "multiply" && !pred) {
			r = int64_t(int32_t(uint32_t(uint64_t(x) * uint64_t(y))));
		} else if (op == "maximum") {
			r = x > y ? x : y;
		} else if (op == "minimum") {
			r = x < y ? x : y;
		} else if (op == "and") {
			r = x & y;
		} else if (op == "or") {
			r = x | y;
		} else if (op == "xor") {
			r = x ^ y;
		} else {
			return L.unsupported("stablehlo." + op + " of " + (pred ? "i1" : "i32") + " values");
		}
		hv[i] = pred ? double(r & 1) : h_wrap32(r);
	}
	out = known_val(L, a.type, a.shape, std::move(hv));
	return true;
}

bool host_unary(Lowering &L, const std::string &op, const Val &a, Val &out) {
	if (!a.known) {
		return L.unsupported("stablehlo." + op + " of i32 or i1 values computed at run time");
	}
	const bool pred = a.type == kPred;
	std::vector<double> hv(a.hv.size());
	for (size_t i = 0; i < hv.size(); ++i) {
		const int64_t x = int64_t(a.hv[i]);
		if (op == "not") {
			hv[i] = pred ? double(1 - x) : h_wrap32(~x);
		} else if (op == "negate" && !pred) {
			hv[i] = h_wrap32(-x);
		} else if (op == "abs" && !pred) {
			hv[i] = h_wrap32(x < 0 ? -x : x);
		} else {
			return L.unsupported("stablehlo." + op + " of " + (pred ? "i1" : "i32") + " values");
		}
	}
	out = known_val(L, a.type, a.shape, std::move(hv));
	return true;
}

// stablehlo.compare of two values known at compile time.
bool host_compare(Lowering &L, const std::string &dir, const std::string &kind, const Val &a, const Val &b, Val &out) {
	if (a.type != b.type || a.shape != b.shape) {
		return L.fail("stablehlo.compare: operands differ in type or shape");
	}
	if (!a.known || !b.known) {
		if (kind == "TOTALORDER" || kind == "UNSIGNED") {
			return L.unsupported("stablehlo.compare of values computed at run time with TOTALORDER or UNSIGNED");
		}
		static const char *const kDirs[] = { "GT", "GE", "LT", "LE", "EQ", "NE" };
		static const ggml_rd_xla_binary_kind kKinds[] = { GGML_RD_XLA_GT, GGML_RD_XLA_GE, GGML_RD_XLA_LT, GGML_RD_XLA_LE, GGML_RD_XLA_EQ, GGML_RD_XLA_NE };
		for (int k = 0; k < 6; ++k) {
			if (dir == kDirs[k]) {
				out = xla_binary(L, kKinds[k], a, b, kPred);
				return true;
			}
		}
		return L.fail("stablehlo.compare: unknown direction '" + dir + "'");
	}
	if (kind == "TOTALORDER") {
		return L.unsupported("stablehlo.compare with TOTALORDER");
	}
	const bool uns = kind == "UNSIGNED";
	std::vector<double> hv(a.hv.size());
	for (size_t i = 0; i < hv.size(); ++i) {
		double x = a.hv[i];
		double y = b.hv[i];
		if (uns) {
			x = double(uint32_t(int32_t(x)));
			y = double(uint32_t(int32_t(y)));
		}
		bool r = false;
		if (dir == "EQ") {
			r = x == y;
		} else if (dir == "NE") {
			r = x != y;
		} else if (dir == "LT") {
			r = x < y;
		} else if (dir == "LE") {
			r = x <= y;
		} else if (dir == "GT") {
			r = x > y;
		} else if (dir == "GE") {
			r = x >= y;
		} else {
			return L.fail("stablehlo.compare: unknown direction '" + dir + "'");
		}
		hv[i] = r ? 1.0 : 0.0;
	}
	out = known_val(L, kPred, a.shape, std::move(hv));
	return true;
}

// A float truncated toward zero, as whole numbers in f32: floor(|x|) with x's sign.
ggml_tensor *truncated(Lowering &L, ggml_tensor *t) {
	ggml_tensor *below = ggml_rd_xla_binary_op(L.ctx, contiguous(L, t), filled(L, {}, 0.0f).t, GGML_RD_XLA_LT);
	return ggml_mul(L.ctx, ggml_floor(L.ctx, ggml_abs(L.ctx, t)), ggml_scale_bias(L.ctx, below, -2.0f, 1.0f));
}

bool op_convert(Lowering &L, const Val &x, const Ty &to, Val &out) {
	if (is_float(x.type) && is_float(to.type)) {
		out = as_type(L, x, to.type);
		return true;
	}
	if (x.t != nullptr && !x.known) {
		// A value computed at run time. Whole numbers are f32 tensors, so most conversions only change the label.
		Val f = as_type(L, x, GGML_TYPE_F32);
		if (to.type == kPred) {
			out = xla_binary(L, GGML_RD_XLA_NE, f, filled(L, x.shape, 0.0f), kPred);
		} else if (to.type == GGML_TYPE_I32 && is_float(x.type)) {
			// Exact for |x| below 2^24; a NaN or a larger value is not what XLA's cvttss2si gives.
			f.t = truncated(L, f.t);
			out = as_type(L, f, GGML_TYPE_I32);
		} else {
			out = as_type(L, f, to.type);
		}
		out.known = false;
		out.hv.clear();
		return true;
	}
	if (!x.known) {
		return L.unsupported("stablehlo.convert of this value");
	}
	std::vector<double> hv(x.hv.size());
	for (size_t i = 0; i < hv.size(); ++i) {
		const double v = x.hv[i];
		if (to.type == kPred) {
			hv[i] = v != 0.0 ? 1.0 : 0.0;
		} else if (to.type == GGML_TYPE_I32) {
			if (std::isnan(v) || v <= -2147483649.0 || v >= 2147483648.0) {
				return L.unsupported("stablehlo.convert of a NaN or out-of-range float to i32");
			}
			hv[i] = std::trunc(v);
		} else {
			hv[i] = h_round(to.type, v);
		}
	}
	out = known_val(L, to.type, x.shape, std::move(hv));
	return true;
}

bool op_iota(Lowering &L, const std::string &attrs, const Ty &to, Val &out) {
	const size_t d = attrs.find("dim = ");
	if (d == std::string::npos) {
		return L.fail("stablehlo.iota: no dim");
	}
	const int64_t dim = std::strtoll(attrs.c_str() + d + 6, nullptr, 10);
	if (dim < 0 || dim >= int64_t(to.shape.size()) || to.type == kPred) {
		return L.fail("stablehlo.iota: dim is out of range, or the type is i1");
	}
	std::vector<double> hv(size_t(count_of(to.shape)));
	each_index(to.shape, [&](const std::vector<int64_t> &o, size_t flat) { hv[flat] = h_round(to.type, double(o[size_t(dim)])); });
	out = known_val(L, to.type, to.shape, std::move(hv));
	return true;
}

// stablehlo.select with a predicate known at compile time. Float operands that are not both known are picked
// element by element by a gather: both are laid out as rows of one element, b's after a's, and ggml_get_rows
// takes row i or row N + i, a copy that keeps every bit.
bool op_select(Lowering &L, const Val &p, const Val &a, const Val &b, Val &out) {
	if (p.type != kPred || a.type != b.type || a.shape != b.shape || (!p.shape.empty() && p.shape != a.shape)) {
		return L.fail("stablehlo.select: the predicate or the operands do not match");
	}
	if (!p.known) {
		if (!on_device(a)) {
			return L.unsupported("stablehlo.select of i32 values with a predicate computed at run time");
		}
		return select_runtime(L, p, a, b, out);
	}
	const size_t n = size_t(count_of(a.shape));
	bool all_true = true, all_false = true;
	for (double v : p.hv) {
		all_true = all_true && v != 0.0;
		all_false = all_false && v == 0.0;
	}
	if (a.known && b.known) {
		std::vector<double> hv(n);
		for (size_t i = 0; i < n; ++i) {
			hv[i] = (p.shape.empty() ? p.hv[0] : p.hv[i]) != 0.0 ? a.hv[i] : b.hv[i];
		}
		out = known_val(L, a.type, a.shape, std::move(hv));
		return true;
	}
	if (all_true || all_false) {
		out = all_true ? a : b;
		return true;
	}
	if (!on_device(a)) {
		return L.unsupported("stablehlo.select of i32 values computed at run time");
	}
	Val fa = device_f32(L, a);
	Val fb = device_f32(L, b);
	ggml_tensor *ra = ggml_reshape_2d(L.ctx, contiguous(L, fa.t), 1, int64_t(n));
	ggml_tensor *rb = ggml_reshape_2d(L.ctx, contiguous(L, fb.t), 1, int64_t(n));
	ggml_tensor *both = ggml_concat(L.ctx, ra, rb, 1);
	ggml_tensor *idx = ggml_new_tensor_1d(L.ctx, GGML_TYPE_I32, int64_t(n));
	std::vector<uint8_t> bytes(n * 4);
	for (size_t i = 0; i < n; ++i) {
		const int32_t at = int32_t(p.hv[i] != 0.0 ? i : n + i);
		std::memcpy(bytes.data() + i * 4, &at, 4);
	}
	L.consts[idx] = std::move(bytes);
	Val o;
	o.t = select_flush(L, ggml_get_rows(L.ctx, both, idx));
	o.shape = { int64_t(n), 1 };
	o.type = GGML_TYPE_F32;
	out = as_type(L, reshaped(L, o, a.shape), a.type);
	return true;
}

// --- gather, slices by run-time starts, updates ---------------------------------------------------------

// "key = [a, b]" or "key = array<i64: a, b>"; absent gives an empty list.
std::vector<int64_t> int_array(const std::string &s, const std::string &key) {
	std::vector<int64_t> out;
	size_t k = 0;
	while ((k = s.find(key, k)) != std::string::npos) {
		// The key as a whole word: "pad" must not match "padding".
		const bool starts = k == 0 || !(std::isalnum(static_cast<unsigned char>(s[k - 1])) || s[k - 1] == '_');
		const size_t e = k + key.size();
		if (starts && s.compare(e, 3, " = ") == 0) {
			break;
		}
		k = e;
	}
	if (k == std::string::npos) {
		return out;
	}
	size_t p = k + key.size() + 3;
	char close = ']';
	if (s.compare(p, 6, "array<") == 0) {
		close = '>';
		p = s.find(':', p);
	} else if (s[p] != '[') {
		char *e = nullptr;
		out.push_back(std::strtoll(s.c_str() + p, &e, 10));
		return out;
	}
	// A nested list ([[1, 1], [2, 2]]) is read flat, up to the bracket that closes the outer one.
	int depth = 0;
	for (++p; p < s.size(); ++p) {
		const char c = s[p];
		if (c == '[') {
			++depth;
		} else if (c == close && depth == 0) {
			break;
		} else if (c == ']') {
			--depth;
		} else if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
			char *e = nullptr;
			out.push_back(std::strtoll(s.c_str() + p, &e, 10));
			p = size_t(e - s.c_str()) - 1;
		}
	}
	return out;
}

// Indices for ggml_get_rows from whole numbers: clamped to [0, n - 1], as XLA's gather clamps a start index.
ggml_tensor *row_index(Lowering &L, const Val &idx, int64_t n) {
	const int64_t m = count_of(idx.shape);
	if (idx.t == nullptr) {
		ggml_tensor *t = ggml_new_tensor_1d(L.ctx, GGML_TYPE_I32, m);
		std::vector<uint8_t> bytes(size_t(m) * 4);
		for (int64_t i = 0; i < m; ++i) {
			const int32_t at = int32_t(std::min<double>(std::max<double>(idx.hv[size_t(i)], 0.0), double(n - 1)));
			std::memcpy(bytes.data() + size_t(i) * 4, &at, 4);
		}
		L.consts[t] = std::move(bytes);
		return t;
	}
	Val f = reshaped(L, as_type(L, idx, GGML_TYPE_F32), { m });
	ggml_tensor *c = xla_against(L, GGML_RD_XLA_MIN, xla_against(L, GGML_RD_XLA_MAX, f.t, 0.0f), float(n - 1));
	return ggml_cast(L.ctx, c, GGML_TYPE_I32);
}

// x with dimension `axis` replaced by the elements idx names (m of them): rows of ggml_get_rows, a 32-bit copy.
bool gather_axis(Lowering &L, const Val &x, size_t axis, ggml_tensor *idx, int64_t m, Val &out) {
	const size_t r = x.shape.size();
	std::vector<int64_t> perm, rest, back(r);
	perm.push_back(int64_t(axis));
	for (size_t i = 0; i < r; ++i) {
		if (i != axis) {
			perm.push_back(int64_t(i));
			rest.push_back(x.shape[i]);
		}
	}
	for (size_t i = 0; i < r; ++i) {
		back[i] = i == axis ? 0 : int64_t(i < axis ? i + 1 : i);
	}
	Val rows = reshaped(L, transposed(L, as_type(L, x, GGML_TYPE_F32), perm), { x.shape[axis], count_of(rest) });
	Val g;
	g.t = ggml_get_rows(L.ctx, contiguous(L, rows.t), idx);
	g.type = GGML_TYPE_F32;
	g.shape = { m, count_of(rest) };
	std::vector<int64_t> shape;
	shape.push_back(m);
	shape.insert(shape.end(), rest.begin(), rest.end());
	size_t wide = 0;
	for (int64_t d : shape) {
		wide += d > 1 ? 1 : 0;
	}
	if (wide > GGML_MAX_DIMS) {
		return L.unsupported("a gather whose result has more than 4 dimensions of size above one");
	}
	out = as_type(L, transposed(L, reshaped(L, g, shape), back), x.type);
	out.known = false;
	out.hv.clear();
	return true;
}

// stablehlo.gather in the forms JAX writes for x[i], take and x[i, j]: every indexed dimension is collapsed
// (slice size 1), the others are taken whole, and the index vector is the last dimension of the indices.
bool op_gather(Lowering &L, const std::string &attrs, const Val &x, const Val &indices, const Ty &to, Val &out) {
	const std::vector<int64_t> offset = int_array(attrs, "offset_dims");
	const std::vector<int64_t> collapsed = int_array(attrs, "collapsed_slice_dims");
	const std::vector<int64_t> map = int_array(attrs, "start_index_map");
	const std::vector<int64_t> ivd = int_array(attrs, "index_vector_dim");
	const std::vector<int64_t> sizes = int_array(attrs, "slice_sizes");
	const size_t r = x.shape.size();
	if (sizes.size() != r || ivd.size() != 1 || map.empty() || indices.shape.empty()) {
		return L.fail("stablehlo.gather: slice_sizes, index_vector_dim or start_index_map do not fit the operand");
	}
	if (!int_array(attrs, "operand_batching_dims").empty() || !int_array(attrs, "start_indices_batching_dims").empty()) {
		return L.unsupported("stablehlo.gather with batching dimensions");
	}
	if (indices.type != GGML_TYPE_I32 || x.t == nullptr) {
		return L.unsupported("stablehlo.gather with indices that are not i32, or of i32 or i1 values known at compile time");
	}
	if (ivd[0] != int64_t(indices.shape.size()) - 1 || indices.shape.back() != int64_t(map.size())) {
		return L.unsupported("stablehlo.gather whose index vector is not the last dimension of the indices");
	}
	std::vector<bool> indexed(r, false);
	for (int64_t d : map) {
		if (d < 0 || d >= int64_t(r) || indexed[size_t(d)]) {
			return L.fail("stablehlo.gather: start_index_map is out of range or repeats a dimension");
		}
		indexed[size_t(d)] = true;
	}
	std::vector<int64_t> sorted_map = map, sorted_collapsed = collapsed;
	std::sort(sorted_map.begin(), sorted_map.end());
	std::sort(sorted_collapsed.begin(), sorted_collapsed.end());
	if (sorted_map != sorted_collapsed) {
		return L.unsupported("stablehlo.gather with an indexed dimension that is not collapsed, or a collapsed one that is not indexed");
	}
	for (size_t d = 0; d < r; ++d) {
		if (sizes[d] != (indexed[d] ? 1 : x.shape[d])) {
			return L.unsupported("stablehlo.gather with a slice that is a window (not one element of an indexed dimension and all of the others)");
		}
	}
	std::vector<int64_t> batch(indices.shape.begin(), indices.shape.end() - 1);
	const int64_t m = count_of(batch);
	// The indexed dimensions first, in the order of the index vector, flattened to one: its row is the linear index.
	std::vector<int64_t> perm = map, rest;
	int64_t rows = 1;
	for (int64_t d : map) {
		rows *= x.shape[size_t(d)];
	}
	for (size_t d = 0; d < r; ++d) {
		if (!indexed[d]) {
			perm.push_back(int64_t(d));
			rest.push_back(x.shape[d]);
		}
	}
	if (rows > (int64_t(1) << 24)) {
		return L.unsupported("stablehlo.gather over more than 2^24 rows");
	}
	Val linear;
	linear.type = GGML_TYPE_I32;
	linear.shape = { m };
	if (indices.t == nullptr) {
		linear.known = true;
		linear.hv.assign(size_t(m), 0.0);
		for (int64_t i = 0; i < m; ++i) {
			double at = 0.0;
			for (size_t j = 0; j < map.size(); ++j) {
				const double n = double(x.shape[size_t(map[j])]);
				at = at * n + std::min(std::max(indices.hv[size_t(i) * map.size() + j], 0.0), n - 1.0);
			}
			linear.hv[size_t(i)] = at;
		}
	} else if (map.size() == 1) {
		linear = reshaped(L, indices, { m });
	} else {
		const Val table = reshaped(L, as_type(L, indices, GGML_TYPE_F32), { m, int64_t(map.size()) });
		ggml_tensor *acc = nullptr;
		for (size_t j = 0; j < map.size(); ++j) {
			Val col;
			if (!slice_dim(L, table, 1, int64_t(j), 1, col)) {
				return false;
			}
			const float n = float(x.shape[size_t(map[j])]);
			ggml_tensor *c = xla_against(L, GGML_RD_XLA_MIN, xla_against(L, GGML_RD_XLA_MAX, ggml_reshape_1d(L.ctx, contiguous(L, col.t), m), 0.0f), n - 1.0f);
			acc = acc == nullptr ? c : ggml_add(L.ctx, ggml_scale(L.ctx, acc, n), c);
		}
		linear.t = acc;
	}
	std::vector<int64_t> flat;
	flat.push_back(rows);
	flat.insert(flat.end(), rest.begin(), rest.end());
	const Val table = reshaped(L, transposed(L, x, perm), flat);
	Val g;
	if (!gather_axis(L, table, 0, row_index(L, linear, rows), m, g)) {
		return false;
	}
	// g is [m, rest...]; the result has the rest at offset_dims and the batch dimensions of the indices elsewhere.
	if (offset.size() != rest.size()) {
		return L.fail("stablehlo.gather: offset_dims does not name every dimension that is kept");
	}
	std::vector<int64_t> now = batch;
	now.insert(now.end(), rest.begin(), rest.end());
	const size_t rr = now.size();
	std::vector<int64_t> out_perm(rr, -1);
	for (size_t j = 0; j < offset.size(); ++j) {
		if (offset[j] < 0 || offset[j] >= int64_t(rr) || out_perm[size_t(offset[j])] != -1) {
			return L.fail("stablehlo.gather: offset_dims is out of range or repeats a dimension");
		}
		out_perm[size_t(offset[j])] = int64_t(batch.size() + j);
	}
	int64_t next_batch = 0;
	for (size_t i = 0; i < rr; ++i) {
		if (out_perm[i] == -1) {
			out_perm[i] = next_batch++;
		}
	}
	size_t wide = 0;
	for (int64_t d : now) {
		wide += d > 1 ? 1 : 0;
	}
	if (wide > GGML_MAX_DIMS) {
		return L.unsupported("stablehlo.gather whose result has more than 4 dimensions of size above one");
	}
	out = transposed(L, reshaped(L, g, now), out_perm);
	(void)to;
	return true;
}

// stablehlo.clamp: min(max(x, lo), hi) with XLA's maximum and minimum.
bool op_clamp(Lowering &L, const Val &lo, const Val &x, const Val &hi, Val &out) {
	if (lo.type != x.type || hi.type != x.type || (!lo.shape.empty() && lo.shape != x.shape) || (!hi.shape.empty() && hi.shape != x.shape)) {
		return L.fail("stablehlo.clamp: the bounds do not match the operand");
	}
	if (lo.known && x.known && hi.known && x.t == nullptr) {
		std::vector<double> hv(x.hv.size());
		for (size_t i = 0; i < hv.size(); ++i) {
			hv[i] = std::min(std::max(x.hv[i], lo.hv[lo.shape.empty() ? 0 : i]), hi.hv[hi.shape.empty() ? 0 : i]);
		}
		out = known_val(L, x.type, x.shape, std::move(hv));
		return true;
	}
	if (!on_device(x) || x.type == kPred) {
		return L.unsupported("stablehlo.clamp of this element type");
	}
	const Val fx = device_f32(L, x);
	const Val fl = device_f32(L, lo);
	const Val fh = device_f32(L, hi);
	Val o;
	o.t = ggml_rd_xla_binary_op(L.ctx, contiguous(L, fx.t), contiguous(L, fl.t), GGML_RD_XLA_MAX);
	o.t = ggml_rd_xla_binary_op(L.ctx, o.t, contiguous(L, fh.t), GGML_RD_XLA_MIN);
	o.shape = x.shape;
	o.type = GGML_TYPE_F32;
	out = as_type(L, o, x.type);
	return true;
}

// The start of a dynamic slice or update: one i32 vector, or one i32 scalar for each dimension. False when a
// start is computed at run time; then `runtime` holds the operands for the caller to refuse or gather by.
bool known_starts(const std::vector<Val> &in, size_t first, size_t rank, std::vector<int64_t> &start) {
	start.clear();
	if (in.size() == first + 1 && in[first].shape.size() == 1 && in[first].shape[0] == int64_t(rank)) {
		if (!in[first].known || in[first].t != nullptr) {
			return false;
		}
		for (double v : in[first].hv) {
			start.push_back(int64_t(v));
		}
		return true;
	}
	for (size_t k = first; k < in.size(); ++k) {
		if (!in[k].known || in[k].t != nullptr || !in[k].shape.empty()) {
			return false;
		}
		start.push_back(int64_t(in[k].hv[0]));
	}
	return start.size() == rank;
}

// stablehlo.dynamic_slice. A start known at compile time is a slice; one computed at run time gathers the
// elements start .. start + size - 1 of each dimension that is not taken whole. XLA clamps the start so the
// slice fits, and so does this.
bool op_dynamic_slice(Lowering &L, const std::string &attrs, const std::vector<Val> &in, Val &out) {
	const Val &x = in[0];
	const size_t r = x.shape.size();
	const std::vector<int64_t> sizes = int_array(attrs, "sizes");
	if (sizes.size() != r || x.t == nullptr) {
		return sizes.size() != r ? L.fail("stablehlo.dynamic_slice: sizes do not match the rank") : L.unsupported("stablehlo.dynamic_slice of i32 or i1 values known at compile time");
	}
	for (size_t d = 0; d < r; ++d) {
		if (sizes[d] < 1 || sizes[d] > x.shape[d]) {
			return L.fail("stablehlo.dynamic_slice: a size is outside its dimension");
		}
	}
	std::vector<int64_t> start;
	Val acc = x;
	if (known_starts(in, 1, r, start)) {
		for (size_t d = 0; d < r; ++d) {
			const int64_t at = std::min(std::max<int64_t>(start[d], 0), x.shape[d] - sizes[d]);
			if (sizes[d] != x.shape[d]) {
				Val next;
				if (!slice_dim(L, acc, d, at, sizes[d], next)) {
					return false;
				}
				acc = next;
			}
		}
		out = acc;
		return true;
	}
	const bool vector = in.size() == 2 && in[1].shape.size() == 1 && in[1].shape[0] == int64_t(r);
	if (!vector && in.size() != 1 + r) {
		return L.fail("stablehlo.dynamic_slice: the start indices do not match the rank");
	}
	for (size_t d = 0; d < r; ++d) {
		if (sizes[d] == x.shape[d]) {
			continue;
		}
		Val st;
		if (vector) {
			if (in[1].t == nullptr) {
				st = known_val(L, GGML_TYPE_I32, {}, { in[1].hv[d] });
			} else if (!slice_dim(L, in[1], 0, int64_t(d), 1, st)) {
				return false;
			}
		} else {
			st = in[1 + d];
		}
		if (st.type != GGML_TYPE_I32) {
			return L.fail("stablehlo.dynamic_slice: a start index is not i32");
		}
		ggml_tensor *s1 = ggml_reshape_1d(L.ctx, contiguous(L, device_f32(L, st).t), 1);
		s1 = xla_against(L, GGML_RD_XLA_MIN, xla_against(L, GGML_RD_XLA_MAX, s1, 0.0f), float(x.shape[d] - sizes[d]));
		std::vector<double> ramp(static_cast<size_t>(sizes[d]));
		for (size_t i = 0; i < ramp.size(); ++i) {
			ramp[i] = double(i);
		}
		Val base = known_val(L, GGML_TYPE_F32, { sizes[d] }, ramp);
		ggml_tensor *idx = ggml_cast(L.ctx, ggml_add(L.ctx, base.t, s1), GGML_TYPE_I32);
		Val next;
		if (!gather_axis(L, acc, d, idx, sizes[d], next)) {
			return false;
		}
		acc = next;
	}
	out = acc;
	out.known = false;
	out.hv.clear();
	return true;
}

// x with the block `update` written at `start`: a concatenation of slices of x around the update, one dimension
// at a time, so every element keeps its bits.
bool paste(Lowering &L, const Val &x, const Val &update, const std::vector<int64_t> &start, size_t dim, Val &out) {
	if (dim == x.shape.size()) {
		out = update;
		return true;
	}
	const int64_t n = x.shape[dim], h = update.shape[dim], s0 = start[dim];
	if (s0 == 0 && h == n) {
		return paste(L, x, update, start, dim + 1, out);
	}
	Val mid_x, mid;
	if (!slice_dim(L, x, dim, s0, h, mid_x) || !paste(L, mid_x, update, start, dim + 1, mid)) {
		return false;
	}
	std::vector<Val> parts;
	Val before, after;
	if (s0 > 0) {
		if (!slice_dim(L, x, dim, 0, s0, before)) {
			return false;
		}
		parts.push_back(before);
	}
	parts.push_back(mid);
	if (s0 + h < n) {
		if (!slice_dim(L, x, dim, s0 + h, n - s0 - h, after)) {
			return false;
		}
		parts.push_back(after);
	}
	return concat_dim(L, parts, dim, out);
}

bool op_update_slice(Lowering &L, const char *what, const Val &x, const Val &update, const std::vector<int64_t> &start, Val &out) {
	const size_t r = x.shape.size();
	if (update.shape.size() != r || update.type != x.type || start.size() != r) {
		return L.fail(std::string(what) + ": the update does not match the operand");
	}
	if (x.t == nullptr || update.t == nullptr) {
		return L.unsupported(std::string(what) + " of i32 or i1 values known at compile time");
	}
	std::vector<int64_t> at = start;
	for (size_t d = 0; d < r; ++d) {
		if (update.shape[d] > x.shape[d]) {
			return L.fail(std::string(what) + ": the update is larger than the operand");
		}
		at[d] = std::min(std::max<int64_t>(start[d], 0), x.shape[d] - update.shape[d]);
	}
	if (!paste(L, x, update, at, 0, out)) {
		return false;
	}
	out.known = false;
	out.hv.clear();
	out.neg_inf = false;
	out.pre = nullptr;
	return true;
}

// stablehlo.scatter in the form JAX writes for x.at[a:b, c:d].set(y): one index vector known at compile time,
// every dimension of the update a window dimension, and a body that returns the update.
bool op_scatter(Lowering &L, const std::string &attrs, const Val &x, const Val &indices, const Val &update, Val &out) {
	const size_t r = x.shape.size();
	const std::vector<int64_t> window = int_array(attrs, "update_window_dims");
	const std::vector<int64_t> to_operand = int_array(attrs, "scatter_dims_to_operand_dims");
	const std::vector<int64_t> ivd = int_array(attrs, "index_vector_dim");
	const size_t ret = attrs.find("stablehlo.return ");
	const size_t bb = attrs.find("^bb0(");
	if (ret == std::string::npos || bb == std::string::npos) {
		return L.fail("stablehlo.scatter: no body");
	}
	// The body's second block argument is the update; returning it is a plain write.
	const size_t comma = attrs.find(", %", bb);
	const size_t colon = comma == std::string::npos ? std::string::npos : attrs.find(':', comma);
	const std::string second = colon == std::string::npos ? std::string() : attrs.substr(comma + 2, colon - comma - 2);
	size_t e = ret + 17;
	while (e < attrs.size() && attrs[e] != ' ' && attrs[e] != ':') {
		++e;
	}
	if (second.empty() || attrs.substr(ret + 17, e - ret - 17) != second) {
		return L.unsupported("stablehlo.scatter with a body other than a write of the update (an add or another combine)");
	}
	if (!int_array(attrs, "inserted_window_dims").empty() || !int_array(attrs, "input_batching_dims").empty() || window.size() != r || update.shape.size() != r) {
		return L.unsupported("stablehlo.scatter whose update is not one window of the operand's rank");
	}
	if (!indices.known || indices.t != nullptr || indices.shape.size() != 1 || (!ivd.empty() && ivd[0] != 0) || indices.shape[0] != int64_t(to_operand.size())) {
		return L.unsupported("stablehlo.scatter with indices computed at run time, or more than one index vector");
	}
	std::vector<int64_t> start(r, 0);
	for (size_t k = 0; k < to_operand.size(); ++k) {
		if (to_operand[k] < 0 || to_operand[k] >= int64_t(r)) {
			return L.fail("stablehlo.scatter: scatter_dims_to_operand_dims is out of range");
		}
		start[size_t(to_operand[k])] = int64_t(indices.hv[k]);
	}
	for (size_t d = 0; d < r; ++d) {
		if (start[d] < 0 || start[d] + update.shape[d] > x.shape[d]) {
			// StableHLO drops a window that does not fit; XLA's result for it is not measured here.
			return L.unsupported("stablehlo.scatter with a window outside the operand");
		}
	}
	return op_update_slice(L, "stablehlo.scatter", x, update, start, out);
}

// --- windows ------------------------------------------------------------------------------------------

// stablehlo.reduce_window applying maximum, minimum or add, with strides and padding and no dilation. Each
// element of the window is a strided slice, and the slices are combined one after another in the window's
// row-major order, so no element is rounded more often than XLA's loop over the window rounds it.
bool op_reduce_window(Lowering &L, const std::string &attrs, const Val &x, const Val &init, const Ty &to, Val &out) {
	const size_t r = x.shape.size();
	const std::vector<int64_t> win = int_array(attrs, "window_dimensions");
	std::vector<int64_t> strides = int_array(attrs, "window_strides");
	const std::vector<int64_t> base_d = int_array(attrs, "base_dilations");
	const std::vector<int64_t> win_d = int_array(attrs, "window_dilations");
	std::vector<int64_t> padding;
	const size_t pk = attrs.find("padding = dense<");
	if (pk != std::string::npos) {
		const size_t pe = attrs.find('>', pk);
		const std::string body = attrs.substr(pk + 16, pe - pk - 16);
		if (body.find('[') == std::string::npos) {
			padding.assign(r * 2, std::strtoll(body.c_str(), nullptr, 10));
		} else {
			padding = int_array("p = " + body, "p");
		}
	}
	if (padding.empty()) {
		padding.assign(r * 2, 0);
	}
	if (strides.empty()) {
		strides.assign(r, 1);
	}
	if (win.size() != r || strides.size() != r || padding.size() != r * 2) {
		return L.fail("stablehlo.reduce_window: the window, strides or padding do not match the rank");
	}
	for (int64_t d : base_d) {
		if (d != 1) {
			return L.unsupported("stablehlo.reduce_window with base dilation");
		}
	}
	for (int64_t d : win_d) {
		if (d != 1) {
			return L.unsupported("stablehlo.reduce_window with window dilation");
		}
	}
	const size_t body = attrs.find("^bb0(");
	const bool is_max = body != std::string::npos && attrs.find("stablehlo.maximum", body) != std::string::npos;
	const bool is_min = body != std::string::npos && attrs.find("stablehlo.minimum", body) != std::string::npos;
	const bool is_add = body != std::string::npos && attrs.find("stablehlo.add", body) != std::string::npos;
	if (int(is_max) + int(is_min) + int(is_add) != 1) {
		return L.unsupported("stablehlo.reduce_window with a body other than one maximum, minimum or add");
	}
	if (!is_float(x.type) || x.t == nullptr || r > GGML_MAX_DIMS) {
		return L.unsupported("stablehlo.reduce_window of values that are not floats, or above rank 4");
	}
	if (!init.known || count_of(init.shape) != 1) {
		return L.unsupported("stablehlo.reduce_window with an init that is not a scalar constant");
	}
	const double iv = init.hv[0];
	if (!(is_add ? (iv == 0.0 && !std::signbit(iv)) : (std::isinf(iv) && (is_max ? iv < 0.0 : iv > 0.0)))) {
		return L.unsupported("stablehlo.reduce_window whose init is not the identity of its body");
	}
	std::vector<int64_t> low(r), high(r), count(r);
	int64_t elements = 1;
	for (size_t d = 0; d < r; ++d) {
		low[d] = padding[d * 2];
		high[d] = padding[d * 2 + 1];
		if (low[d] < 0 || high[d] < 0 || win[d] < 1 || strides[d] < 1) {
			return L.unsupported("stablehlo.reduce_window with negative padding");
		}
		const int64_t n = x.shape[d] + low[d] + high[d];
		if (n < win[d]) {
			return L.fail("stablehlo.reduce_window: the window is larger than the padded operand");
		}
		count[d] = (n - win[d]) / strides[d] + 1;
		elements *= win[d];
	}
	if (count != to.shape) {
		return L.fail("stablehlo.reduce_window: the result shape does not match the window");
	}
	if (elements > 256) {
		return L.unsupported("stablehlo.reduce_window with a window of more than 256 elements");
	}
	Val padded;
	if (!pad_edges(L, as_type(L, x, GGML_TYPE_F32), low, high, float(iv), padded)) {
		return false;
	}
	padded.t = contiguous(L, padded.t);
	Val acc;
	bool first = true;
	bool ok = true;
	each_index(win, [&](const std::vector<int64_t> &off, size_t) {
		if (!ok) {
			return;
		}
		std::vector<int64_t> span(r);
		for (size_t d = 0; d < r; ++d) {
			span[d] = (count[d] - 1) * strides[d] + 1;
		}
		Val piece;
		ok = slice_core(L, padded, off, strides, span, count, piece);
		if (!ok) {
			return;
		}
		if (first) {
			acc = piece;
			first = false;
		} else if (is_add) {
			acc.t = ggml_add(L.ctx, acc.t, piece.t);
		} else {
			acc.t = ggml_rd_xla_binary_op(L.ctx, acc.t, piece.t, is_max ? GGML_RD_XLA_MAX : GGML_RD_XLA_MIN);
		}
	});
	if (!ok) {
		return false;
	}
	acc.shape = count;
	acc.type = GGML_TYPE_F32;
	out = as_type(L, acc, x.type);
	return true;
}

// stablehlo.convolution with one or two spatial dimensions: strides, padding, kernel (rhs) dilation, input (lhs)
// dilation as JAX writes a transposed convolution, and feature groups. The operands are transposed to ggml's
// layout ([N, C, H, W] and [O, I, KH, KW] row-major), the input is zero-stuffed for an lhs dilation and padded
// explicitly, and the product is ggml's CONV_2D (CONV_2D_DW for a depthwise kernel).
bool op_convolution(Lowering &L, const std::string &attrs, const Val &lhs, const Val &rhs, const Ty &to, Val &out) {
	const size_t dn = attrs.find("dim_numbers = ");
	const size_t wn = attrs.find("window = {");
	if (dn == std::string::npos) {
		return L.fail("stablehlo.convolution: no dim_numbers");
	}
	// "[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]": three lists of tokens.
	std::vector<std::vector<std::string>> lists;
	size_t p = dn + 14;
	while (lists.size() < 3) {
		const size_t lb = attrs.find('[', p);
		const size_t rb = lb == std::string::npos ? std::string::npos : attrs.find(']', lb);
		if (rb == std::string::npos) {
			return L.fail("stablehlo.convolution: cannot read dim_numbers");
		}
		std::vector<std::string> toks;
		std::string cur;
		for (size_t i = lb + 1; i <= rb; ++i) {
			if (attrs[i] == ',' || attrs[i] == ']') {
				toks.push_back(trim(cur));
				cur.clear();
			} else {
				cur += attrs[i];
			}
		}
		lists.push_back(toks);
		p = rb + 1;
	}
	const size_t rank = lhs.shape.size();
	const size_t ns = rank < 2 ? 0 : rank - 2;
	if (lists[0].size() != rank || lists[1].size() != rank || lists[2].size() != rank || rhs.shape.size() != rank) {
		return L.fail("stablehlo.convolution: dim_numbers do not match the operands");
	}
	if (ns < 1 || ns > 2) {
		return L.unsupported("stablehlo.convolution with " + std::to_string(ns) + " spatial dimensions (one or two)");
	}
	if (!is_float(lhs.type) || !is_float(rhs.type) || lhs.t == nullptr || rhs.t == nullptr) {
		return L.unsupported("stablehlo.convolution of values that are not floats");
	}
	// The position of each token in a list: the first, the second, then the spatial ones in order.
	std::function<bool(const std::vector<std::string> &, const char *, const char *, std::vector<int64_t> &)> order =
			[&](const std::vector<std::string> &toks, const char *first, const char *second, std::vector<int64_t> &perm) {
				perm.assign(2 + ns, -1);
				for (size_t i = 0; i < toks.size(); ++i) {
					if (toks[i] == first) {
						perm[0] = int64_t(i);
					} else if (toks[i] == second) {
						perm[1] = int64_t(i);
					} else {
						char *e = nullptr;
						const long k = std::strtol(toks[i].c_str(), &e, 10);
						if (toks[i].empty() || *e != '\0' || k < 0 || k >= long(ns)) {
							return false;
						}
						perm[2 + size_t(k)] = int64_t(i);
					}
				}
				for (int64_t v : perm) {
					if (v < 0) {
						return false;
					}
				}
				return true;
			};
	std::vector<int64_t> in_perm, k_perm, out_at;
	if (!order(lists[0], "b", "f", in_perm) || !order(lists[1], "o", "i", k_perm) || !order(lists[2], "b", "f", out_at)) {
		return L.fail("stablehlo.convolution: cannot read dim_numbers");
	}
	const std::string window = wn == std::string::npos ? std::string() : attrs.substr(wn, attrs.find('}', wn) - wn);
	std::vector<int64_t> stride = int_array(window, "stride"), pad = int_array(window, "pad"), ld = int_array(window, "lhs_dilate"), rd = int_array(window, "rhs_dilate");
	if (window.find("true") != std::string::npos) {
		return L.unsupported("stablehlo.convolution with a reversed window");
	}
	if (stride.empty()) {
		stride.assign(ns, 1);
	}
	if (pad.empty()) {
		pad.assign(ns * 2, 0);
	}
	if (ld.empty()) {
		ld.assign(ns, 1);
	}
	if (rd.empty()) {
		rd.assign(ns, 1);
	}
	if (stride.size() != ns || pad.size() != ns * 2 || ld.size() != ns || rd.size() != ns) {
		return L.fail("stablehlo.convolution: the window does not match the spatial dimensions");
	}
	const std::vector<int64_t> fg = int_array(attrs, "feature_group_count"), bg = int_array(attrs, "batch_group_count");
	const int64_t groups = fg.empty() ? 1 : fg[0];
	if (!bg.empty() && bg[0] != 1) {
		return L.unsupported("stablehlo.convolution with a batch group count above one");
	}
	Val x = transposed(L, as_type(L, lhs, GGML_TYPE_F32), in_perm); // [N, C, (H,) W]
	Val k = transposed(L, as_type(L, rhs, GGML_TYPE_F32), k_perm); // [O, I, (KH,) KW]
	if (ns == 1) {
		x = reshaped(L, x, { x.shape[0], x.shape[1], 1, x.shape[2] });
		k = reshaped(L, k, { k.shape[0], k.shape[1], 1, k.shape[2] });
		stride.insert(stride.begin(), 1);
		ld.insert(ld.begin(), 1);
		rd.insert(rd.begin(), 1);
		pad.insert(pad.begin(), 2, 0);
	}
	const int64_t C = x.shape[1], O = k.shape[0], I = k.shape[1];
	if (groups < 1 || C % groups != 0 || O % groups != 0 || I * groups != C) {
		return L.fail("stablehlo.convolution: the feature group count does not divide the channels");
	}
	std::vector<int64_t> low(4, 0), high(4, 0);
	for (size_t d = 0; d < 2; ++d) {
		if (stride[d] < 1 || ld[d] < 1 || rd[d] < 1) {
			return L.fail("stablehlo.convolution: a stride or dilation below one");
		}
		if (pad[d * 2] < 0 || pad[d * 2 + 1] < 0) {
			return L.unsupported("stablehlo.convolution with negative padding");
		}
		if (ld[d] > 1) {
			Val next;
			if (!dilate_dim(L, x, 2 + d, ld[d] - 1, next)) {
				return false;
			}
			x = next;
		}
		low[2 + d] = pad[d * 2];
		high[2 + d] = pad[d * 2 + 1];
	}
	Val padded;
	if (!pad_edges(L, x, low, high, 0.0f, padded)) {
		return false;
	}
	x = padded;
	std::vector<int64_t> oshape = { x.shape[0], O, 0, 0 };
	for (size_t d = 0; d < 2; ++d) {
		const int64_t span = (k.shape[2 + d] - 1) * rd[d] + 1;
		if (x.shape[2 + d] < span) {
			return L.fail("stablehlo.convolution: the window is larger than the padded input");
		}
		oshape[2 + d] = (x.shape[2 + d] - span) / stride[d] + 1;
	}
	const int s0 = int(stride[1]), s1 = int(stride[0]), d0 = int(rd[1]), d1 = int(rd[0]);
	Val y;
	y.type = GGML_TYPE_F32;
	y.shape = oshape;
	if (groups == 1) {
		y.t = ggml_conv_2d_direct(L.ctx, contiguous(L, k.t), contiguous(L, x.t), s0, s1, 0, 0, d0, d1);
	} else if (groups == C && I == 1 && O == C) {
		y.t = ggml_conv_2d_dw_direct(L.ctx, contiguous(L, k.t), contiguous(L, x.t), s0, s1, 0, 0, d0, d1);
	} else {
		if (groups > 16) {
			return L.unsupported("stablehlo.convolution with more than 16 feature groups that is not depthwise with a multiplier of one");
		}
		std::vector<Val> parts;
		for (int64_t g = 0; g < groups; ++g) {
			Val xs, ks;
			if (!slice_dim(L, x, 1, g * I, I, xs) || !slice_dim(L, k, 0, g * (O / groups), O / groups, ks)) {
				return false;
			}
			Val part;
			part.type = GGML_TYPE_F32;
			part.shape = { oshape[0], O / groups, oshape[2], oshape[3] };
			part.t = ggml_conv_2d_direct(L.ctx, contiguous(L, ks.t), contiguous(L, xs.t), s0, s1, 0, 0, d0, d1);
			parts.push_back(part);
		}
		if (!concat_dim(L, parts, 1, y)) {
			return false;
		}
	}
	if (ns == 1) {
		y = reshaped(L, y, { oshape[0], O, oshape[3] });
	}
	// y is [N, O, spatial...]; out_at[j] is where dimension j of that layout sits in the result.
	std::vector<int64_t> out_perm(rank);
	for (size_t j = 0; j < rank; ++j) {
		out_perm[size_t(out_at[j])] = int64_t(j);
	}
	out = as_type(L, transposed(L, y, out_perm), to.type);
	out.known = false;
	out.hv.clear();
	return true;
}

bool statement(Lowering &L, const std::string &line) {
	const size_t eq = line.find(" = ");
	if (line[0] != '%' || eq == std::string::npos) {
		return L.fail("cannot read statement '" + line.substr(0, 60) + "'");
	}
	const std::string name = line.substr(0, eq);
	std::string rest = line.substr(eq + 3);
	if (name.find(':') != std::string::npos) {
		return L.unsupported("an op with more than one result (a variadic reduce, as argmax writes)");
	}
	if (starts_with(rest, "\"stablehlo.")) {
		// The generic form: "stablehlo.gather"(%a, %b) <{...}> : ...
		const size_t q = rest.find('"', 1);
		if (q == std::string::npos) {
			return L.fail("cannot read statement '" + line.substr(0, 60) + "'");
		}
		rest = rest.substr(1, q - 1) + rest.substr(q + 1);
	}
	if (starts_with(rest, "chlo.erf ") || starts_with(rest, "chlo.erfc ") || starts_with(rest, "chlo.square ")) {
		// The CHLO ops JAX writes that have a lowering here; they are handled under the same names.
		rest = "stablehlo." + rest.substr(5);
	}
	if (starts_with(rest, "__alias ")) {
		// The result of an inlined call: the callee's returned value under the caller's name.
		std::map<std::string, Val>::const_iterator it = L.vals.find(rest.substr(8));
		if (it == L.vals.end()) {
			return L.fail("call: " + rest.substr(8) + " is not defined");
		}
		L.vals[name] = it->second;
		return true;
	}
	if (!starts_with(rest, "stablehlo.")) {
		return L.unsupported("'" + rest.substr(0, rest.find(' ')) + "' (only stablehlo ops)");
	}
	const size_t sp = rest.find(' ');
	std::string op = rest.substr(10, sp == std::string::npos ? std::string::npos : sp - 10);
	std::string tail = sp == std::string::npos ? std::string() : rest.substr(sp + 1);
	const size_t paren = op.find('(');
	if (paren != std::string::npos) {
		// "reduce(%x init: %i) ..." has no space before its operand list.
		tail = rest.substr(10 + paren);
		op = op.substr(0, paren);
	}
	if (op == "constant") {
		size_t colon = line.rfind(" : ");
		Ty ty;
		size_t e = 0;
		std::string perr;
		if (colon == std::string::npos || !parse_type_at(line, colon + 3, ty, e, perr)) {
			return L.type_error("stablehlo.constant", perr.empty() ? std::string("no type") : perr);
		}
		Val v;
		if (!op_constant(L, line, ty, v)) {
			return false;
		}
		L.vals[name] = v;
		return true;
	}
	if (op != "reduce" && tail.compare(0, 1, "(") == 0) {
		// "convolution(%a, %b) dim_numbers = ..." and the generic form list the operands in parentheses.
		const size_t close = tail.find(')');
		if (close == std::string::npos) {
			return L.fail("stablehlo." + op + ": cannot read its operands");
		}
		tail = tail.substr(1, close - 1) + tail.substr(close + 1);
	}
	if (op == "reduce" && tail.compare(0, 1, "(") == 0) {
		// "(%x init: %i) applies ..." reads as the operands "%x, %i".
		const size_t close = tail.find(')');
		const size_t init_at = tail.find(" init: ");
		if (close == std::string::npos || init_at == std::string::npos || init_at > close) {
			return L.fail("stablehlo.reduce: cannot read its operands");
		}
		tail = tail.substr(1, init_at - 1) + ", " + tail.substr(init_at + 7, close - init_at - 7) + tail.substr(close + 1);
	}
	std::string direction;
	if (op == "compare") {
		// "compare  GE, %a, %b,  SIGNED : ..." puts the direction first.
		const size_t c = tail.find(',');
		if (c == std::string::npos) {
			return L.fail("stablehlo.compare: no direction");
		}
		direction = trim(tail.substr(0, c));
		tail = trim(tail.substr(c + 1));
	}
	size_t after = 0;
	const std::vector<std::string> ops = operands(tail, after);
	const std::string attrs = tail.substr(after);
	const size_t colon = attrs.rfind(" : ");
	if (colon == std::string::npos) {
		return L.fail("stablehlo." + op + ": no type after ':'");
	}
	const std::string types = attrs.substr(colon + 3);
	Ty to;
	size_t e = 0;
	std::string perr;
	// The result type follows "->"; without one, the last type listed (select lists its predicate's first).
	const size_t arrow = types.rfind("-> ");
	const size_t last = types.rfind("tensor<");
	if (!parse_type_at(types, arrow != std::string::npos ? arrow + 3 : last == std::string::npos ? 0 : last, to, e, perr)) {
		return L.type_error("stablehlo." + op, perr);
	}
	std::vector<Val> in;
	bool all_float = is_float(to.type);
	for (const std::string &o : ops) {
		std::map<std::string, Val>::const_iterator it = L.vals.find(o);
		if (it == L.vals.end()) {
			return L.fail("stablehlo." + op + ": " + o + " is not defined");
		}
		in.push_back(it->second);
		all_float = all_float && is_float(it->second.type);
	}
	Val out;
	bool ok = false;
	bool folds = true; // the op sets out.known itself
	if (op == "square") {
		if (in.size() != 1) {
			return L.fail("chlo.square takes one operand");
		}
		in.push_back(in[0]);
		op = "multiply";
	}
	if (op == "add" || op == "subtract" || op == "multiply" || op == "divide" || op == "power" || op == "maximum" || op == "minimum" || op == "and" || op == "or" || op == "xor") {
		if (in.size() != 2) {
			return L.fail("stablehlo." + op + " takes two operands");
		}
		if (in[0].shape != in[1].shape) {
			return L.fail("stablehlo." + op + ": operand shapes differ");
		}
		const bool both_known = in[0].known && in[1].known;
		const bool minmax = op == "maximum" || op == "minimum";
		const bool logic = op == "and" || op == "or" || op == "xor";
		if (all_float && logic) {
			return L.unsupported("stablehlo." + op + " of float values");
		}
		if (all_float && minmax && !(op == "maximum" && (in[0].neg_inf || in[1].neg_inf))) {
			// XLA's maximum and minimum of values computed at run time (a maximum against a constant -inf is
			// softmax's no-op, and op_binary returns the other operand for it, as before).
			const ggml_type rt = in[0].type;
			out = xla_binary(L, op == "maximum" ? GGML_RD_XLA_MAX : GGML_RD_XLA_MIN, in[0], in[1], GGML_TYPE_F32);
			out = as_type(L, out, rt);
			ok = true;
			folds = false;
		} else if (all_float) {
			ok = op_binary(L, op, in[0], in[1], out);
			folds = false;
		} else if (!both_known && in[0].type == kPred && in[1].type == kPred && (logic || minmax)) {
			// i1 values computed at run time are 0 and 1: and is the minimum, or the maximum, xor not-equal.
			const ggml_rd_xla_binary_kind k = op == "and" || op == "minimum" ? GGML_RD_XLA_MIN : op == "xor" ? GGML_RD_XLA_NE : GGML_RD_XLA_MAX;
			out = xla_binary(L, k, in[0], in[1], kPred);
			ok = true;
			folds = false;
		} else if (!both_known && in[0].type == GGML_TYPE_I32 && in[1].type == GGML_TYPE_I32) {
			// i32 values computed at run time are whole numbers in f32: exact while they stay inside 2^24.
			if (minmax) {
				out = xla_binary(L, op == "maximum" ? GGML_RD_XLA_MAX : GGML_RD_XLA_MIN, in[0], in[1], GGML_TYPE_I32);
			} else if (op == "add" || op == "subtract" || op == "multiply") {
				const Val fa = device_f32(L, in[0]);
				const Val fb = device_f32(L, in[1]);
				Val o = fa;
				o.t = op == "add" ? ggml_add(L.ctx, fa.t, fb.t) : op == "subtract" ? ggml_sub(L.ctx, fa.t, fb.t) : ggml_mul(L.ctx, fa.t, fb.t);
				o.neg_inf = false;
				o.pre = nullptr;
				out = as_type(L, o, GGML_TYPE_I32);
			} else {
				return L.unsupported("stablehlo." + op + " of i32 values computed at run time (add, subtract, multiply, maximum and minimum only)");
			}
			ok = true;
			folds = false;
		} else {
			ok = host_binary(L, op, in[0], in[1], out);
		}
	} else if (op == "negate" || op == "exponential" || op == "sine" || op == "cosine" || op == "log" || op == "sqrt" || op == "rsqrt" || op == "abs" || op == "floor" || op == "sign" || op == "not" || op == "tanh" || op == "logistic" || op == "erf" || op == "erfc") {
		if (in.size() != 1) {
			return L.fail("stablehlo." + op + " takes one operand");
		}
		if (all_float && op != "not") {
			ok = op_unary(L, op, in[0], out);
			folds = false;
		} else if (op == "not" && in[0].type == kPred && !in[0].known) {
			// not of a run-time i1 (0 or 1) is equal-to-zero.
			out = xla_binary(L, GGML_RD_XLA_EQ, in[0], known_val(L, kPred, in[0].shape, std::vector<double>(size_t(count_of(in[0].shape)), 0.0)), kPred);
			ok = true;
			folds = false;
		} else if (!in[0].known && in[0].type == GGML_TYPE_I32 && in[0].t != nullptr && (op == "negate" || op == "abs")) {
			Val o = as_type(L, in[0], GGML_TYPE_F32);
			// 0 - x and not -x: a whole number's zero stays +0.
			o.t = op == "abs" ? ggml_abs(L.ctx, o.t) : ggml_scale_bias(L.ctx, o.t, -1.0f, 0.0f);
			out = as_type(L, o, GGML_TYPE_I32);
			ok = true;
			folds = false;
		} else {
			ok = host_unary(L, op, in[0], out);
		}
	} else if (op == "compare") {
		const size_t c = attrs.find(", ");
		const std::string kind = c == std::string::npos || c > colon ? std::string() : trim(attrs.substr(c + 2, colon - c - 2));
		ok = in.size() == 2 ? host_compare(L, direction, kind, in[0], in[1], out) : L.fail("stablehlo.compare takes two operands");
	} else if (op == "select") {
		ok = in.size() == 3 ? op_select(L, in[0], in[1], in[2], out) : L.fail("stablehlo.select takes three operands");
	} else if (op == "iota") {
		ok = in.empty() ? op_iota(L, attrs, to, out) : L.fail("stablehlo.iota takes no operands");
	} else if (op == "convert") {
		ok = in.size() == 1 ? op_convert(L, in[0], to, out) : L.fail("stablehlo.convert takes one operand");
	} else if (op == "reshape") {
		if (in.size() != 1 || count_of(in[0].shape) != count_of(to.shape)) {
			return L.fail("stablehlo.reshape: element counts differ");
		}
		out = reshaped(L, in[0], to.shape);
		ok = true;
	} else if (op == "transpose") {
		std::vector<int64_t> perm;
		if (in.size() != 1 || !int_list(attrs, "dims", 0, perm, nullptr) || perm.size() != in[0].shape.size()) {
			return L.fail("stablehlo.transpose: bad dims");
		}
		out = transposed(L, in[0], perm);
		ok = true;
	} else if (op == "broadcast_in_dim") {
		std::vector<int64_t> dims;
		if (in.size() != 1 || !int_list(attrs, "dims", 0, dims, nullptr)) {
			return L.fail("stablehlo.broadcast_in_dim: bad dims");
		}
		ok = op_broadcast(L, in[0], dims, to, out);
	} else if (op == "slice") {
		ok = in.size() == 1 ? op_slice(L, attrs, in[0], to, out) : L.fail("stablehlo.slice takes one operand");
	} else if (op == "concatenate") {
		ok = op_concatenate(L, attrs, in, out);
	} else if (op == "pad") {
		ok = in.size() == 2 ? op_pad(L, attrs, in[0], in[1], out) : L.fail("stablehlo.pad takes an operand and a padding value");
	} else if (op == "reduce") {
		ok = in.size() == 2 ? op_reduce(L, attrs, in[0], in[1], to, out) : L.fail("stablehlo.reduce takes one operand and an init");
		folds = ok && out.t == nullptr;
	} else if (op == "dot_general") {
		ok = in.size() == 2 ? op_dot(L, attrs, in[0], in[1], to, out) : L.fail("stablehlo.dot_general takes two operands");
		folds = false;
	} else if (op == "gather") {
		ok = in.size() == 2 ? op_gather(L, attrs, in[0], in[1], to, out) : L.fail("stablehlo.gather takes an operand and indices");
		folds = false;
	} else if (op == "clamp") {
		ok = in.size() == 3 ? op_clamp(L, in[0], in[1], in[2], out) : L.fail("stablehlo.clamp takes three operands");
	} else if (op == "dynamic_slice") {
		ok = in.size() >= 2 ? op_dynamic_slice(L, attrs, in, out) : L.fail("stablehlo.dynamic_slice takes an operand and start indices");
	} else if (op == "dynamic_update_slice") {
		std::vector<int64_t> start;
		if (in.size() < 3) {
			return L.fail("stablehlo.dynamic_update_slice takes an operand, an update and start indices");
		}
		if (!known_starts(in, 2, in[0].shape.size(), start)) {
			return L.unsupported("stablehlo.dynamic_update_slice with start indices computed at run time");
		}
		ok = op_update_slice(L, "stablehlo.dynamic_update_slice", in[0], in[1], start, out);
	} else if (op == "scatter") {
		ok = in.size() == 3 ? op_scatter(L, attrs, in[0], in[1], in[2], out) : L.unsupported("stablehlo.scatter of more than one operand");
	} else if (op == "reduce_window") {
		ok = in.size() == 2 ? op_reduce_window(L, attrs, in[0], in[1], to, out) : L.unsupported("stablehlo.reduce_window of more than one operand");
		folds = false;
	} else if (op == "convolution") {
		ok = in.size() == 2 ? op_convolution(L, attrs, in[0], in[1], to, out) : L.fail("stablehlo.convolution takes two operands");
		folds = false;
	} else {
		return L.unsupported("stablehlo." + op);
	}
	if (!ok) {
		return false;
	}
	if (!folds) {
		// A result computed on the device is not known here, even when its operands were.
		out.known = false;
		out.hv.clear();
	}
	if (out.shape != to.shape || out.type != to.type) {
		return L.fail("stablehlo." + op + ": the result is " + std::to_string(out.shape.size()) + "-d of type " + ggml_type_name(out.type)
				+ " but the module declares a different tensor type");
	}
	L.vals[name] = out;
	return true;
}

// Records for every tensor the outputs reach, the parameters first and in order.
bool describe(Lowering &L, const std::vector<ggml_tensor *> &outs, Program &p) {
	std::unordered_map<const ggml_tensor *, int> id;
	std::vector<ggml_tensor *> order;
	for (ggml_tensor *t : L.params) {
		id[t] = int(order.size());
		order.push_back(t);
	}
	std::vector<std::pair<ggml_tensor *, int>> stack;
	for (ggml_tensor *root : outs) {
		stack.emplace_back(root, 0);
		while (!stack.empty()) {
			ggml_tensor *t = stack.back().first;
			const int k = stack.back().second;
			if (id.count(t) != 0) {
				stack.pop_back();
				continue;
			}
			ggml_tensor *dep = nullptr;
			int next = k;
			while (next <= GGML_MAX_SRC && dep == nullptr) {
				ggml_tensor *c = next < GGML_MAX_SRC ? t->src[next] : t->view_src;
				++next;
				if (c != nullptr && c != t && id.count(c) == 0) {
					dep = c;
				}
			}
			if (dep != nullptr) {
				stack.back().second = next;
				stack.emplace_back(dep, 0);
			} else {
				id[t] = int(order.size());
				order.push_back(t);
				stack.pop_back();
			}
		}
	}
	for (size_t i = 0; i < order.size(); ++i) {
		const ggml_tensor *t = order[i];
		TensorRec r;
		r.is_arg = i < L.params.size();
		r.type = int(t->type);
		for (int d = 0; d < GGML_MAX_DIMS; ++d) {
			r.ne[d] = t->ne[d];
			r.nb[d] = uint64_t(t->nb[d]);
		}
		r.op = int(t->op);
		if (!r.is_arg) {
			std::memcpy(r.op_params, t->op_params, sizeof r.op_params);
			for (int s = 0; s < GGML_MAX_SRC; ++s) {
				r.src[s] = t->src[s] == nullptr ? -1 : (t->src[s] == t ? int(i) : id[t->src[s]]);
			}
			if (t->view_src != nullptr) {
				r.view_src = id[t->view_src];
				r.view_offs = uint64_t(t->view_offs);
			}
			if (t->op == GGML_OP_NONE && t->view_src == nullptr) {
				std::unordered_map<const ggml_tensor *, std::vector<uint8_t>>::const_iterator c = L.consts.find(t);
				if (c == L.consts.end()) {
					return L.fail("an operand is neither a parameter nor a constant");
				}
				Program::Const k;
				k.id = int(i);
				k.bytes = c->second;
				p.consts.push_back(std::move(k));
			}
		}
		char nm[32];
		std::snprintf(nm, sizeof nm, "v%zu", i);
		r.name = nm;
		p.t.push_back(std::move(r));
	}
	for (size_t i = 0; i < L.params.size(); ++i) {
		p.args.push_back(int(i));
	}
	for (ggml_tensor *o : outs) {
		p.outs.push_back(id[o]);
	}
	return true;
}

} // namespace

bool is_stablehlo(const std::string &text) {
	return text.find("ggml.arg") == std::string::npos && text.find("func.func") != std::string::npos;
}

namespace {

struct Func {
	std::vector<std::string> params; // %arg names, in order
	std::vector<std::string> body; // statements, then the return line
};

bool ident_char(char c) {
	return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '$' || c == '-';
}

// Every %name in line, renamed by f.
template <typename F>
std::string rename(const std::string &line, F f) {
	std::string o;
	size_t i = 0;
	while (i < line.size()) {
		if (line[i] != '%') {
			o += line[i++];
			continue;
		}
		size_t j = i + 1;
		while (j < line.size() && ident_char(line[j])) {
			++j;
		}
		o += f(line.substr(i, j - i));
		i = j;
	}
	return o;
}

// "%r = call @f(%a, %b) : ..." (or func.call): the callee's statements, renamed so they cannot collide, with
// its parameters bound to the caller's operands, then an alias line that gives the returned value the caller's
// name. Calls inside the callee are inlined the same way.
bool inline_call(Lowering &L, const std::map<std::string, Func> &funcs, const std::string &line, int depth, int &counter, std::vector<std::string> &out) {
	const size_t eq = line.find(" = ");
	const size_t at = line.find('@');
	const size_t open = line.find('(', at);
	const size_t close = line.find(')', open);
	if (eq == std::string::npos || at == std::string::npos || open == std::string::npos || close == std::string::npos) {
		return L.fail("call: cannot read '" + line.substr(0, 60) + "'");
	}
	const std::string result = line.substr(0, eq);
	if (result.find(':') != std::string::npos) {
		return L.unsupported("a call with more than one result");
	}
	const std::string callee = line.substr(at + 1, open - at - 1);
	std::map<std::string, Func>::const_iterator f = funcs.find(callee);
	if (f == funcs.end()) {
		return L.fail("call: no function @" + callee);
	}
	if (depth > 32) {
		return L.unsupported("calls nested deeper than 32 (a recursive function)");
	}
	size_t after = 0;
	const std::vector<std::string> args = operands(line.substr(open + 1, close - open - 1), after);
	if (args.size() != f->second.params.size()) {
		return L.fail("call: @" + callee + " takes " + std::to_string(f->second.params.size()) + " operands");
	}
	std::map<std::string, std::string> bound;
	for (size_t i = 0; i < args.size(); ++i) {
		bound[f->second.params[i]] = args[i];
	}
	const std::string prefix = "%i" + std::to_string(counter++) + "_";
	std::function<std::string(const std::string &)> map = [&](const std::string &v) {
		std::map<std::string, std::string>::const_iterator b = bound.find(v);
		return b != bound.end() ? b->second : prefix + v.substr(1);
	};
	for (const std::string &s : f->second.body) {
		const std::string r = rename(s, map);
		if (starts_with(r, "return ")) {
			size_t a2 = 0;
			const std::vector<std::string> ret = operands(r.substr(7), a2);
			if (ret.size() != 1) {
				return L.unsupported("a function that returns other than one value");
			}
			out.push_back(result + " = __alias " + ret[0]);
			return true;
		}
		const size_t e2 = r.find(" = ");
		if (e2 != std::string::npos && (starts_with(r.substr(e2 + 3), "call @") || starts_with(r.substr(e2 + 3), "func.call @"))) {
			if (!inline_call(L, funcs, r, depth + 1, counter, out)) {
				return false;
			}
			continue;
		}
		out.push_back(r);
	}
	return L.fail("call: @" + callee + " has no return");
}

} // namespace

bool lower_stablehlo(const std::string &text, Program &p, std::string &err) {
	Lowering L;
	std::vector<std::string> lines;
	size_t pos = 0;
	while (pos <= text.size()) {
		size_t nl = text.find('\n', pos);
		if (nl == std::string::npos) {
			nl = text.size();
		}
		const std::string l = trim(text.substr(pos, nl - pos));
		if (!l.empty()) {
			lines.push_back(l);
		}
		pos = nl + 1;
	}
	// An op with a region (reduce_window, scatter, sort, a variadic reduce) spans several lines: join them to its
	// first, so that every statement is one line.
	{
		std::vector<std::string> joined;
		for (size_t i = 0; i < lines.size(); ++i) {
			std::string cur = lines[i];
			if (cur[0] == '%') {
				int depth = 0;
				for (char c : cur) {
					depth += c == '{' ? 1 : c == '}' ? -1 : 0;
				}
				while (i + 1 < lines.size() && (depth > 0 || starts_with(lines[i + 1], "reducer("))) {
					++i;
					for (char c : lines[i]) {
						depth += c == '{' ? 1 : c == '}' ? -1 : 0;
					}
					cur += " " + lines[i];
				}
			}
			joined.push_back(cur);
		}
		lines.swap(joined);
	}
	// The functions: @main and the private ones it calls, which are inlined.
	std::map<std::string, Func> funcs;
	std::string main_sig;
	std::string current;
	bool have_main = false;
	for (const std::string &line : lines) {
		if (!L.err.empty()) {
			break;
		}
		if (current.empty()) {
			if (starts_with(line, "module ") || line == "}") {
				continue;
			}
			if (!starts_with(line, "func.func")) {
				L.fail("text outside a function: '" + line.substr(0, 40) + "'");
				break;
			}
			const size_t at = line.find('@');
			const size_t open = line.find('(', at);
			if (at == std::string::npos || open == std::string::npos) {
				L.fail("cannot read '" + line.substr(0, 60) + "'");
				break;
			}
			current = line.substr(at + 1, open - at - 1);
			if (funcs.count(current) != 0) {
				L.fail("function @" + current + " is defined twice");
				break;
			}
			Func &f = funcs[current];
			const size_t close = line.find(") ->", open);
			const std::string sig = close == std::string::npos ? line.substr(open + 1) : line.substr(open + 1, close - open - 1);
			if (current == "main") {
				have_main = true;
				main_sig = sig;
			}
			size_t p2 = 0;
			while ((p2 = sig.find('%', p2)) != std::string::npos) {
				const size_t colon = sig.find(':', p2);
				if (colon == std::string::npos) {
					break;
				}
				f.params.push_back(sig.substr(p2, colon - p2));
				p2 = colon;
			}
			continue;
		}
		if (line == "}") {
			current.clear();
			continue;
		}
		funcs[current].body.push_back(line);
	}
	if (L.err.empty() && !have_main) {
		L.unsupported("a module without a function named @main");
	}
	// main's statements with every call inlined.
	std::vector<std::string> flat;
	int counter = 0;
	if (L.err.empty()) {
		for (const std::string &line : funcs["main"].body) {
			const size_t eq = line.find(" = ");
			if (eq != std::string::npos && (starts_with(line.substr(eq + 3), "call @") || starts_with(line.substr(eq + 3), "func.call @"))) {
				if (!inline_call(L, funcs, line, 0, counter, flat)) {
					break;
				}
				continue;
			}
			flat.push_back(line);
		}
	}
	ggml_init_params ip = { ggml_tensor_overhead() * (flat.size() * 192 + 8192) + 4096, nullptr, true };
	L.ctx = ggml_init(ip);
	if (L.ctx == nullptr) {
		err = "ggml_init failed";
		return false;
	}
	// main's parameters, in order.
	size_t p2 = 0;
	while (L.err.empty() && (p2 = main_sig.find("%arg", p2)) != std::string::npos) {
		const size_t colon = main_sig.find(':', p2);
		Ty ty;
		size_t e = 0;
		std::string perr;
		if (colon == std::string::npos || !parse_type_at(main_sig, colon + 2, ty, e, perr)) {
			L.type_error("main parameter", perr.empty() ? std::string("no type") : perr);
			break;
		}
		if (ty.type == kPred) {
			L.unsupported("an i1 parameter of @main");
			break;
		}
		Val v;
		// An i32 parameter is an f32 tensor of whole numbers, like every run-time i32; Execute converts the
		// caller's buffer (Program::arg_i32).
		v.t = new_tensor(L, ty.type == GGML_TYPE_I32 ? GGML_TYPE_F32 : ty.type, ty.shape);
		v.shape = ty.shape;
		v.type = ty.type;
		L.vals[main_sig.substr(p2, colon - p2)] = v;
		L.params.push_back(v.t);
		p.arg_i32.push_back(ty.type == GGML_TYPE_I32 ? 1 : 0);
		p2 = e;
	}
	bool returned = false;
	std::vector<ggml_tensor *> outs;
	for (const std::string &line : flat) {
		if (!L.err.empty()) {
			break;
		}
		if (returned) {
			L.fail("text after the return in @main");
			break;
		}
		if (starts_with(line, "return ") || line == "return") {
			size_t after = 0;
			const std::vector<std::string> ops = operands(line.size() > 7 ? line.substr(7) : std::string(), after);
			const size_t colon = line.find(" : ");
			if (ops.empty() || colon == std::string::npos) {
				L.fail("return without operands or types");
				break;
			}
			for (const std::string &o : ops) {
				std::map<std::string, Val>::const_iterator it = L.vals.find(o);
				if (it == L.vals.end()) {
					L.fail("return: " + o + " is not defined");
					break;
				}
				if (!is_float(it->second.type) && !(it->second.type == GGML_TYPE_I32 && it->second.t != nullptr)) {
					L.unsupported("returning an i1 value, or an i32 value known at compile time");
					break;
				}
				bool is_param = false;
				for (ggml_tensor *pt : L.params) {
					is_param = is_param || pt == it->second.t;
				}
				if (is_param) {
					L.unsupported("returning a parameter unchanged");
					break;
				}
				// A run-time i32 result leaves as real i32 (the f32 to i32 copy kernel).
				outs.push_back(it->second.type == GGML_TYPE_I32 ? ggml_cast(L.ctx, contiguous(L, it->second.t), GGML_TYPE_I32) : contiguous(L, it->second.t));
			}
			returned = true;
			continue;
		}
		if (!statement(L, line)) {
			break;
		}
	}
	if (L.err.empty() && !returned) {
		L.fail("no return in @main");
	}
	bool ok = L.err.empty() && describe(L, outs, p);
	if (!ok) {
		err = L.err;
	}
	ggml_free(L.ctx);
	return ok;
}

} // namespace pjrt_rd
