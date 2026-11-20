// The StableHLO gate of pjrt-rd, driven only through the PJRT C API, shared by the guest probe
// (probe_pjrt.cpp, on ggml-rd in the engine) and the host test (tests/pjrt_host, on ggml's CPU backend).
//
//   run_cases     real StableHLO modules from JAX (stablehlo_cases.inc), each compiled and executed, against the
//                 output XLA's CPU backend gave for the same module. A mutated module is each case's control: it
//                 must be refused or give another output. In bit mode a case must be bit-equal unless it carries
//                 a measured reason (XFAIL), and then it must meet the functional bar and must NOT be bit-equal.
//                 In functional mode (another device than the GPU kernels) every case must meet the bar.
//   run_refusals  valid StableHLO outside the lowering, and malformed text, must each come back with its code.
//   run_module    one module as a pipeline stage: module.mlir and in<k>.bin from a reader of named blobs, the
//                 results as out<k>.bin, compared with expect<k>.bin when asked.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "pjrt_c_api.h"

namespace stablehlo_gate {

#ifndef PJRT_ARGS
#define PJRT_ARGS(T, name) \
	T name; \
	std::memset(&name, 0, sizeof name); \
	name.struct_size = T##_STRUCT_SIZE
#endif

inline const PJRT_Api *g_api = nullptr;

// Consumes the error. Returns its text, empty when there was none.
inline std::string take(PJRT_Error *e, PJRT_Error_Code *code = nullptr) {
	if (e == nullptr) {
		return std::string();
	}
	PJRT_ARGS(PJRT_Error_Message_Args, m);
	m.error = e;
	g_api->PJRT_Error_Message(&m);
	std::string s(m.message, m.message_size);
	if (code != nullptr) {
		PJRT_ARGS(PJRT_Error_GetCode_Args, c);
		c.error = e;
		g_api->PJRT_Error_GetCode(&c);
		*code = c.code;
	}
	PJRT_ARGS(PJRT_Error_Destroy_Args, d);
	d.error = e;
	g_api->PJRT_Error_Destroy(&d);
	return s.empty() ? std::string("(no text)") : s;
}

inline PJRT_Client *make_client() {
	PJRT_ARGS(PJRT_Plugin_Initialize_Args, pi);
	PJRT_ARGS(PJRT_Client_Create_Args, cc);
	if (!take(g_api->PJRT_Plugin_Initialize(&pi)).empty() || !take(g_api->PJRT_Client_Create(&cc)).empty()) {
		std::printf("PROBE pjrt: no client\n");
		return nullptr;
	}
	return cc.client;
}

inline void destroy_client(PJRT_Client *c) {
	PJRT_ARGS(PJRT_Client_Destroy_Args, d);
	d.client = c;
	take(g_api->PJRT_Client_Destroy(&d));
}

inline PJRT_Buffer *upload(PJRT_Client *c, const void *data, size_t bytes) {
	PJRT_ARGS(PJRT_Client_BufferFromHostBuffer_Args, a);
	const int64_t dim = int64_t(bytes);
	a.client = c;
	a.data = data;
	a.type = PJRT_Buffer_Type_U8;
	a.dims = &dim;
	a.num_dims = 1;
	a.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
	const std::string err = take(g_api->PJRT_Client_BufferFromHostBuffer(&a));
	if (!err.empty()) {
		std::printf("PROBE pjrt: BufferFromHostBuffer failed: %s\n", err.c_str());
		return nullptr;
	}
	PJRT_ARGS(PJRT_Event_Destroy_Args, ed);
	ed.event = a.done_with_host_buffer;
	take(g_api->PJRT_Event_Destroy(&ed));
	return a.buffer;
}

inline void drop(PJRT_Buffer *b) {
	if (b == nullptr) {
		return;
	}
	PJRT_ARGS(PJRT_Buffer_Destroy_Args, d);
	d.buffer = b;
	take(g_api->PJRT_Buffer_Destroy(&d));
}

struct Run {
	bool ok = false;
	std::vector<std::vector<uint8_t>> out;
	std::string error;
};

// Compiles and executes. The number of results is the executable's own.
inline Run execute(PJRT_Client *client, const std::string &text, const std::vector<PJRT_Buffer *> &args) {
	Run r;
	for (PJRT_Buffer *b : args) {
		if (b == nullptr) {
			r.error = "an argument could not be uploaded";
			return r;
		}
	}
	PJRT_Program prog;
	std::memset(&prog, 0, sizeof prog);
	prog.struct_size = PJRT_Program_STRUCT_SIZE;
	prog.code = const_cast<char *>(text.data());
	prog.code_size = text.size();
	prog.format = "mlir";
	prog.format_size = 4;
	PJRT_ARGS(PJRT_Client_Compile_Args, cc);
	cc.client = client;
	cc.program = &prog;
	r.error = take(g_api->PJRT_Client_Compile(&cc));
	if (!r.error.empty()) {
		return r;
	}
	size_t n_outs = 0;
	{
		PJRT_ARGS(PJRT_LoadedExecutable_GetExecutable_Args, ge);
		ge.loaded_executable = cc.executable;
		r.error = take(g_api->PJRT_LoadedExecutable_GetExecutable(&ge));
		if (r.error.empty()) {
			PJRT_ARGS(PJRT_Executable_NumOutputs_Args, no);
			no.executable = ge.executable;
			r.error = take(g_api->PJRT_Executable_NumOutputs(&no));
			n_outs = no.num_outputs;
			PJRT_ARGS(PJRT_Executable_Destroy_Args, ed);
			ed.executable = ge.executable;
			take(g_api->PJRT_Executable_Destroy(&ed));
		}
	}
	if (r.error.empty()) {
		PJRT_ARGS(PJRT_ExecuteOptions, opt);
		PJRT_Buffer *const *arg_list = args.data();
		std::vector<PJRT_Buffer *> outs(n_outs, nullptr);
		PJRT_Buffer **out_list = outs.data();
		PJRT_ARGS(PJRT_LoadedExecutable_Execute_Args, ex);
		ex.executable = cc.executable;
		ex.options = &opt;
		ex.argument_lists = &arg_list;
		ex.num_devices = 1;
		ex.num_args = args.size();
		ex.output_lists = &out_list;
		r.error = take(g_api->PJRT_LoadedExecutable_Execute(&ex));
		if (r.error.empty()) {
			r.ok = true;
			for (PJRT_Buffer *o : outs) {
				PJRT_ARGS(PJRT_Buffer_ToHostBuffer_Args, q);
				q.src = o;
				take(g_api->PJRT_Buffer_ToHostBuffer(&q));
				std::vector<uint8_t> v(q.dst_size);
				PJRT_ARGS(PJRT_Buffer_ToHostBuffer_Args, th);
				th.src = o;
				th.dst = v.data();
				th.dst_size = v.size();
				const std::string err = take(g_api->PJRT_Buffer_ToHostBuffer(&th));
				if (!err.empty()) {
					r.ok = false;
					r.error = "ToHostBuffer(result): " + err;
				}
				r.out.push_back(std::move(v));
				drop(o);
			}
		}
	}
	PJRT_ARGS(PJRT_LoadedExecutable_Destroy_Args, ld);
	ld.executable = cc.executable;
	take(g_api->PJRT_LoadedExecutable_Destroy(&ld));
	return r;
}

enum class StablehloDtype { F32, BF16, I32 };

struct StablehloArg {
	StablehloDtype dtype;
	int count;
	const unsigned char *bytes;
	size_t size;
};

struct StablehloCase {
	const char *name;
	const char *hlo;
	const char *bad;
	const StablehloArg *args;
	int nargs;
	const unsigned char *out;
	size_t out_size;
	StablehloDtype out_dtype;
	const char *note;
	const char *xfail; // why the output is not bit-equal to XLA CPU's; nullptr when it must be
};

#include "stablehlo_cases.inc"

inline size_t element_size(StablehloDtype d) {
	return d == StablehloDtype::BF16 ? 2 : 4;
}

inline float element(const unsigned char *p, StablehloDtype d, size_t i) {
	if (d == StablehloDtype::F32) {
		float f;
		std::memcpy(&f, p + i * 4, 4);
		return f;
	}
	if (d == StablehloDtype::I32) {
		int32_t v;
		std::memcpy(&v, p + i * 4, 4);
		return float(v);
	}
	uint16_t h;
	std::memcpy(&h, p + i * 2, 2);
	const uint32_t w = uint32_t(h) << 16;
	float f;
	std::memcpy(&f, &w, 4);
	return f;
}

// The bits of a float mapped so that adjacent values are adjacent integers; an i32 is its own value.
inline int64_t ordered(const unsigned char *p, StablehloDtype d, size_t i) {
	if (d == StablehloDtype::I32) {
		int32_t v;
		std::memcpy(&v, p + i * 4, 4);
		return v;
	}
	if (d == StablehloDtype::BF16) {
		uint16_t h;
		std::memcpy(&h, p + i * 2, 2);
		return ((h & 0x8000u) != 0 ? -1 : 1) * int64_t(h & 0x7fffu);
	}
	uint32_t w;
	std::memcpy(&w, p + i * 4, 4);
	return ((w & 0x80000000u) != 0 ? -1 : 1) * int64_t(w & 0x7fffffffu);
}

struct Compared {
	size_t n = 0;
	size_t differing = 0;
	int64_t max_ulp = 0;
	double rel_l2 = 0.0; // over the finite elements; 1 when an element is finite on one side only, or another infinity
	double max_abs = 0.0; // the largest difference of two finite elements
	size_t other_class = 0;
};

inline Compared compare_to(const unsigned char *want, const std::vector<uint8_t> &got, size_t want_size, StablehloDtype d) {
	Compared c;
	if (got.size() != want_size) {
		c.n = want_size;
		c.differing = want_size;
		c.max_ulp = INT64_MAX;
		c.rel_l2 = 1.0;
		return c;
	}
	const size_t esz = element_size(d);
	c.n = want_size / esz;
	double num = 0.0;
	double den = 0.0;
	for (size_t i = 0; i < c.n; ++i) {
		const float w = element(want, d, i);
		const float g = element(got.data(), d, i);
		// IEEE 754 and StableHLO leave a NaN's payload unspecified (XLA CPU writes 7fc00000, the GPU's 0/0 is 7fffffff),
		// so two NaNs are the same result. A NaN against a number is a difference.
		if (std::isnan(w) && std::isnan(g)) {
			continue;
		}
		if (std::memcmp(want + i * esz, got.data() + i * esz, esz) != 0) {
			++c.differing;
			c.max_ulp = std::max<int64_t>(c.max_ulp, std::llabs(ordered(want, d, i) - ordered(got.data(), d, i)));
		}
		// An infinity or a NaN has no distance: it is either the same value or a different class of result.
		if (!std::isfinite(w) || !std::isfinite(g)) {
			c.other_class += w == g ? 0 : 1;
			continue;
		}
		num += (double(g) - double(w)) * (double(g) - double(w));
		den += double(w) * double(w);
		c.max_abs = std::max(c.max_abs, std::fabs(double(g) - double(w)));
	}
	c.rel_l2 = c.other_class != 0 ? 1.0 : den > 0.0 ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
	return c;
}

struct Refusal {
	const char *name;
	const char *body; // the statements of @main, after its parameter %arg0: tensor<4xf32>
	PJRT_Error_Code code;
};

// Valid StableHLO outside what the lowering handles, and malformed text, must each come back as a
// specific error from Compile and never reach ggml.
inline size_t run_refusals(PJRT_Client *client, size_t *total) {
	static const Refusal kRefusals[] = {
		{ "unsupported_op", "%0 = stablehlo.tan %arg0 : tensor<4xf32>\n    return %0 : tensor<4xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "hex_constant", "%0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>\n    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>\n    return %1 : tensor<4xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "reduce_multiply", "%0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>\n    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.multiply across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>\n    return %1 : tensor<f32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "reduce_init_not_identity", "%0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>\n    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>\n    return %1 : tensor<f32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "reduce_bad_dimension", "%0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>\n    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [3] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>\n    return %1 : tensor<f32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "remainder_of_values", "%0 = stablehlo.remainder %arg0, %arg0 : tensor<4xf32>\n    return %0 : tensor<4xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "pad_nonzero_value", "%0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>\n    %1 = stablehlo.pad %arg0, %0, low = [1], high = [0], interior = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<5xf32>\n    return %1 : tensor<5xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "pad_interior", "%0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>\n    %1 = stablehlo.pad %arg0, %0, low = [0], high = [0], interior = [1] : (tensor<4xf32>, tensor<f32>) -> tensor<7xf32>\n    return %1 : tensor<7xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "slice_past_end", "%0 = stablehlo.slice %arg0 [2:6] : (tensor<4xf32>) -> tensor<4xf32>\n    return %0 : tensor<4xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "concatenate_rank_mismatch", "%0 = stablehlo.concatenate %arg0, %arg0, dim = 3 : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>\n    return %0 : tensor<8xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "shape_mismatch", "%0 = stablehlo.add %arg0, %arg0 : tensor<5xf32>\n    return %0 : tensor<5xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "wrong_declared_type", "%0 = stablehlo.negate %arg0 : tensor<5xf32>\n    return %0 : tensor<5xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "undefined_operand", "%0 = stablehlo.add %arg0, %nope : tensor<4xf32>\n    return %0 : tensor<4xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "no_return", "%0 = stablehlo.negate %arg0 : tensor<4xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "returns_parameter", "return %arg0 : tensor<4xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "garbage_statement", "this is not a statement", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "bad_broadcast", "%0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xf32>) -> tensor<4x4xf32>\n    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<4xf32>) -> tensor<4x4xf32>\n    return %1 : tensor<4x4xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "reshape_count", "%0 = stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<5xf32>\n    return %0 : tensor<5xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
	};
	static const char *kWholeModules[][2] = {
		{ "integer_type", "module @m {\n  func.func public @main(%arg0: tensor<4xi32>) -> (tensor<4xi32>) {\n    %0 = stablehlo.negate %arg0 : tensor<4xi32>\n    return %0 : tensor<4xi32>\n  }\n}" },
		{ "rank_five", "module @m {\n  func.func public @main(%arg0: tensor<2x2x2x2x2xf32>) -> (tensor<2x2x2x2x2xf32>) {\n    %0 = stablehlo.negate %arg0 : tensor<2x2x2x2x2xf32>\n    return %0 : tensor<2x2x2x2x2xf32>\n  }\n}" },
		{ "dynamic_shape", "module @m {\n  func.func public @main(%arg0: tensor<?xf32>) -> (tensor<?xf32>) {\n    %0 = stablehlo.negate %arg0 : tensor<?xf32>\n    return %0 : tensor<?xf32>\n  }\n}" },
		{ "recursive_call", "module @m {\n  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {\n    %0 = call @loop(%arg0) : (tensor<4xf32>) -> tensor<4xf32>\n    return %0 : tensor<4xf32>\n  }\n  func.func private @loop(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {\n    %0 = call @loop(%arg0) : (tensor<4xf32>) -> tensor<4xf32>\n    return %0 : tensor<4xf32>\n  }\n}" },
		{ "call_undefined", "module @m {\n  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {\n    %0 = call @nope(%arg0) : (tensor<4xf32>) -> tensor<4xf32>\n    return %0 : tensor<4xf32>\n  }\n}" },
		{ "runtime_int_compare", "module @m {\n  func.func public @main(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {\n    %0 = stablehlo.compare  GT, %arg0, %arg0,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>\n    %1 = stablehlo.select %0, %arg1, %arg1 : tensor<4xi1>, tensor<4xf32>\n    return %1 : tensor<4xf32>\n  }\n}" },
		{ "runtime_integer", "module @m {\n  func.func public @main(%arg0: tensor<4xi32>) -> (tensor<4xf32>) {\n    %0 = stablehlo.convert %arg0 : (tensor<4xi32>) -> tensor<4xf32>\n    return %0 : tensor<4xf32>\n  }\n}" },
	};
	size_t held = 0;
	std::vector<std::pair<std::string, std::pair<std::string, PJRT_Error_Code> > > all;
	for (const Refusal &r : kRefusals) {
		all.push_back({ r.name, { std::string("module @m {\n  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {\n    ") + r.body + "\n  }\n}", r.code } });
	}
	for (const char *const *w : kWholeModules) {
		all.push_back({ w[0], { w[1], std::strstr(w[0], "undefined") != nullptr ? PJRT_Error_Code_INVALID_ARGUMENT : PJRT_Error_Code_UNIMPLEMENTED } });
	}
	for (const std::pair<std::string, std::pair<std::string, PJRT_Error_Code> > &r : all) {
		PJRT_Program prog;
		std::memset(&prog, 0, sizeof prog);
		prog.struct_size = PJRT_Program_STRUCT_SIZE;
		prog.code = const_cast<char *>(r.second.first.data());
		prog.code_size = r.second.first.size();
		prog.format = "mlir";
		prog.format_size = 4;
		PJRT_ARGS(PJRT_Client_Compile_Args, cc);
		cc.client = client;
		cc.program = &prog;
		PJRT_Error_Code got = PJRT_Error_Code_OK;
		const std::string msg = take(g_api->PJRT_Client_Compile(&cc), &got);
		const bool ok = !msg.empty() && got == r.second.second;
		held += ok ? 1 : 0;
		std::printf("PROBE pjrt stablehlo refusal %-20s %s code=%d wanted=%d %.100s\n", r.first.c_str(), ok ? "held  " : "FAILED", int(got), int(r.second.second), msg.c_str());
	}
	*total = all.size();
	return held;
}


enum class Mode {
	Bits, // the GPU kernels: bit equality with XLA CPU, except the cases with a measured reason
	Functional, // another device (ggml's CPU backend): the functional bar only
};

// Real StableHLO from JAX, compiled and executed through the C API, against the output XLA's CPU
// backend gave for the same module. A mutated module is the control: it must be refused or give
// another output.
// device_differs (functional mode): cases this device cannot meet, each with its measured reason. Such a case
// must miss the bar, so the list only shrinks.
inline bool run_cases(PJRT_Client *client, const std::string &which, Mode mode, const std::map<std::string, std::string> *device_differs = nullptr) {
	bool ok = client != nullptr;
	size_t ran = 0, bit_equal = 0, expected_failures = 0, controls_held = 0, functional_met = 0;
	{
		std::vector<uint8_t> flipped(kStablehloCases[0].out, kStablehloCases[0].out + kStablehloCases[0].out_size);
		flipped[0] ^= 1;
		const Compared self = compare_to(kStablehloCases[0].out, flipped, flipped.size(), kStablehloCases[0].out_dtype);
		std::printf("PROBE pjrt stablehlo compare control: one flipped bit is seen as %zu differing element(s) (must be 1)\n", self.differing);
		ok = ok && self.differing == 1;
		// A result of zeros must not pass a case whose reference is not zeros.
		const std::vector<uint8_t> zeros(kStablehloCases[0].out_size, 0);
		const Compared dead = compare_to(kStablehloCases[0].out, zeros, zeros.size(), kStablehloCases[0].out_dtype);
		std::printf("PROBE pjrt stablehlo compare control: an all-zero result has rel_l2=%.3f (must be 1)\n", dead.rel_l2);
		ok = ok && dead.rel_l2 > 0.99 && dead.differing != 0;
	}
	for (const StablehloCase &c : kStablehloCases) {
		if (client == nullptr || (which != "all" && which != c.name)) {
			continue;
		}
		std::vector<PJRT_Buffer *> bufs;
		for (int k = 0; k < c.nargs; ++k) {
			bufs.push_back(upload(client, c.args[k].bytes, c.args[k].size));
		}
		const Run good = execute(client, c.hlo, bufs);
		Compared cmp;
		if (good.ok && good.out.size() == 1) {
			cmp = compare_to(c.out, good.out[0], c.out_size, c.out_dtype);
		} else {
			cmp.n = c.out_size;
			cmp.differing = c.out_size;
			cmp.max_ulp = INT64_MAX;
			cmp.rel_l2 = 1.0;
		}
		const Run bad = execute(client, c.bad, bufs);
		const bool control = !bad.ok || bad.out.size() != 1 || compare_to(c.out, bad.out[0], c.out_size, c.out_dtype).differing != 0;
		const bool bits = good.ok && cmp.differing == 0;
		const double bar = c.out_dtype == StablehloDtype::BF16 ? 1e-3 : 1e-4; // the oracle bar of the graph probes
		const bool functional = good.ok && cmp.rel_l2 <= bar;
		++ran;
		bit_equal += bits ? 1 : 0;
		functional_met += functional ? 1 : 0;
		expected_failures += c.xfail != nullptr ? 1 : 0;
		controls_held += control ? 1 : 0;
		std::printf("PROBE pjrt stablehlo %-22s ran=%d bit_equal=%d differing=%zu/%zu max_ulp=%lld rel_l2=%.3e control=%s%s%s\n", c.name,
				good.ok ? 1 : 0, bits ? 1 : 0, cmp.differing, cmp.n, (long long)cmp.max_ulp, cmp.rel_l2, control ? "held" : "FAILED",
				good.ok ? "" : " error=", good.ok ? "" : good.error.c_str());
		if (good.ok && good.out.size() == 1 && cmp.differing != 0 && (mode == Mode::Bits || !functional)) {
			// Name the first few elements that differ, with the bit patterns, so a failure can be read.
			const size_t esz = element_size(c.out_dtype);
			size_t shown = 0;
			for (size_t i = 0; i * esz < c.out_size && i * esz < good.out[0].size() && shown < 4; ++i) {
				if (std::memcmp(c.out + i * esz, good.out[0].data() + i * esz, esz) != 0
						&& !(std::isnan(element(c.out, c.out_dtype, i)) && std::isnan(element(good.out[0].data(), c.out_dtype, i)))) {
					uint32_t want_bits = 0, got_bits = 0;
					std::memcpy(&want_bits, c.out + i * esz, esz);
					std::memcpy(&got_bits, good.out[0].data() + i * esz, esz);
					std::printf("PROBE pjrt stablehlo %-22s   element %zu: XLA CPU %0*x, this device %0*x\n", c.name, i, int(esz * 2), want_bits, int(esz * 2), got_bits);
					++shown;
				}
			}
		}
		const std::map<std::string, std::string>::const_iterator dd = device_differs == nullptr ? std::map<std::string, std::string>::const_iterator() : device_differs->find(c.name);
		if (mode == Mode::Functional && device_differs != nullptr && dd != device_differs->end()) {
			std::printf("PROBE pjrt stablehlo %-22s expected to miss the bar on this device (%s) %s\n", c.name, dd->second.c_str(),
					functional ? "NOT NEEDED: it meets the bar now, so remove it from the list" : "as measured");
			ok = ok && !functional && control;
		} else if (mode == Mode::Functional) {
			if (!functional) {
				std::printf("PROBE pjrt stablehlo %-22s functional bar %.0e NOT MET\n", c.name, bar);
			}
			ok = ok && functional && control;
		} else if (c.xfail != nullptr) {
			std::printf("PROBE pjrt stablehlo %-22s expected not bit-equal (%s); functional bar %.0e %s\n", c.name, c.xfail, bar,
					bits ? "NOT NEEDED: it is bit-equal now, so remove it from XFAIL" : (functional ? "met" : "NOT MET"));
			ok = ok && !bits && functional && control;
		} else {
			ok = ok && bits && control;
		}
		for (PJRT_Buffer *b : bufs) {
			drop(b);
		}
	}
	size_t refusals = 0, refusals_held = 0;
	if (client != nullptr && which == "all") {
		refusals_held = run_refusals(client, &refusals);
		ok = ok && refusals_held == refusals;
	}
	std::printf("PROBE pjrt stablehlo refusals held=%zu/%zu\n", refusals_held, refusals);
	std::printf("PROBE pjrt stablehlo SUMMARY mode=%s cases=%zu bit_equal=%zu functional=%zu expected_not_bit_equal=%zu controls_held=%zu\n",
			mode == Mode::Bits ? "bits" : "functional", ran, bit_equal, functional_met, expected_failures, controls_held);
	return ok && ran > 0;
}

// --- one module, its inputs and its reference from named blobs -------------------------------------------

// Reads a named blob (module.mlir, in0.bin, ...). False when there is none.
typedef std::function<bool(const std::string &name, std::vector<uint8_t> &bytes)> Reader;
// The results, by name (out0.bin, ...).
typedef std::map<std::string, std::vector<uint8_t>> Store;

inline uint64_t fnv1a(const std::vector<uint8_t> &v) {
	uint64_t h = 1469598103934665603ull;
	for (uint8_t b : v) {
		h = (h ^ b) * 1099511628211ull;
	}
	return h;
}

// The parameters of @main and the element type of each of its results, read from the module's text.
inline bool main_signature(const std::string &text, size_t *params, std::vector<StablehloDtype> *results) {
	const size_t at = text.find("@main(");
	const size_t arrow = at == std::string::npos ? std::string::npos : text.find(") ->", at);
	const size_t brace = arrow == std::string::npos ? std::string::npos : text.find('\n', arrow);
	if (brace == std::string::npos) {
		return false;
	}
	*params = 0;
	for (size_t p = text.find("%arg", at); p != std::string::npos && p < arrow; p = text.find("%arg", p + 4)) {
		++*params;
	}
	results->clear();
	for (size_t p = text.find("tensor<", arrow); p != std::string::npos && p < brace; p = text.find("tensor<", p + 7)) {
		const size_t gt = text.find('>', p);
		const std::string body = text.substr(p + 7, gt - p - 7);
		const std::string elem = body.substr(body.rfind('x') == std::string::npos ? 0 : body.rfind('x') + 1);
		results->push_back(elem == "bf16" ? StablehloDtype::BF16 : elem == "i32" ? StablehloDtype::I32 : StablehloDtype::F32);
	}
	return true;
}

// One StableHLO module as a pipeline stage: module.mlir and in<k>.bin (the raw little-endian bytes of each
// parameter of @main, in order) come from `read`; the results go to `results` as out<k>.bin. Each input is
// uploaded and dropped before the next is read, so the caller never holds more than one.
//
// With `compare`, every result is compared with expect<k>.bin at `tolerance` relative L2 (of the result's own
// element type), and two more controls run: a result of zeros must miss the reference, and module_bad.mlir (a
// mutated module) must be refused or miss it. One control always runs: the first input one byte short must be
// refused. A missing module, input, reference or control module is a FAIL.
inline bool run_module(PJRT_Client *client, const Reader &read, Store &results, bool compare, double tolerance) {
	std::vector<uint8_t> blob;
	if (client == nullptr || !read("module.mlir", blob) || blob.empty()) {
		std::printf("PROBE pjrt module: FAIL no module.mlir\n");
		return false;
	}
	const std::string text(blob.begin(), blob.end());
	size_t params = 0;
	std::vector<StablehloDtype> types;
	if (!main_signature(text, &params, &types)) {
		std::printf("PROBE pjrt module: FAIL module.mlir has no func @main\n");
		return false;
	}
	std::vector<PJRT_Buffer *> bufs;
	size_t in_bytes = 0;
	bool ok = true;
	for (size_t k = 0; k < params && ok; ++k) {
		const std::string name = "in" + std::to_string(k) + ".bin";
		if (!read(name, blob)) {
			std::printf("PROBE pjrt module: FAIL @main has %zu parameters and there is no %s\n", params, name.c_str());
			ok = false;
			break;
		}
		bufs.push_back(upload(client, blob.data(), blob.size()));
		in_bytes += blob.size();
	}
	std::printf("PROBE pjrt module: %zu bytes of text, %zu inputs, %zu bytes\n", text.size(), bufs.size(), in_bytes);
	Run good;
	if (ok) {
		good = execute(client, text, bufs);
		ok = good.ok;
		if (!good.ok) {
			std::printf("PROBE pjrt module: FAIL %s\n", good.error.c_str());
		}
	}
	size_t compared = 0;
	std::vector<std::vector<uint8_t>> expect(good.out.size());
	for (size_t k = 0; k < good.out.size() && good.ok; ++k) {
		const std::string name = "out" + std::to_string(k) + ".bin";
		std::printf("PROBE pjrt module: %s bytes=%zu fnv1a=%016llx\n", name.c_str(), good.out[k].size(), (unsigned long long)fnv1a(good.out[k]));
		results[name] = good.out[k];
		if (!compare) {
			continue;
		}
		const StablehloDtype d = k < types.size() ? types[k] : StablehloDtype::F32;
		if (!read("expect" + std::to_string(k) + ".bin", expect[k]) || expect[k].empty()) {
			std::printf("PROBE pjrt module: FAIL no expect%zu.bin to compare %s with\n", k, name.c_str());
			ok = false;
			continue;
		}
		++compared;
		const Compared c = compare_to(expect[k].data(), good.out[k], expect[k].size(), d);
		const Compared dead = compare_to(expect[k].data(), std::vector<uint8_t>(expect[k].size(), 0), expect[k].size(), d);
		const bool met = c.rel_l2 <= tolerance;
		const bool dead_fails = dead.rel_l2 > tolerance;
		std::printf("PROBE pjrt module: %s against expect%zu.bin: differing=%zu/%zu max_ulp=%lld rel_l2=%.3e max_abs=%.3e tolerance=%.1e %s; control: zeros for a result rel_l2=%.3f %s\n",
				name.c_str(), k, c.differing, c.n, (long long)c.max_ulp, c.rel_l2, c.max_abs, tolerance, met ? "met" : "NOT MET", dead.rel_l2,
				dead_fails ? "misses, as it must" : "MEETS it: the reference is dead");
		ok = ok && met && dead_fails;
	}
	if (compare && good.ok) {
		if (!read("module_bad.mlir", blob) || blob.empty()) {
			std::printf("PROBE pjrt module: FAIL a reference without module_bad.mlir (the control)\n");
			ok = false;
		} else {
			const Run bad = execute(client, std::string(blob.begin(), blob.end()), bufs);
			bool differs = !bad.ok || bad.out.size() != good.out.size();
			for (size_t k = 0; k < bad.out.size() && !differs; ++k) {
				differs = !expect[k].empty() && compare_to(expect[k].data(), bad.out[k], expect[k].size(), k < types.size() ? types[k] : StablehloDtype::F32).rel_l2 > tolerance;
			}
			std::printf("PROBE pjrt module: control module_bad.mlir %s\n", differs ? "is refused or misses the reference, as it must" : "MEETS the reference");
			ok = ok && differs;
		}
	}
	if (good.ok && !bufs.empty()) {
		std::vector<PJRT_Buffer *> shorter = bufs;
		shorter[0] = read("in0.bin", blob) && blob.size() > 1 ? upload(client, blob.data(), blob.size() - 1) : nullptr;
		const Run cut = execute(client, text, shorter);
		std::printf("PROBE pjrt module: control an input one byte short is %s\n", cut.ok ? "ACCEPTED" : "refused, as it must be");
		ok = ok && !cut.ok;
		drop(shorter[0]);
	}
	for (PJRT_Buffer *b : bufs) {
		drop(b);
	}
	std::printf("PROBE pjrt module SUMMARY inputs=%zu outputs=%zu compared=%zu %s\n", bufs.size(), good.out.size(), compared, ok ? "ok" : "FAILED");
	return ok;
}

} // namespace stablehlo_gate
