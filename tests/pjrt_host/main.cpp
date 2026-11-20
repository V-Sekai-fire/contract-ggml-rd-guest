// pjrt-rd on a host, with ggml's CPU backend as the device.
//
//   pjrt_host cases [name|all]          the StableHLO gate (guest/ggml_test/stablehlo_gate.h) in functional mode:
//                                       every case within the bar of XLA CPU's output, every control and refusal held
//   pjrt_host run <dir> [tolerance]     one module as a pipeline stage: <dir>/module.mlir on <dir>/in<k>.bin, writes
//                                       <dir>/out<k>.bin; with a tolerance, compares with <dir>/expect<k>.bin and
//                                       runs <dir>/module_bad.mlir as the control
//   pjrt_host lower <file.mlir>         compiles only; prints "ok" or the refusal (the coverage report uses it)
//
// The kernels here are ggml-cpu's, not ggml-rd's, so this checks the lowering and not bit equality with XLA: the
// guest probe (pjrt stablehlo:all, in the engine) is the gate for that.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <pmmintrin.h>
#include <xmmintrin.h>
#endif

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "pjrt_rd.h"
#include "stablehlo_gate.h"

namespace {

bool read_file(const std::string &path, std::vector<uint8_t> &out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		return false;
	}
	out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
	return true;
}

} // namespace

int main(int argc, char **argv) {
	const std::string cmd = argc > 1 ? argv[1] : "";
	if (cmd != "cases" && cmd != "run" && cmd != "lower") {
		std::printf("usage: pjrt_host cases [name|all] | run <dir> [tolerance] | lower <file.mlir>\n");
		return 2;
	}
#if defined(__x86_64__) || defined(_M_X64)
	// XLA CPU runs with denormals read as zero and flushed to zero; so does this device (threads inherit it).
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif
	ggml_backend_t be = ggml_backend_cpu_init();
	if (be == nullptr) {
		std::printf("RESULT: FAIL no CPU backend\n");
		return 1;
	}
	ggml_backend_cpu_set_n_threads(be, 4);
	pjrt_rd::attach(be);
	stablehlo_gate::g_api = GetPjrtApi();
	PJRT_Client *client = stablehlo_gate::make_client();
	bool ok = false;
	if (cmd == "cases") {
		// Cases ggml-cpu cannot meet, with what was measured. Each must miss the bar, so the list only shrinks.
		const std::map<std::string, std::string> device_differs = {
			{ "log_stress", "ggml-cpu's log is the C library's logf, which returns about -103.97 for a denormal; XLA CPU reads a denormal as zero "
							"and returns -inf (201 of 4096 inputs differ, the denormals by class)" },
		};
		ok = stablehlo_gate::run_cases(client, argc > 2 ? argv[2] : "all", stablehlo_gate::Mode::Functional, &device_differs);
	} else if (cmd == "lower" && argc > 2) {
		std::vector<uint8_t> text;
		if (!read_file(argv[2], text)) {
			std::printf("RESULT: FAIL cannot read %s\n", argv[2]);
			return 1;
		}
		PJRT_Program prog;
		std::memset(&prog, 0, sizeof prog);
		prog.struct_size = PJRT_Program_STRUCT_SIZE;
		prog.code = reinterpret_cast<char *>(text.data());
		prog.code_size = text.size();
		prog.format = "mlir";
		prog.format_size = 4;
		PJRT_ARGS(PJRT_Client_Compile_Args, cc);
		cc.client = client;
		cc.program = &prog;
		PJRT_Error_Code code = PJRT_Error_Code_OK;
		const std::string err = stablehlo_gate::take(stablehlo_gate::g_api->PJRT_Client_Compile(&cc), &code);
		if (err.empty()) {
			PJRT_ARGS(PJRT_LoadedExecutable_Destroy_Args, ld);
			ld.executable = cc.executable;
			stablehlo_gate::take(stablehlo_gate::g_api->PJRT_LoadedExecutable_Destroy(&ld));
			std::printf("LOWER ok\n");
		} else {
			std::printf("LOWER refused code=%d %s\n", int(code), err.c_str());
		}
		ok = err.empty();
	} else if (cmd == "run" && argc > 2) {
		const std::string dir = argv[2];
		const bool compare = argc > 3;
		const double tolerance = compare ? std::strtod(argv[3], nullptr) : 1e-4;
		const stablehlo_gate::Reader read = [&dir](const std::string &name, std::vector<uint8_t> &bytes) { return read_file(dir + "/" + name, bytes); };
		stablehlo_gate::Store results;
		ok = (!compare || tolerance > 0.0) && stablehlo_gate::run_module(client, read, results, compare, tolerance);
		for (const std::pair<const std::string, std::vector<uint8_t>> &o : results) {
			std::ofstream f(dir + "/" + o.first, std::ios::binary);
			f.write(reinterpret_cast<const char *>(o.second.data()), std::streamsize(o.second.size()));
			ok = ok && bool(f);
		}
	}
	if (client != nullptr) {
		stablehlo_gate::destroy_client(client);
	}
	pjrt_rd::attach(nullptr);
	ggml_backend_free(be);
	std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
	return ok ? 0 : 1;
}
