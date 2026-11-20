// The two custom ops of the StableHLO lowering (ops/xla_compare.cpp, ops/rsqrt_zen2.cpp) register a ggml-rd
// packer next to their CPU function. On the host only the CPU function runs, so the packer's four entry points
// are stubs: registering does nothing and no kernel is ever found.
#include "rd_pack.h"

namespace ggml_rd {

void register_op(const OpEntry &) {}

int kernel_index(const char *) {
	return -1;
}

bool grid_1d(Pack &, uint64_t) {
	return false;
}

bool tensor_fits(const ggml_tensor *) {
	return false;
}

} // namespace ggml_rd
