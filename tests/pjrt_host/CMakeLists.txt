cmake_minimum_required(VERSION 3.16)
project(pjrt_host C CXX)
# pjrt-rd on ggml's CPU backend, as a host program: no engine, no guest, no GPU.
#   cmake -S tests/pjrt_host -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_ROOT=<ggml checkout>
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED GGML_ROOT)
	message(FATAL_ERROR "pass -DGGML_ROOT=<the ggml checkout>")
endif()
get_filename_component(RD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(${GGML_ROOT} ggml EXCLUDE_FROM_ALL)
add_executable(pjrt_host
	main.cpp
	rd_pack_stubs.cpp
	${RD_ROOT}/guest/pjrt-rd/pjrt_rd.cpp
	${RD_ROOT}/guest/pjrt-rd/program.cpp
	${RD_ROOT}/guest/pjrt-rd/stablehlo.cpp
	${RD_ROOT}/guest/ggml-rd/ops/xla_compare.cpp
	${RD_ROOT}/guest/ggml-rd/ops/rsqrt_zen2.cpp)
target_include_directories(pjrt_host PRIVATE ${RD_ROOT}/guest/pjrt-rd ${RD_ROOT}/guest/ggml-rd ${RD_ROOT}/guest/ggml_test
	${RD_ROOT}/kernels/ggml ${GGML_ROOT}/src)
# The custom ops' CPU functions use fused multiply-adds on purpose (std::fma); nothing else may contract.
target_compile_options(pjrt_host PRIVATE -ffp-contract=off)
target_link_libraries(pjrt_host PRIVATE ggml ggml-cpu ggml-base)
