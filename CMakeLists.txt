cmake_minimum_required(VERSION 3.26)
project(vllm_supa LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep CMake installs local to this plugin by default.  An explicit
# -DCMAKE_INSTALL_PREFIX still takes precedence for packaging or staging.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
      "Install prefix for vllm-supa extensions" FORCE)
endif()

# CUDA by default, can be overridden by using -DVLLM_TARGET_DEVICE=... (used by setup.py)
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")
message(STATUS "Enable coverage: ${ENABLE_COVERAGE}")

# Marlin kernels significantly increase build time. Disable this option to
# skip their generation and compilation; corresponding ops will be unavailable.
option(USE_MARLIN_KERNELS "Build the Marlin and Marlin-MOE GEMM kernels" ON)
message(STATUS "Use Marlin kernels: ${USE_MARLIN_KERNELS}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11" "3.12" "3.13")
find_python_from_executable(${Python3_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")

set(VLLM_GPU_LANG "CUDA")

# CUDA runtime / math libs via gstub; enable the CUDA language.
if(VLLM_GPU_LANG STREQUAL "CUDA")
  if(NOT DEFINED ENV{GSTUB_DIR})
    message(FATAL_ERROR "GSTUB_DIR environment variable not set. Please ensure the suda package is installed")
  endif()
  set(GSTUB_DIR $ENV{GSTUB_DIR})
  set(CUDAToolkit_ROOT "${GSTUB_DIR}/env")
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 90)

  # CUDA toolkit version from nvcc --version (replaces the former SUDA_VERSION
  # provided by the suda cmake package).
  execute_process(
    COMMAND nvcc --version
    OUTPUT_VARIABLE _NVCC_VER_OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _NVCC_RC)
  if(_NVCC_RC EQUAL 0 AND _NVCC_VER_OUT MATCHES "release ([0-9]+\\.[0-9]+)")
    set(CUDA_VERSION "${CMAKE_MATCH_1}" CACHE STRING "CUDA version from nvcc")
    message(STATUS "Detected CUDA version (nvcc): ${CUDA_VERSION}")
  else()
    message(WARNING "Failed to parse CUDA version from `nvcc --version`; "
                    "CUDA_VERSION-driven flag gating will be skipped.")
  endif()
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
find_package(Torch REQUIRED)

# Get torch_supa package location, TORCH_SUPA_PATH & TORCH_SUPA_LIB is used in define_extension_target()
run_python(TORCH_SUPA_PATH
  "import torch_supa, os; print(os.path.dirname(torch_supa.__file__))"
  "Failed to locate torch_supa package")
message(STATUS "TorchSUPA root: ${TORCH_SUPA_PATH}")
file(GLOB TORCH_SUPA_LIB "${TORCH_SUPA_PATH}/lib/*.so")
message(STATUS "TorchSUPA Libs: ${TORCH_SUPA_LIB}")
if(NOT TORCH_SUPA_LIB)
    message(FATAL_ERROR "No .so files found in directory: ${TORCH_SUPA_LIB}")
endif()


#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})


#
# Use FetchContent for C++ dependencies that are compiled as part of vLLM's build process.
# setup.py will override FETCHCONTENT_BASE_DIR to play nicely with sccache.
# Each dependency that produces build artifacts should override its BINARY_DIR to avoid
# conflicts between build types. It should instead be set to ${CMAKE_BINARY_DIR}/<dependency>.
#
include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

#
# _C extension
#

set(VLLM_EXT_SRC
  "csrc/torch_bindings.cpp")


if(ENABLE_COVERAGE)
  message(STATUS "Enabling coverage instrumentation.")
  add_compile_options(--coverage)
  add_link_options(--coverage)
endif()

set_gencode_flags_for_srcs(
  SRCS "${VLLM_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

set_gencode_flags_for_srcs(
    SRCS "${VLLM_EXT_SRC}"
    CUDA_ARCHS "${SUPA_ARCH}")

message(STATUS "Enabling C extension.")
message(STATUS "VLLM_GPU_LANG: ${VLLM_GPU_LANG}")
message(STATUS "VLLM_EXT_SRC: ${VLLM_EXT_SRC}")
message(STATUS "VLLM_GPU_FLAGS: ${VLLM_GPU_FLAGS}")
message(STATUS "VLLM_GPU_ARCHES: ${VLLM_GPU_ARCHES}")
define_extension_target(
  _C
  DESTINATION vllm_supa
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)

# Link fake cuda stubs into _C (for cumem_allocator symbol resolution)
link_fake_cuda_stubs(_C)

#
# _C_stable_libtorch extension (ops registered via STABLE_TORCH_LIBRARY)
#
set(VLLM_STABLE_EXT_SRC
  "csrc/libtorch_stable/torch_bindings.cpp"
  "csrc/libtorch_stable/shim_supa.cpp"
  "csrc/libtorch_stable/activation_kernels.cu"
  "csrc/libtorch_stable/ngram_embedding_kernels.cu"
  "csrc/libtorch_stable/pos_encoding_kernels.cu"
  "csrc/libtorch_stable/layernorm_kernels.cu"
  "csrc/libtorch_stable/fused_qknorm_rope_kernel.cu"
  "csrc/libtorch_stable/layernorm_quant_kernels.cu"
  "csrc/libtorch_stable/quantization/gptq/q_gemm.cu"
  "csrc/libtorch_stable/quantization/w8a8/int8/scaled_quant.cu"
  "csrc/libtorch_stable/quantization/w8a8/fp8/common.cu"
  "csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
  "csrc/libtorch_stable/quantization/gguf/gguf_kernel.cu"
  "csrc/libtorch_stable/quantization/activation_kernels.cu"
  "csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu"
  "csrc/libtorch_stable/cache_kernels.cu"
  "csrc/libtorch_stable/cache_kernels_fused.cu"
  "csrc/libtorch_stable/attention/merge_attn_states.cu"
  "csrc/libtorch_stable/sampler.cu"
  "csrc/libtorch_stable/topk.cu"
  "csrc/libtorch_stable/cooperative_topk.cu"
  "csrc/libtorch_stable/mamba/selective_scan_fwd.cu"
  "csrc/libtorch_stable/cuda_utils_kernels.cu"
  "csrc/libtorch_stable/custom_all_gather_reduce_scatter.cu"
  "csrc/libtorch_stable/custom_all_gather_reduce_scatter_ops.cpp"
  "csrc/libtorch_stable/custom_all_reduce.cu"
  "csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu"
  "csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu"
  "csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu"
  "csrc/libtorch_stable/minimax_reduce_rms_kernel.cu"
  "csrc/libtorch_stable/cuda_view.cu"
  "csrc/libtorch_stable/fp32_router_gemm.cu"
  "csrc/libtorch_stable/fp32_router_gemm_entry.cu"
  )

if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_STABLE_EXT_SRC
    "csrc/libtorch_stable/permute_cols.cu"
    "csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu"
    "csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu")
endif()

# Only build Marlin kernels if we are building for at least some compatible archs.
# Keep building Marlin for 9.0 as there are some group sizes and shapes that
# are not supported by Machete yet.

#
# For the Marlin kernels we automatically generate sources for various
# preselected input type pairs and schedules.
# Generate sources:
set(MARLIN_GEN_SCRIPT
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/quantization/marlin/generate_kernels.py)
file(MD5 ${MARLIN_GEN_SCRIPT} MARLIN_GEN_SCRIPT_HASH)
list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
set(MARLIN_GEN_SCRIPT_HASH_AND_ARCH "${MARLIN_GEN_SCRIPT_HASH}(ARCH:${CUDA_ARCHS_STR})")

message(STATUS "Marlin generation script hash: ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")
message(STATUS "Last run Marlin generate script hash: $CACHE{MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")

if (NOT USE_MARLIN_KERNELS)
  message(STATUS "Marlin generation skipped, USE_MARLIN_KERNELS is OFF.")
elseif (NOT DEFINED CACHE{MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
    OR NOT $CACHE{MARLIN_GEN_SCRIPT_HASH_AND_ARCH} STREQUAL ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=$ENV{PYTHONPATH}
      ${Python_EXECUTABLE} ${MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
    RESULT_VARIABLE marlin_generation_result
    OUTPUT_VARIABLE marlin_generation_result
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
  )

  if (NOT marlin_generation_result EQUAL 0)
    message(FATAL_ERROR "Marlin generation failed."
                        " Result: \"${marlin_generation_result}\""
                        "\nCheck the log for details: "
                        "${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log")
  else()
    set(MARLIN_GEN_SCRIPT_HASH_AND_ARCH ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
        CACHE STRING "Last run Marlin generate script hash and arch" FORCE)
    message(STATUS "Marlin generation completed successfully.")
  endif()
else()
  message(STATUS "Marlin generation script has not changed, skipping generation.")
endif()

# The generated kernels and marlin.cu (which #includes the generated
# kernel_selector.h) are only compiled when Marlin is enabled.  The repack and
# preprocess ops do not depend on the generated kernels, so they stay in.
if (USE_MARLIN_KERNELS)
  file(GLOB MARLIN_TEMPLATE_KERNEL_SRC "csrc/libtorch_stable/quantization/marlin/sm80_kernel_*_float16.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_TEMPLATE_KERNEL_SRC}"
    CUDA_ARCHS "${MARLIN_ARCHS}")

  set_source_files_properties(${MARLIN_TEMPLATE_KERNEL_SRC}
    PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")

  list(APPEND VLLM_STABLE_EXT_SRC ${MARLIN_TEMPLATE_KERNEL_SRC})

  file(GLOB MARLIN_TEMPLATE_BF16_KERNEL_SRC "csrc/libtorch_stable/quantization/marlin/sm80_kernel_*_bfloat16.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_TEMPLATE_BF16_KERNEL_SRC}"
    CUDA_ARCHS "${MARLIN_BF16_ARCHS}")

  set_source_files_properties(${MARLIN_TEMPLATE_BF16_KERNEL_SRC}
    PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")

  list(APPEND VLLM_STABLE_EXT_SRC ${MARLIN_TEMPLATE_BF16_KERNEL_SRC})

  set(MARLIN_SRCS
      "csrc/libtorch_stable/quantization/marlin/marlin.cu"
      "csrc/libtorch_stable/quantization/marlin/marlin_int4_fp8_preprocess.cu"
      "csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu"
      "csrc/libtorch_stable/quantization/marlin/awq_marlin_repack.cu")
else()
  set(MARLIN_SRCS
      "csrc/libtorch_stable/quantization/marlin/marlin_int4_fp8_preprocess.cu"
      "csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu"
      "csrc/libtorch_stable/quantization/marlin/awq_marlin_repack.cu")
endif()

set_gencode_flags_for_srcs(
  SRCS "${MARLIN_SRCS}"
  CUDA_ARCHS "${MARLIN_OTHER_ARCHS}")

set_source_files_properties(${MARLIN_SRCS}
  PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
list(APPEND VLLM_STABLE_EXT_SRC "${MARLIN_SRCS}")

if (USE_MARLIN_KERNELS)
  message(STATUS "Building Marlin kernels for archs: ${MARLIN_OTHER_ARCHS}")
else()
  message(STATUS "Not building Marlin kernels, USE_MARLIN_KERNELS is OFF. "
                 "marlin_gemm will have no implementation registered.")
endif()

#
# Machete kernels skipp need cutlass

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set_gencode_flags_for_srcs(
    SRCS "${VLLM_STABLE_EXT_SRC}"
    CUDA_ARCHS "${SUPA_ARCH}")
endif()

message(STATUS "Enabling C_stable extension.")
define_extension_target(
  _C_stable_libtorch
  DESTINATION vllm_supa
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_STABLE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)

# Link fake cuda stubs into _C_stable_libtorch
link_fake_cuda_stubs(_C_stable_libtorch)

# Set TORCH_TARGET_VERSION for stable ABI compatibility.
# This ensures we only use C-shim APIs available in PyTorch 2.11.
# _C_stable_libtorch is abi compatible with PyTorch >= TORCH_TARGET_VERSION
# which is currently set to 2.11.
target_compile_definitions(_C_stable_libtorch PRIVATE
  TORCH_TARGET_VERSION=0x020B000000000000ULL)

# Needed to use cuda APIs from C-shim
target_compile_definitions(_C_stable_libtorch PRIVATE
  USE_CUDA)
#
# _moe_C extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/libtorch_stable/moe/torch_bindings.cpp"
  "csrc/libtorch_stable/shim_supa.cpp"
  "csrc/libtorch_stable/moe/moe_align_sum_kernels.cu"
  "csrc/libtorch_stable/moe/topk_softmax_kernels.cu"
  "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_MOE_EXT_SRC
    "csrc/libtorch_stable/moe/moe_wna16.cu"
    "csrc/libtorch_stable/moe/grouped_topk_kernels.cu"
    )
endif()
# not support cutlass
# if(VLLM_GPU_LANG STREQUAL "CUDA")
#   set(MOE_PERMUTE_SRC
#       "csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
#       "csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu")

#   list(APPEND VLLM_MOE_EXT_SRC "${MOE_PERMUTE_SRC}")
# endif()

list(APPEND VLLM_MOE_EXT_SRC "${VLLM_MOE_WNA16_SRC}")



#
# For the Marlin MOE kernels we automatically generate sources for various
# preselected input type pairs and schedules.
# Generate sources:
set(MOE_MARLIN_GEN_SCRIPT
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py)
file(MD5 ${MOE_MARLIN_GEN_SCRIPT} MOE_MARLIN_GEN_SCRIPT_HASH)
list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
set(MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH "${MOE_MARLIN_GEN_SCRIPT_HASH}(ARCH:${CUDA_ARCHS_STR})")

message(STATUS "Marlin MOE generation script hash with arch: ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")
message(STATUS "Last run Marlin MOE generate script hash with arch: $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")

if (NOT USE_MARLIN_KERNELS)
  message(STATUS "Marlin MOE generation skipped, USE_MARLIN_KERNELS is OFF.")
elseif (NOT DEFINED CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
    OR NOT $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH} STREQUAL ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=$ENV{PYTHONPATH}
      ${Python_EXECUTABLE} ${MOE_MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
    RESULT_VARIABLE moe_marlin_generation_result
    OUTPUT_VARIABLE moe_marlin_generation_output
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
  )

  if (NOT moe_marlin_generation_result EQUAL 0)
    message(FATAL_ERROR "Marlin MOE generation failed."
                        " Result: \"${moe_marlin_generation_result}\""
                        "\nCheck the log for details: "
                        "${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log")
  else()
    set(MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
        CACHE STRING "Last run Marlin MOE generate script hash" FORCE)
    message(STATUS "Marlin MOE generation completed successfully.")
  endif()
else()
  message(STATUS "Marlin MOE generation script has not changed, skipping generation.")
endif()

# The generated kernels and ops.cu (which #includes the generated
# kernel_selector.h) are only compiled when Marlin is enabled.
if (USE_MARLIN_KERNELS)
  file(GLOB MARLIN_MOE_SRC "csrc/libtorch_stable/moe/marlin_moe_wna16/sm80_kernel_*.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_MOE_SRC}"
    CUDA_ARCHS "${MARLIN_MOE_ARCHS}")

  set_source_files_properties(${MARLIN_MOE_SRC}
    PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")

  list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SRC})


  set(MARLIN_MOE_OTHER_SRC "csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_MOE_OTHER_SRC}"
    CUDA_ARCHS "${MARLIN_MOE_OTHER_ARCHS}")

  set_source_files_properties(${MARLIN_MOE_OTHER_SRC}
    PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")

  list(APPEND VLLM_MOE_EXT_SRC "${MARLIN_MOE_OTHER_SRC}")

  message(STATUS "Building Marlin MOE kernels for archs: ${MARLIN_MOE_OTHER_ARCHS}")
else()
  message(STATUS "Not building Marlin MOE kernels, USE_MARLIN_KERNELS is OFF. "
                 "moe_wna16_marlin_gemm will have no implementation registered.")
endif()

# DeepSeek V3 router GEMM kernel - requires SM90+

cuda_archs_loose_intersection(DSV3_ROUTER_GEMM_ARCHS "9.0a;10.0a;10.1a;10.3a" "${CUDA_ARCHS}")


set(DSV3_ROUTER_GEMM_SRC
  "csrc/libtorch_stable/moe/dsv3_router_gemm_entry.cu"
  "csrc/libtorch_stable/moe/dsv3_router_gemm_float_out.cu"
  "csrc/libtorch_stable/moe/dsv3_router_gemm_bf16_out.cu")
set_gencode_flags_for_srcs(
  SRCS "${DSV3_ROUTER_GEMM_SRC}"
  CUDA_ARCHS "${DSV3_ROUTER_GEMM_ARCHS}")
list(APPEND VLLM_MOE_EXT_SRC "${DSV3_ROUTER_GEMM_SRC}")
message(STATUS "Building DSV3 router GEMM kernel for archs: ${DSV3_ROUTER_GEMM_ARCHS}")


message(STATUS "Enabling _moe_C_stable_libtorch extension.")
define_extension_target(
  _moe_C_stable_libtorch
  DESTINATION vllm_supa
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)

target_compile_definitions(_moe_C_stable_libtorch PRIVATE
  TORCH_TARGET_VERSION=0x020B000000000000ULL)
target_compile_definitions(_moe_C_stable_libtorch PRIVATE USE_CUDA)



find_path(SUDNN_INCLUDE_PATH NAMES sudnn.h HINTS ${FULLSTACK_PATH}/sudnn/include/sudnn REQUIRED)
find_library(SUDNN_LIB NAMES sudnn HINTS ${FULLSTACK_PATH}/sudnn/lib ${FULLSTACK_PATH}/sudnn/lib64 REQUIRED)
find_path(
  SUPACCL_INCLUDE_PATH
  NAMES sub/block/block_scan.suh
  HINTS ${FULLSTACK_PATH}/supaccl/include REQUIRED)

message(STATUS "SUDNN_INCLUDE_PATH: ${SUDNN_INCLUDE_PATH}")
message(STATUS "SUDNN_LIB: ${SUDNN_LIB}")
message(STATUS "SUPACCL_INCLUDE_PATH: ${SUPACCL_INCLUDE_PATH}")

# _supa_C
set(_SUPA_C_SRC
    "csrc/supa/torch_bindings.cpp"
    "csrc/supa/fp8_blockwise_gemm_kernel.su"
    "csrc/supa/fp8_gemm_kernel.cpp"
    "csrc/supa/per_token_group_quant_8bit_v2.su"
)

set(SUPA_C_BRCC_FLAGS
    "-foffload-lto=full"
    "-mllvm"
    "-br-tcore-async-src-register=0"
    "-mllvm"
    "-br-allow-tensor-reg-spill=1"
    "-mllvm"
    "-br-preallocate-tensor-regs=0"
    "-mllvm"
    "-unroll-threshold-default=4000"
    "-use-mira=false"
    "-Xarch_device"
    "-ffast-math"
)

set(INCLUDE_DIRS
    ${TORCH_SUPA_PATH}/include
    ${TORCH_SUPA_PATH}/include/sutlass/include/
    ${SUPA_INCLUDE_DIRS}
    ${SUDNN_INCLUDE_PATH}
    ${SUPACCL_INCLUDE_PATH}
    ${TORCH_SUPA_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

message(STATUS "Enabling supa_C extension.")
ADD_SUPA_PYTHON_LIBRARY(
  _supa_C
  DESTINATION vllm_supa
  SOURCES ${_SUPA_C_SRC}
  SU_FLAGS ${SUPA_C_BRCC_FLAGS}
  INCLUDE_DIRECTORIES ${INCLUDE_DIRS}
  LIBRARIES sublas sudnn torch_supa supa
  LIBRARY_DIRS ${TORCH_LIBRARIES} ${TORCH_SUPA_PATH}/lib)
