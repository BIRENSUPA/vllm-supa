// TODO: Remove this shim and call the stable API directly once torch_supa available.
#pragma push_macro("TORCH_STABLE_ONLY")
#pragma push_macro("TORCH_TARGET_VERSION")
#undef TORCH_STABLE_ONLY
#undef TORCH_TARGET_VERSION

#pragma push_macro("USE_CUDA")
#ifndef USE_CUDA
#define USE_CUDA
#endif
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/c/shim.h>
#pragma pop_macro("USE_CUDA")

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <torch/csrc/utils/cpp_stacktraces.h>
#include <cstring>

AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index,
    void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(cudaStream_t*)(ret_stream) = at::cuda::getCurrentCUDAStream(device_index);
  });
}

AOTITorchError torch_get_current_cuda_blas_handle(void** ret_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(cublasHandle_t*)(ret_handle) = at::cuda::getCurrentCUDABlasHandle();
  });
}

AOTITorchError torch_c10_cuda_check_msg(
    int32_t err,
    const char* filename,
    const char* function_name,
    uint32_t line_number,
    bool include_device_assertions,
    char** error_msg) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *error_msg = nullptr;
    try {
      c10::cuda::c10_cuda_check_implementation(
          err, filename, function_name, line_number, include_device_assertions);
    } catch (const c10::AcceleratorError& e) {
      const char* what_str = torch::get_cpp_stacktraces_enabled()
          ? e.what()
          : e.what_without_backtrace();
      size_t msg_len = std::strlen(what_str);
      *error_msg = new char[msg_len + 1];
      std::memcpy(*error_msg, what_str, msg_len + 1);
    }
  });
}

void torch_c10_cuda_free_error_msg(char* error_msg) {
  delete[] error_msg;
}

#pragma pop_macro("TORCH_TARGET_VERSION")
#pragma pop_macro("TORCH_STABLE_ONLY")
