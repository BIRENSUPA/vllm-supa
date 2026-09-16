/* Copyright (C) 2025-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/utils/ParamsHash.h>
#include <sudnn.h>
#include <sudnn_frontend/sudnn_frontend.h>
#include <supa_runtime.h>
#include <torch/all.h>
#include <torch_supa/csrc/core/supa/SUPACachingAllocator.h>
#include <torch_supa/csrc/core/supa/SUPAFunctions.h>
#include <torch_supa/csrc/core/supa/SUPAStream.h>
#include <torch_supa/csrc/utils/Utils.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define SGL_CHECK(condition, ...)                                                          \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "Check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      std::fprintf(stderr, __VA_ARGS__);                                                   \
      std::fprintf(stderr, "\n");                                                          \
      std::abort();                                                                        \
    }                                                                                      \
  } while (0)

#define _DISPATCH_CASE_F16(c_type, ...) \
  case at::ScalarType::Half: {          \
    using c_type = __half;              \
    return __VA_ARGS__();               \
  }

#define _DISPATCH_CASE_BF16(c_type, ...) \
  case at::ScalarType::BFloat16: {       \
    using c_type = __br_bfloat16;        \
    return __VA_ARGS__();                \
  }

#define _DISPATCH_CASE_FP8_E4M3(c_type, ...) \
  case at::ScalarType::Float8_e4m3fn: {      \
    using c_type = __br_fp8_e4m3;            \
    return __VA_ARGS__();                    \
  }

#define _DISPATCH_CASE_FP8_E5M2(c_type, ...) \
  case at::ScalarType::Float8_e5m2: {        \
    using c_type = __br_fp8_e5m2;            \
    return __VA_ARGS__();                    \
  }

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)                 \
  [&]() -> bool {                                                                        \
    switch (pytorch_dtype) {                                                             \
      _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                            \
      _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                           \
      default:                                                                           \
        std::ostringstream oss;                                                          \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype; \
        TORCH_CHECK(false, oss.str());                                                   \
        return false;                                                                    \
    }                                                                                    \
  }()

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(pytorch_dtype, c_type, ...)           \
  [&]() -> bool {                                                                        \
    switch (pytorch_dtype) {                                                             \
      case at::ScalarType::Float: {                                                      \
        using c_type = float;                                                            \
        return __VA_ARGS__();                                                            \
      }                                                                                  \
        _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                          \
        _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                         \
      default:                                                                           \
        std::ostringstream oss;                                                          \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype; \
        TORCH_CHECK(false, oss.str());                                                   \
        return false;                                                                    \
    }                                                                                    \
  }()

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(pytorch_dtype, c_type, ...)                      \
  [&]() -> bool {                                                                            \
    switch (pytorch_dtype) {                                                                 \
      _DISPATCH_CASE_FP8_E4M3(c_type, __VA_ARGS__)                                           \
      _DISPATCH_CASE_FP8_E5M2(c_type, __VA_ARGS__)                                           \
      default:                                                                               \
        std::ostringstream oss;                                                              \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch fp8 data type " << pytorch_dtype; \
        TORCH_CHECK(false, oss.str());                                                       \
        return false;                                                                        \
    }                                                                                        \
  }()

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE(pytorch_dtype, c_type, ...)                      \
  [&]() -> bool {                                                                        \
    switch (pytorch_dtype) {                                                             \
      _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                            \
      _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                           \
      _DISPATCH_CASE_FP8_E4M3(c_type, __VA_ARGS__)                                       \
      _DISPATCH_CASE_FP8_E5M2(c_type, __VA_ARGS__)                                       \
      default:                                                                           \
        std::ostringstream oss;                                                          \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype; \
        TORCH_CHECK(false, oss.str());                                                   \
        return false;                                                                    \
    }                                                                                    \
  }()

#define _DISPATCH_SWITCH(var_name, cond, ...)                                           \
  [&]() -> bool {                                                                       \
    switch (cond) {                                                                     \
      __VA_ARGS__                                                                       \
      default:                                                                          \
        std::ostringstream oss;                                                         \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch " var_name " " << int(cond); \
        TORCH_CHECK(false, oss.str());                                                  \
        return false;                                                                   \
    }                                                                                   \
  }()

#define _DISPATCH_SWITCH_U16x2(var1_name, var2_name, cond1, cond2, ...)                                             \
  [&]() -> bool {                                                                                                   \
    switch (pack_u16(cond1, cond2)) {                                                                               \
      __VA_ARGS__                                                                                                   \
      default:                                                                                                      \
        std::ostringstream oss;                                                                                     \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch (" var1_name ", " var2_name "): (" << int(cond1) << ", " \
            << int(cond2) << ")";                                                                                   \
        TORCH_CHECK(false, oss.str());                                                                              \
        return false;                                                                                               \
    }                                                                                                               \
  }()

#define _DISPATCH_CASE(case_expr, case_var, ...) \
  case case_expr: {                              \
    constexpr auto case_var = case_expr;         \
    return __VA_ARGS__();                        \
  }

#define _DISPATCH_CASE_U16x2(case_expr1, case_expr2, case_var1, case_var2, ...) \
  case pack_u16(case_expr1, case_expr2): {                                      \
    constexpr auto case_var1 = case_expr1;                                      \
    constexpr auto case_var2 = case_expr2;                                      \
    return __VA_ARGS__();                                                       \
  }

#define DISPATCH_BOOL(expr, const_expr, ...) \
  [&]() -> bool {                            \
    if (expr) {                              \
      constexpr bool const_expr = true;      \
      return __VA_ARGS__();                  \
    } else {                                 \
      constexpr bool const_expr = false;     \
      return __VA_ARGS__();                  \
    }                                        \
  }()

inline void check_shape(const at::Tensor& a, const at::Tensor& b, const char* a_name, const char* b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ", a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name, ".size(", i, ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads) \
  TORCH_CHECK(                                               \
      num_qo_heads % num_kv_heads == 0,                      \
      "num_qo_heads(",                                       \
      num_qo_heads,                                          \
      ") must be divisible by num_kv_heads(",                \
      num_kv_heads,                                          \
      ")")

#define CHECK_SUPA(x) TORCH_CHECK(torch_supa::utils::is_supa(x), #x " must be a SUPA tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimension")

#define CHECK_INPUT(x) \
  CHECK_SUPA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_SUPA(x);                           \
  CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

inline bool is_float8_tensor(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::Float8_e4m3fn || tensor.scalar_type() == at::ScalarType::Float8_e5m2;
}

struct supa_error : public std::runtime_error {
  /**
   * @brief Constructs a `supa_error` object with the given `message`.
   *
   * @param message The error char array used to construct `supa_error`
   */
  supa_error(const char* message) : std::runtime_error(message) {}
  /**
   * @brief Constructs a `supa_error` object with the given `message` string.
   *
   * @param message The `std::string` used to construct `supa_error`
   */
  supa_error(std::string const& message) : supa_error{message.c_str()} {}
};

#define CHECK_SUPA_SUCCESS(cmd)                                         \
  do {                                                                  \
    supaError_t e = cmd;                                                \
    if (e != supaSuccess) {                                             \
      std::stringstream _message;                                       \
      auto s = supaGetErrorString(e);                                   \
      _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
      throw supa_error(_message.str());                                 \
    }                                                                   \
  } while (0)

#define CHECK_IS_SUPA(x) TORCH_CHECK(x.device().is_supa(), #x " must be a SUPA tensor")
#define CHECK_IS_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SUPA_INPUT(x) \
  CHECK_IS_SUPA(x);         \
  CHECK_IS_CONTIGUOUS(x)

inline int getSMVersion() {
  int device{-1};
  CHECK_SUPA_SUCCESS(supaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CHECK_SUPA_SUCCESS(supaDeviceGetAttribute(&sm_major, supaDevAttrComputeCapabilityMajor, device));
  CHECK_SUPA_SUCCESS(supaDeviceGetAttribute(&sm_minor, supaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

inline bool isDeviceType(const std::string& device_type) {
  int deviceCount;
  CHECK_SUPA_SUCCESS(supaGetDeviceCount(&deviceCount));

  int device_id = -1;
  if (deviceCount >= 1) {
    CHECK_SUPA_SUCCESS(supaGetDevice(&device_id));
  } else {
    return false;
  }

  supaDeviceProp prop;
  CHECK_SUPA_SUCCESS(supaGetDeviceProperties(&prop, device_id));
  if (device_type == std::string(prop.name)) {
    return true;
  }
  return false;
}

inline bool getBoolEnv(char const* name) {
  char const* ebr = std::getenv(name);
  return ebr && ebr[0] == '1' && ebr[1] == '\0';
}

inline bool getEnvEnablePDL() {
  static std::once_flag flag;
  static bool enablePDL = false;
  std::call_once(flag, [&]() {
    // PDL will be enabled by setting the env variables `TRTLLM_ENABLE_PDL` to `1`
    enablePDL = getBoolEnv("TRTLLM_ENABLE_PDL");
  });
  return enablePDL;
}

#define DISPATCH_CASE_INTEGRAL_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

#define DISPATCH_CASE_FLOAT_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...) AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOAT_TYPES(__VA_ARGS__))

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#define WARP_SIZE 32

template <typename srcDtype>
__device__ __forceinline__ float castToFloat(srcDtype val) {
  return static_cast<float>(val);
}

template <typename dstDtype>
__device__ __forceinline__ dstDtype castFromFloat(float val) {
  return static_cast<dstDtype>(val);
}

constexpr float FP8_E4M3_MAX = 448.0f;

#define FULL_MASK 0xffffffff

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  int* addr_as_i = (int*)addr;
  int old = *addr_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ __forceinline__ float warpReduceMax(float value) {
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 16));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 8));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 4));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 2));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 1));
  return value;
}

__device__ __forceinline__ float blockReduceMax(float value) {
  static __shared__ float warpLevelMaxs[WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;

  value = warpReduceMax(value);

  if (laneId == 0) warpLevelMaxs[warpId] = value;
  __syncthreads();

  value = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelMaxs[laneId] : 0;
  if (warpId == 0) value = warpReduceMax(value);

  return value;
}

// Pads to a multiple of `alignment` rows.
inline torch::Tensor pad_tensor(const torch::Tensor& tensor, int64_t alignment = 4, bool is_column_major = false) {
  int64_t rows = tensor.size(0);
  int64_t cols = tensor.size(1);
  int64_t pad_rows = (alignment - (rows % alignment)) % alignment;  // Compute padding size

  if (pad_rows == 0) {
    return tensor;  // Already aligned
  }

  torch::Tensor padding = torch::zeros({pad_rows, cols}, tensor.options());
  torch::Tensor tensor_padded = torch::cat({tensor, padding}, 0);  // Pad along rows

  // Ensure column-major layout
  if (is_column_major) {
    return tensor_padded.t().contiguous().t();
  }
  return tensor_padded;
}

// Get the next power of 2 of a number
inline uint32_t next_pow2(uint32_t x) noexcept {
  if (x <= 1) return 1;
  return 1u << (32 - __builtin_clz(x - 1));
}

#define SGLANG_LDG(arg) *(arg)

#define SUPA_CHECK(func)                                                                                             \
  {                                                                                                                  \
    supaError_t e = (func);                                                                                          \
    if (e != supaSuccess) {                                                                                          \
      std::cerr << "SUPA Error: " << supaGetErrorString(e) << " (" << e << ") " << __FILE__ << ": line " << __LINE__ \
                << std::endl;                                                                                        \
      return e;                                                                                                      \
    }                                                                                                                \
  }

#define AT_SUDNN_FRONTEND_CHECK(EXPR, ...)                                                       \
  do {                                                                                           \
    auto error_object = EXPR;                                                                    \
    if (!error_object.is_good()) {                                                               \
      TORCH_CHECK_WITH(SuDNNError, false, "suDNN Frontend error: ", error_object.get_message()); \
    }                                                                                            \
  } while (0)
