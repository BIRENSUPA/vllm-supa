/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */
#include "core/registration.h"
#include "supa/supa_kernel_ops.h"

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability) {
  return true;
}

bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability) {
  return true;
}

bool cutlass_group_gemm_supported(int64_t cuda_device_capability) {
  return true;
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def(
      "supa_fp8_blockwise_scaled_mm(Tensor(a!) out, Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b) "
      "-> Tensor(a!)");
  m.impl("supa_fp8_blockwise_scaled_mm", torch::kPrivateUse1, &supa_fp8_blockwise_scaled_mm);

  m.def(
      "per_token_group_quant_8bit_v2(Tensor input, Tensor! output_q, Tensor! output_s, int group_size,"
      " float eps, float fp8_min, float fp8_max, bool scale_ue8m0, bool fuse_silu_and_mul, Tensor? masked_m) -> ()");
  m.impl("per_token_group_quant_8bit_v2", torch::kPrivateUse1, &sgl_per_token_group_quant_8bit_v2);

  m.def(
      "per_token_group_quant_8bit_v2_interleaved(Tensor input, Tensor! output_q, Tensor! output_s,"
      " int group_size, float eps, float fp8_min, float fp8_max, bool scale_ue8m0) -> ()");
  m.impl(
      "per_token_group_quant_8bit_v2_interleaved",
      torch::kPrivateUse1,
      &sgl_per_token_group_quant_8bit_v2_interleaved);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, m) {
  m.impl("cutlass_scaled_mm_supports_fp8", &cutlass_scaled_mm_supports_fp8);
  m.impl("cutlass_scaled_mm_supports_block_fp8", &cutlass_scaled_mm_supports_block_fp8);
  m.impl("cutlass_group_gemm_supported", &cutlass_group_gemm_supported);
}
