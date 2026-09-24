/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/ATen.h>
#include <stdio.h>
#include <sublas.h>
#include <sublasLt.h>
#include <torch/all.h>
#include <torch_supa/csrc/core/supa/SUPACachingAllocator.h>
#include <torch_supa/csrc/core/supa/SUPAContext.h>
#include <torch_supa/csrc/aten/supa/SublasContext.h>

#define CHECK_SUBLAS_ERROR(error)                                                      \
  do {                                                                                 \
    if (error != SUBLAS_STATUS_SUCCESS) {                                              \
      printf("sublas error, at %s, : %d. error code %d\n", __FILE__, __LINE__, error); \
      return;                                                                          \
    }                                                                                  \
  } while (0)

void sublas_fp8_scaled_mm(
    const void* __restrict__ ptr_A,
    const void* __restrict__ ptr_B,
    void* __restrict__ ptr_D,
    const void* __restrict__ ptr_scale_A,
    const void* __restrict__ ptr_scale_B,
    const void* __restrict__ ptr_C,
    float beta,
    int m,
    int n,
    int k,
    const int quant_type,
    supaDataType out_type,
    supaStream_t stream,
    int32_t batch = 1,
    at::Tensor workspace_buffer = at::Tensor(),
    int64_t sublas_handle = -1) {
  sublasLtHandle_t handle =
      sublas_handle == -1 ? at::supa::getCurrentSuBlasLtHandle() : reinterpret_cast<sublasLtHandle_t>(sublas_handle);
  void* d_workspace = workspace_buffer.numel() == 0 ? c10::supa::SUPACachingAllocator::raw_alloc_with_stream(
                                                          SUBLAS_DEFAULT_WORKSPACE_SIZE, stream)
                                                    : workspace_buffer.data_ptr();
  sublasLtMatmulDesc_t operation_desc = nullptr;
  sublasLtEpilogue_t epilogue_mode = SUBLASLT_EPILOGUE_DEFAULT;
  sublasLtMatmulMatrixScale_t a_scale_mode = SUBLASLT_MATMUL_MATRIX_SCALE_END;
  sublasLtMatmulMatrixScale_t b_scale_mode = SUBLASLT_MATMUL_MATRIX_SCALE_END;

  if (0 == quant_type) {                                         // ds-r1-fp8 style
    a_scale_mode = SUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;      // per group-wise quant
    b_scale_mode = SUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;  // per block-wise quant
  } else if (1 == quant_type) {                                  // per token/channel-wise quant
    a_scale_mode = SUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    b_scale_mode = SUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
  } else if (2 == quant_type) {  // per tensor-wise quant
    a_scale_mode = SUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    b_scale_mode = SUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  } else if (3 == quant_type) {  // MXFP8 block-scaled (BR288), UE8M0 1x32 granularity
    a_scale_mode = SUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    b_scale_mode = SUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  }

  sublasOperation_t transA = SUBLAS_OP_N;  // A
  sublasOperation_t transB = SUBLAS_OP_T;  // B^T

  CHECK_SUBLAS_ERROR(sublasLtMatmulDescCreate(&operation_desc, SUBLAS_COMPUTE_32F, SUPA_R_32F));

  CHECK_SUBLAS_ERROR(
      sublasLtMatmulDescSetAttribute(operation_desc, SUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
  CHECK_SUBLAS_ERROR(
      sublasLtMatmulDescSetAttribute(operation_desc, SUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

  CHECK_SUBLAS_ERROR(sublasLtMatmulDescSetAttribute(
      operation_desc, SUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue_mode, sizeof(epilogue_mode)));
  CHECK_SUBLAS_ERROR(sublasLtMatmulDescSetAttribute(
      operation_desc, SUBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)));
  CHECK_SUBLAS_ERROR(sublasLtMatmulDescSetAttribute(
      operation_desc, SUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ptr_scale_A, sizeof(ptr_scale_A)));
  CHECK_SUBLAS_ERROR(sublasLtMatmulDescSetAttribute(
      operation_desc, SUBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)));
  CHECK_SUBLAS_ERROR(sublasLtMatmulDescSetAttribute(
      operation_desc, SUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &ptr_scale_B, sizeof(ptr_scale_B)));

  sublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr, d_desc = nullptr;

  CHECK_SUBLAS_ERROR(sublasLtMatrixLayoutCreate(&a_desc, SUPA_R_8F_E4M3, m, k, k));
  CHECK_SUBLAS_ERROR(sublasLtMatrixLayoutCreate(&b_desc, SUPA_R_8F_E4M3, n, k, k));
  CHECK_SUBLAS_ERROR(sublasLtMatrixLayoutCreate(&c_desc, out_type, m, n, n));
  CHECK_SUBLAS_ERROR(sublasLtMatrixLayoutCreate(&d_desc, out_type, m, n, n));

  const int32_t batch_count = batch;
  int64_t strideA = m * k;
  int64_t strideB = n * k;
  int64_t strideC = m * n;
  int64_t strideD = m * n;
  CHECK_SUBLAS_ERROR(
      sublasLtMatrixLayoutSetAttribute(a_desc, SUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
  CHECK_SUBLAS_ERROR(
      sublasLtMatrixLayoutSetAttribute(b_desc, SUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
  CHECK_SUBLAS_ERROR(
      sublasLtMatrixLayoutSetAttribute(c_desc, SUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
  CHECK_SUBLAS_ERROR(
      sublasLtMatrixLayoutSetAttribute(d_desc, SUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

  CHECK_SUBLAS_ERROR(
      sublasLtMatrixLayoutSetAttribute(a_desc, SUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
  CHECK_SUBLAS_ERROR(
      sublasLtMatrixLayoutSetAttribute(b_desc, SUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
  CHECK_SUBLAS_ERROR(
      sublasLtMatrixLayoutSetAttribute(c_desc, SUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
  CHECK_SUBLAS_ERROR(
      sublasLtMatrixLayoutSetAttribute(d_desc, SUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));

  sublasLtOrder_t order_a = SUBLASLT_ORDER_ROW;
  sublasLtOrder_t order_b = SUBLASLT_ORDER_ROW;
  CHECK_SUBLAS_ERROR(sublasLtMatrixLayoutSetAttribute(a_desc, SUBLASLT_MATRIX_LAYOUT_ORDER, &order_a, sizeof(order_a)));
  CHECK_SUBLAS_ERROR(sublasLtMatrixLayoutSetAttribute(b_desc, SUBLASLT_MATRIX_LAYOUT_ORDER, &order_b, sizeof(order_b)));
  sublasLtOrder_t order_cd = SUBLASLT_ORDER_ROW;
  CHECK_SUBLAS_ERROR(
      sublasLtMatrixLayoutSetAttribute(c_desc, SUBLASLT_MATRIX_LAYOUT_ORDER, &order_cd, sizeof(order_cd)));
  CHECK_SUBLAS_ERROR(
      sublasLtMatrixLayoutSetAttribute(d_desc, SUBLASLT_MATRIX_LAYOUT_ORDER, &order_cd, sizeof(order_cd)));

  constexpr float alpha = 1.0f;

  CHECK_SUBLAS_ERROR(sublasLtMatmul(
      handle,
      operation_desc,
      &alpha,
      ptr_A,
      a_desc,
      ptr_B,
      b_desc,
      &beta,
      ptr_C,
      c_desc,
      ptr_D,
      d_desc,
      NULL,
      d_workspace,
      SUBLAS_DEFAULT_WORKSPACE_SIZE,
      stream));

  c10::supa::SUPACachingAllocator::raw_delete(d_workspace);
  CHECK_SUBLAS_ERROR(sublasLtMatmulDescDestroy(operation_desc));
}

void fp8_scaled_mm(
    torch::Tensor& out,
    const torch::Tensor& mat_a,  // shape: [m, k], stride: (k, 1)
    const torch::Tensor& mat_b,  // shape: [n, k], stride: (k, 1) or [k, n], stride: (1, k)
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(
      (mat_a.size(1) * mat_a.element_size()) % 16 == 0, "mat_a must be multiple of 16 bytes for memory alignment");

  TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor");
  TORCH_CHECK(out.stride(1) == 1, "out must be a row major tensor");

  const int64_t m = mat_a.size(0);
  const int64_t k = mat_a.size(1);
  torch::Tensor b = (mat_b.stride(1) == 1) ? mat_b : mat_b.t();
  const int64_t n = b.size(0);

  TORCH_CHECK(out.size(0) == m && out.size(1) == n, "out has incorrect shape");

  TORCH_CHECK(b.stride(1) == 1, "mat_b must be a column major tensor");
  TORCH_CHECK(mat_a.size(1) == b.size(1), "mat_a and mat_b shapes cannot be multiplied");

  TORCH_CHECK((b.size(1) * b.element_size()) % 16 == 0, "mat_b must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK(mat_a.scalar_type() == torch::kFloat8_e4m3fn, "mat_a must be Float8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "mat_b must be Float8_e4m3fn");
  TORCH_CHECK(out.scalar_type() == torch::kHalf || out.scalar_type() == torch::kBFloat16, "out must be Half or BFloat16");

  TORCH_CHECK(scales_a.numel() == mat_a.size(0), "size of scales_a is not matched");
  TORCH_CHECK(scales_b.numel() == b.size(0), "size of scales_b is not matched");
  TORCH_CHECK(scales_a.is_contiguous(), "scales_a must be contiguous");
  TORCH_CHECK(scales_b.is_contiguous(), "scales_b msut be contiguous");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be Float32");

  void* bias_ptr = nullptr;
  float beta = 0.0f;
  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(0), "size of bias is not matched");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->dtype() == out.scalar_type(), "bias dtype must match output dtype");
    bias_ptr = bias->data_ptr();
    beta = 1.0f;
  }

  TORCH_CHECK((out.size(1) * out.element_size()) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");

  supaStream_t stream = c10::supa::getCurrentSUPAStream().stream();

  if (out.scalar_type() == torch::kBFloat16) {
    sublas_fp8_scaled_mm(
        mat_a.data_ptr(),
        b.data_ptr(),
        out.data_ptr(),
        scales_a.data_ptr(),
        scales_b.data_ptr(),
        bias_ptr,
        beta,
        m,
        n,
        k,
        1,
        SUPA_R_16BF,
        stream);
  } else {
    sublas_fp8_scaled_mm(
        mat_a.data_ptr(),
        b.data_ptr(),
        out.data_ptr(),
        scales_a.data_ptr(),
        scales_b.data_ptr(),
        bias_ptr,
        beta,
        m,
        n,
        k,
        1,
        SUPA_R_16F,
        stream);
  }
}

void bmm_fp8(
    at::Tensor A,
    at::Tensor B,
    at::Tensor D,
    at::Tensor A_scale,
    at::Tensor B_scale,
    at::Tensor workspace_buffer,
    int64_t sublas_handle) {
  TORCH_CHECK(A.dim() == 3, "Expected 3D tensor for A");  // A [b,m,k]
  TORCH_CHECK(B.dim() == 3, "Expected 3D tensor for B");  // B [b,k,n] stride[n*k, 1, k] -> B [b,n,k] stride[n*k,k,1]
  TORCH_CHECK(D.dim() == 3, "Expected 3D tensor for D");  // D [b,m,n]

  TORCH_CHECK(A.size(0) == B.size(0) && A.size(0) == D.size(0), "Batch sizes must match");
  TORCH_CHECK(A.stride(2) == 1, "A must be a row major tensor");
  TORCH_CHECK((A.size(2) * A.element_size()) % 16 == 0, "A must be multiple of 16 bytes for memory alignment");

  const int64_t batch = A.size(0);
  const int64_t m = A.size(1);
  const int64_t k = A.size(2);
  torch::Tensor b = (B.size(2) == A.size(2)) ? B : B.transpose(-2, -1);
  // B [b,n,k] stride[n*k,k,1]
  TORCH_CHECK(b.stride(2) == 1, "B must be a column major tensor");
  TORCH_CHECK(A.size(2) == b.size(2), "A and B shapes cannot be multiplied");
  const int64_t n = b.size(1);
  TORCH_CHECK(A.size(1) == D.size(1) && b.size(1) == D.size(2), "Result tensor has incorrect shape");

  torch::Dtype out_dtype = D.scalar_type();
  TORCH_CHECK((b.size(2) * b.element_size()) % 16 == 0, "B must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK(A.scalar_type() == torch::kFloat8_e4m3fn, "A must be Float8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "B must be Float8_e4m3fn");
  TORCH_CHECK(out_dtype == torch::kHalf || out_dtype == torch::kBFloat16, "out_dtype must be Half or BFloat16");

  TORCH_CHECK(A_scale.numel() == 1, "size of scales_a is not matched");
  TORCH_CHECK(B_scale.numel() == 1, "size of scales_b is not matched");
  TORCH_CHECK(A_scale.scalar_type() == torch::kFloat32, "scales_a must be Float32");
  TORCH_CHECK(B_scale.scalar_type() == torch::kFloat32, "scales_b must be Float32");
  TORCH_CHECK((D.size(2) * D.element_size()) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");
  void* bias_ptr = nullptr;
  float beta = 0.0f;

  supaStream_t stream = c10::supa::getCurrentSUPAStream().stream();
  if (out_dtype == torch::kBFloat16) {
    sublas_fp8_scaled_mm(
        A.data_ptr(),
        b.data_ptr(),
        D.data_ptr(),
        A_scale.data_ptr(),
        B_scale.data_ptr(),
        bias_ptr,
        beta,
        m,
        n,
        k,
        2,
        SUPA_R_16BF,
        stream,
        batch,
        workspace_buffer,
        sublas_handle);
  } else {
    sublas_fp8_scaled_mm(
        A.data_ptr(),
        b.data_ptr(),
        D.data_ptr(),
        A_scale.data_ptr(),
        B_scale.data_ptr(),
        bias_ptr,
        beta,
        m,
        n,
        k,
        2,
        SUPA_R_16F,
        stream,
        batch,
        workspace_buffer,
        sublas_handle);
  }
}

void mxfp8_scaled_mm(
    torch::Tensor& out,
    const torch::Tensor& mat_a,      // [m, k] FP8 E4M3
    const torch::Tensor& mat_b,     // [n, k] FP8 E4M3
    const torch::Tensor& scales_a,  // [align_up(m,16), k/32] uint8 UE8M0, interleaved
    const torch::Tensor& scales_b) {  // [n, k/32] uint8 UE8M0, interleaved
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be row major");
  TORCH_CHECK(out.stride(1) == 1, "out must be row major");

  const int64_t m = mat_a.size(0);
  const int64_t k = mat_a.size(1);
  const int64_t n = mat_b.size(0);
  const int64_t m_scale_rows = ((m + 15) / 16) * 16;

  TORCH_CHECK(mat_b.stride(1) == 1, "mat_b must be row major [n, k]");
  TORCH_CHECK(out.size(0) == m && out.size(1) == n, "out has incorrect shape");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(1), "mat_a and mat_b K dimension mismatch");
  TORCH_CHECK(mat_a.scalar_type() == torch::kFloat8_e4m3fn, "mat_a must be Float8_e4m3fn");
  TORCH_CHECK(mat_b.scalar_type() == torch::kFloat8_e4m3fn, "mat_b must be Float8_e4m3fn");
  TORCH_CHECK(out.scalar_type() == torch::kHalf || out.scalar_type() == torch::kBFloat16, "out must be Half or BFloat16");
  TORCH_CHECK(scales_a.scalar_type() == torch::kUInt8, "scales_a must be uint8 (UE8M0)");
  TORCH_CHECK(scales_b.scalar_type() == torch::kUInt8, "scales_b must be uint8 (UE8M0)");
  TORCH_CHECK(scales_a.size(0) == m_scale_rows && scales_a.size(1) == k / 32,
              "scales_a shape mismatch, expected [align_up(m, 16), k/32]");
  TORCH_CHECK(scales_b.size(0) == n && scales_b.size(1) == k / 32, "scales_b shape mismatch, expected [n, k/32]");
  TORCH_CHECK(scales_a.is_contiguous(), "scales_a must be contiguous");
  TORCH_CHECK(scales_b.is_contiguous(), "scales_b must be contiguous");
  TORCH_CHECK((out.size(1) * out.element_size()) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");

  supaStream_t stream = c10::supa::getCurrentSUPAStream().stream();

  if (out.scalar_type() == torch::kBFloat16) {
    sublas_fp8_scaled_mm(
        mat_a.data_ptr(), mat_b.data_ptr(), out.data_ptr(),
        scales_a.data_ptr(), scales_b.data_ptr(), nullptr, 0.0f,
        m, n, k, 3, SUPA_R_16BF, stream);
  } else {
    sublas_fp8_scaled_mm(
        mat_a.data_ptr(), mat_b.data_ptr(), out.data_ptr(),
        scales_a.data_ptr(), scales_b.data_ptr(), nullptr, 0.0f,
        m, n, k, 3, SUPA_R_16F, stream);
  }
}

torch::Tensor supa_fp8_per_tensor_scaled_mm(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(
      (mat_a.size(1) * mat_a.element_size()) % 16 == 0, "mat_a must be multiple of 16 bytes for memory alignment");

  TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor");
  TORCH_CHECK(out.stride(1) == 1, "out must be a row major tensor");

  const int64_t m = mat_a.size(0);
  const int64_t k = mat_a.size(1);
  torch::Tensor b = (mat_b.stride(1) == 1) ? mat_b : mat_b.t();
  const int64_t n = b.size(0);

  TORCH_CHECK(out.size(0) == m && out.size(1) == n, "out has incorrect shape");
  TORCH_CHECK(b.stride(1) == 1, "mat_b must be a column major tensor");
  TORCH_CHECK(mat_a.size(1) == b.size(1), "mat_a and mat_b shapes cannot be multiplied");
  TORCH_CHECK((b.size(1) * b.element_size()) % 16 == 0, "mat_b must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK(mat_a.scalar_type() == torch::kFloat8_e4m3fn, "mat_a must be Float8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "mat_b must be Float8_e4m3fn");
  TORCH_CHECK(out.scalar_type() == torch::kHalf || out.scalar_type() == torch::kBFloat16, "out must be Half or BFloat16");

  TORCH_CHECK(scales_a.numel() == 1, "scales_a must have exactly 1 element for per-tensor quant");
  TORCH_CHECK(scales_b.numel() == 1, "scales_b must have exactly 1 element for per-tensor quant");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be Float32");
  TORCH_CHECK((out.size(1) * out.element_size()) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");

  supaStream_t stream = c10::supa::getCurrentSUPAStream().stream();

  if (out.scalar_type() == torch::kBFloat16) {
    sublas_fp8_scaled_mm(
        mat_a.data_ptr(),
        b.data_ptr(),
        out.data_ptr(),
        scales_a.data_ptr(),
        scales_b.data_ptr(),
        nullptr,
        0.0f,
        m,
        n,
        k,
        2,
        SUPA_R_16BF,
        stream);
  } else {
    sublas_fp8_scaled_mm(
        mat_a.data_ptr(),
        b.data_ptr(),
        out.data_ptr(),
        scales_a.data_ptr(),
        scales_b.data_ptr(),
        nullptr,
        0.0f,
        m,
        n,
        k,
        2,
        SUPA_R_16F,
        stream);
  }

  if (bias) {
    TORCH_CHECK(bias->numel() == n, "size of bias is not matched");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->dtype() == out.scalar_type(), "bias dtype must match output dtype");
    out.add_(*bias);
  }

  return out;
}
