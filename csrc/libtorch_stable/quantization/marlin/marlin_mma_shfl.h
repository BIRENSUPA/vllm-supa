#pragma once

// Software simulation of mma.sync.aligned instructions using __shfl_sync.
// Replaces NVIDIA tensor core PTX with warp-shuffle-based matrix multiply
// for Biren SUPA GPUs that lack hardware tensor cores.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {
namespace mma_shfl {

// ============================================================================
// Replaces: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// PTX: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//      {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
// Fragment layout (per thread, lane = threadIdx.x % 32, g = lane/4, t = lane%4):
//   A[0..3]: 4 uint32, each packing 2 fp16 values from rows g and g+8
//   B[0..1]: 2 uint32, each packing 2 fp16 values from col g
//   C[0..3]: 4 float, output at D[g][t*2], D[g][t*2+1], D[g+8][t*2], D[g+8][t*2+1]
// ============================================================================
__device__ inline void m16n8k16_f32_f16(const uint32_t* a, const uint32_t* b,
                                        float* c) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

#pragma unroll
  for (int kt = 0; kt < 4; kt++) {
    // Gather A values for rows g and g+8 from thread g*4+kt
    int a_src = g * 4 + kt;
    uint32_t a0 = __shfl_sync(0xffffffff, a[0], a_src);
    uint32_t a1 = __shfl_sync(0xffffffff, a[1], a_src);
    uint32_t a2 = __shfl_sync(0xffffffff, a[2], a_src);
    uint32_t a3 = __shfl_sync(0xffffffff, a[3], a_src);

    // Gather B values for col t*2 from thread (t*2)*4+kt
    int b_src_c0 = (t * 2) * 4 + kt;
    uint32_t b0_c0 = __shfl_sync(0xffffffff, b[0], b_src_c0);
    uint32_t b1_c0 = __shfl_sync(0xffffffff, b[1], b_src_c0);

    // Gather B values for col t*2+1 from thread (t*2+1)*4+kt
    int b_src_c1 = (t * 2 + 1) * 4 + kt;
    uint32_t b0_c1 = __shfl_sync(0xffffffff, b[0], b_src_c1);
    uint32_t b1_c1 = __shfl_sync(0xffffffff, b[1], b_src_c1);

    // Unpack half2 values
    half2 ha0 = *reinterpret_cast<half2*>(&a0);
    half2 ha1 = *reinterpret_cast<half2*>(&a1);
    half2 ha2 = *reinterpret_cast<half2*>(&a2);
    half2 ha3 = *reinterpret_cast<half2*>(&a3);

    half2 hb0_c0 = *reinterpret_cast<half2*>(&b0_c0);
    half2 hb1_c0 = *reinterpret_cast<half2*>(&b1_c0);
    half2 hb0_c1 = *reinterpret_cast<half2*>(&b0_c1);
    half2 hb1_c1 = *reinterpret_cast<half2*>(&b1_c1);

    // Fragment layout: a[0]=row_g k_lo, a[1]=row_g+8 k_lo,
    //                  a[2]=row_g k_hi, a[3]=row_g+8 k_hi
    // b[0]=k_lo, b[1]=k_hi

    // c[0] = D[g][t*2]: use a[0] (row g, k_lo) + a[2] (row g, k_hi)
    c[0] += __half2float(ha0.x) * __half2float(hb0_c0.x) +
            __half2float(ha0.y) * __half2float(hb0_c0.y) +
            __half2float(ha2.x) * __half2float(hb1_c0.x) +
            __half2float(ha2.y) * __half2float(hb1_c0.y);

    // c[1] = D[g][t*2+1]
    c[1] += __half2float(ha0.x) * __half2float(hb0_c1.x) +
            __half2float(ha0.y) * __half2float(hb0_c1.y) +
            __half2float(ha2.x) * __half2float(hb1_c1.x) +
            __half2float(ha2.y) * __half2float(hb1_c1.y);

    // c[2] = D[g+8][t*2]: use a[1] (row g+8, k_lo) + a[3] (row g+8, k_hi)
    c[2] += __half2float(ha1.x) * __half2float(hb0_c0.x) +
            __half2float(ha1.y) * __half2float(hb0_c0.y) +
            __half2float(ha3.x) * __half2float(hb1_c0.x) +
            __half2float(ha3.y) * __half2float(hb1_c0.y);

    // c[3] = D[g+8][t*2+1]
    c[3] += __half2float(ha1.x) * __half2float(hb0_c1.x) +
            __half2float(ha1.y) * __half2float(hb0_c1.y) +
            __half2float(ha3.x) * __half2float(hb1_c1.x) +
            __half2float(ha3.y) * __half2float(hb1_c1.y);
  }
}

// ============================================================================
// Replaces: mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
// PTX: mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
//      {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};
// Output: 2 uint32 (packed half2), c[0] = half2(D[g][t*2], D[g][t*2+1]),
//                                  c[1] = half2(D[g+8][t*2], D[g+8][t*2+1])
// ============================================================================
__device__ inline void m16n8k16_f16_f16(const uint32_t* a, const uint32_t* b,
                                        uint32_t* c) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  // Compute in float, then convert to half
  half2 hc0 = *reinterpret_cast<half2*>(&c[0]);
  half2 hc1 = *reinterpret_cast<half2*>(&c[1]);
  float acc0 = __half2float(hc0.x);  // D[g][t*2]
  float acc1 = __half2float(hc0.y);  // D[g][t*2+1]
  float acc2 = __half2float(hc1.x);  // D[g+8][t*2]
  float acc3 = __half2float(hc1.y);  // D[g+8][t*2+1]

#pragma unroll
  for (int kt = 0; kt < 4; kt++) {
    int a_src = g * 4 + kt;
    uint32_t a0 = __shfl_sync(0xffffffff, a[0], a_src);
    uint32_t a1 = __shfl_sync(0xffffffff, a[1], a_src);
    uint32_t a2 = __shfl_sync(0xffffffff, a[2], a_src);
    uint32_t a3 = __shfl_sync(0xffffffff, a[3], a_src);

    int b_src_c0 = (t * 2) * 4 + kt;
    uint32_t b0_c0 = __shfl_sync(0xffffffff, b[0], b_src_c0);
    uint32_t b1_c0 = __shfl_sync(0xffffffff, b[1], b_src_c0);

    int b_src_c1 = (t * 2 + 1) * 4 + kt;
    uint32_t b0_c1 = __shfl_sync(0xffffffff, b[0], b_src_c1);
    uint32_t b1_c1 = __shfl_sync(0xffffffff, b[1], b_src_c1);

    half2 ha0 = *reinterpret_cast<half2*>(&a0);
    half2 ha1 = *reinterpret_cast<half2*>(&a1);
    half2 ha2 = *reinterpret_cast<half2*>(&a2);
    half2 ha3 = *reinterpret_cast<half2*>(&a3);

    half2 hb0_c0 = *reinterpret_cast<half2*>(&b0_c0);
    half2 hb1_c0 = *reinterpret_cast<half2*>(&b1_c0);
    half2 hb0_c1 = *reinterpret_cast<half2*>(&b0_c1);
    half2 hb1_c1 = *reinterpret_cast<half2*>(&b1_c1);

    acc0 += __half2float(ha0.x) * __half2float(hb0_c0.x) +
            __half2float(ha0.y) * __half2float(hb0_c0.y) +
            __half2float(ha2.x) * __half2float(hb1_c0.x) +
            __half2float(ha2.y) * __half2float(hb1_c0.y);

    acc1 += __half2float(ha0.x) * __half2float(hb0_c1.x) +
            __half2float(ha0.y) * __half2float(hb0_c1.y) +
            __half2float(ha2.x) * __half2float(hb1_c1.x) +
            __half2float(ha2.y) * __half2float(hb1_c1.y);

    acc2 += __half2float(ha1.x) * __half2float(hb0_c0.x) +
            __half2float(ha1.y) * __half2float(hb0_c0.y) +
            __half2float(ha3.x) * __half2float(hb1_c0.x) +
            __half2float(ha3.y) * __half2float(hb1_c0.y);

    acc3 += __half2float(ha1.x) * __half2float(hb0_c1.x) +
            __half2float(ha1.y) * __half2float(hb0_c1.y) +
            __half2float(ha3.x) * __half2float(hb1_c1.x) +
            __half2float(ha3.y) * __half2float(hb1_c1.y);
  }

  // Pack results back as half2
  half2 r0 = __halves2half2(__float2half(acc0), __float2half(acc1));
  half2 r1 = __halves2half2(__float2half(acc2), __float2half(acc3));
  c[0] = *reinterpret_cast<uint32_t*>(&r0);
  c[1] = *reinterpret_cast<uint32_t*>(&r1);
}

// ============================================================================
// Replaces: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// PTX: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//      {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
// ============================================================================
__device__ inline void m16n8k16_f32_bf16(const uint32_t* a, const uint32_t* b,
                                         float* c) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

#pragma unroll
  for (int kt = 0; kt < 4; kt++) {
    int a_src = g * 4 + kt;
    uint32_t a0 = __shfl_sync(0xffffffff, a[0], a_src);
    uint32_t a1 = __shfl_sync(0xffffffff, a[1], a_src);
    uint32_t a2 = __shfl_sync(0xffffffff, a[2], a_src);
    uint32_t a3 = __shfl_sync(0xffffffff, a[3], a_src);

    int b_src_c0 = (t * 2) * 4 + kt;
    uint32_t b0_c0 = __shfl_sync(0xffffffff, b[0], b_src_c0);
    uint32_t b1_c0 = __shfl_sync(0xffffffff, b[1], b_src_c0);

    int b_src_c1 = (t * 2 + 1) * 4 + kt;
    uint32_t b0_c1 = __shfl_sync(0xffffffff, b[0], b_src_c1);
    uint32_t b1_c1 = __shfl_sync(0xffffffff, b[1], b_src_c1);

    nv_bfloat162 ba0 = *reinterpret_cast<nv_bfloat162*>(&a0);
    nv_bfloat162 ba1 = *reinterpret_cast<nv_bfloat162*>(&a1);
    nv_bfloat162 ba2 = *reinterpret_cast<nv_bfloat162*>(&a2);
    nv_bfloat162 ba3 = *reinterpret_cast<nv_bfloat162*>(&a3);

    nv_bfloat162 bb0_c0 = *reinterpret_cast<nv_bfloat162*>(&b0_c0);
    nv_bfloat162 bb1_c0 = *reinterpret_cast<nv_bfloat162*>(&b1_c0);
    nv_bfloat162 bb0_c1 = *reinterpret_cast<nv_bfloat162*>(&b0_c1);
    nv_bfloat162 bb1_c1 = *reinterpret_cast<nv_bfloat162*>(&b1_c1);

    c[0] += __bfloat162float(ba0.x) * __bfloat162float(bb0_c0.x) +
            __bfloat162float(ba0.y) * __bfloat162float(bb0_c0.y) +
            __bfloat162float(ba2.x) * __bfloat162float(bb1_c0.x) +
            __bfloat162float(ba2.y) * __bfloat162float(bb1_c0.y);

    c[1] += __bfloat162float(ba0.x) * __bfloat162float(bb0_c1.x) +
            __bfloat162float(ba0.y) * __bfloat162float(bb0_c1.y) +
            __bfloat162float(ba2.x) * __bfloat162float(bb1_c1.x) +
            __bfloat162float(ba2.y) * __bfloat162float(bb1_c1.y);

    c[2] += __bfloat162float(ba1.x) * __bfloat162float(bb0_c0.x) +
            __bfloat162float(ba1.y) * __bfloat162float(bb0_c0.y) +
            __bfloat162float(ba3.x) * __bfloat162float(bb1_c0.x) +
            __bfloat162float(ba3.y) * __bfloat162float(bb1_c0.y);

    c[3] += __bfloat162float(ba1.x) * __bfloat162float(bb0_c1.x) +
            __bfloat162float(ba1.y) * __bfloat162float(bb0_c1.y) +
            __bfloat162float(ba3.x) * __bfloat162float(bb1_c1.x) +
            __bfloat162float(ba3.y) * __bfloat162float(bb1_c1.y);
  }
}

// ============================================================================
// Helper: unpack 4 fp8 e4m3 values from uint32 to float array
// ============================================================================
__device__ inline void unpack_fp8x4_to_float(uint32_t packed, float out[4]) {
  __nv_fp8x4_e4m3 fp8_val = *reinterpret_cast<__nv_fp8x4_e4m3*>(&packed);
  // Extract individual bytes and convert
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);
  __nv_fp8_e4m3 v0, v1, v2, v3;
  memcpy(&v0, &bytes[0], 1);
  memcpy(&v1, &bytes[1], 1);
  memcpy(&v2, &bytes[2], 1);
  memcpy(&v3, &bytes[3], 1);
  out[0] = float(v0);
  out[1] = float(v1);
  out[2] = float(v2);
  out[3] = float(v3);
}

// ============================================================================
// Replaces: mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32
// PTX: mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32
//      {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};
// (one k=16 slice of the full m16n8k32 operation)
// Fragment layout for fp8 (4 values per uint32):
//   A uses a[idx*2], a[idx*2+1] (2 regs = 8 fp8 values per thread)
//   B uses b[idx] (1 reg = 4 fp8 values per thread)
//   Thread covers k = 4*tid..4*tid+3 (4 k-values per thread, 4 threads = 16)
// ============================================================================
__device__ inline void m16n8k16_f32_e4m3(const uint32_t* a, const uint32_t* b,
                                         float* c, int idx) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

#pragma unroll
  for (int kt = 0; kt < 4; kt++) {
    // A for row g: from thread g*4+kt, covers k=4*kt..4*kt+3
    int a_src = g * 4 + kt;
    uint32_t a_row_g = __shfl_sync(0xffffffff, a[idx * 2], a_src);
    uint32_t a_row_g8 = __shfl_sync(0xffffffff, a[idx * 2 + 1], a_src);

    // B for col t*2: from thread (t*2)*4+kt
    int b_src_c0 = (t * 2) * 4 + kt;
    uint32_t b_col0 = __shfl_sync(0xffffffff, b[idx], b_src_c0);

    // B for col t*2+1: from thread (t*2+1)*4+kt
    int b_src_c1 = (t * 2 + 1) * 4 + kt;
    uint32_t b_col1 = __shfl_sync(0xffffffff, b[idx], b_src_c1);

    float fa_g[4], fa_g8[4], fb_c0[4], fb_c1[4];
    unpack_fp8x4_to_float(a_row_g, fa_g);
    unpack_fp8x4_to_float(a_row_g8, fa_g8);
    unpack_fp8x4_to_float(b_col0, fb_c0);
    unpack_fp8x4_to_float(b_col1, fb_c1);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      c[0] += fa_g[i] * fb_c0[i];
      c[1] += fa_g[i] * fb_c1[i];
      c[2] += fa_g8[i] * fb_c0[i];
      c[3] += fa_g8[i] * fb_c1[i];
    }
  }
}

// ============================================================================
// Helper: unpack 4 int8 values from uint32 (sign-extended to int32)
// ============================================================================
__device__ inline void unpack_int8x4(uint32_t packed, int32_t out[4]) {
  const int8_t* bytes = reinterpret_cast<const int8_t*>(&packed);
  out[0] = (int32_t)bytes[0];
  out[1] = (int32_t)bytes[1];
  out[2] = (int32_t)bytes[2];
  out[3] = (int32_t)bytes[3];
}

// ============================================================================
// Replaces: mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite
// PTX: mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite
//      {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};
// (one k=16 slice of the full m16n8k32 operation)
// Same fragment layout as fp8 (4 values per uint32)
// ============================================================================
__device__ inline void m16n8k16_s32_s8(const uint32_t* a, const uint32_t* b,
                                       int32_t* c, int idx) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

#pragma unroll
  for (int kt = 0; kt < 4; kt++) {
    int a_src = g * 4 + kt;
    uint32_t a_row_g = __shfl_sync(0xffffffff, a[idx * 2], a_src);
    uint32_t a_row_g8 = __shfl_sync(0xffffffff, a[idx * 2 + 1], a_src);

    int b_src_c0 = (t * 2) * 4 + kt;
    uint32_t b_col0 = __shfl_sync(0xffffffff, b[idx], b_src_c0);

    int b_src_c1 = (t * 2 + 1) * 4 + kt;
    uint32_t b_col1 = __shfl_sync(0xffffffff, b[idx], b_src_c1);

    int32_t ia_g[4], ia_g8[4], ib_c0[4], ib_c1[4];
    unpack_int8x4(a_row_g, ia_g);
    unpack_int8x4(a_row_g8, ia_g8);
    unpack_int8x4(b_col0, ib_c0);
    unpack_int8x4(b_col1, ib_c1);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      c[0] += ia_g[i] * ib_c0[i];
      c[1] += ia_g[i] * ib_c1[i];
      c[2] += ia_g8[i] * ib_c0[i];
      c[3] += ia_g8[i] * ib_c1[i];
    }
  }
}

// ============================================================================
// Replaces: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// PTX: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//      {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
// Decomposed into two k=16 calls (idx=0 and idx=1)
// ============================================================================
__device__ inline void m16n8k32_f32_e4m3(const uint32_t* a, const uint32_t* b,
                                         float* c) {
  m16n8k16_f32_e4m3(a, b, c, 0);
  m16n8k16_f32_e4m3(a, b, c, 1);
}

// ============================================================================
// Replaces: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite
// PTX: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite
//      {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
// Decomposed into two k=16 calls
// ============================================================================
__device__ inline void m16n8k32_s32_s8(const uint32_t* a, const uint32_t* b,
                                       int32_t* c) {
  m16n8k16_s32_s8(a, b, c, 0);
  m16n8k16_s32_s8(a, b, c, 1);
}

// ============================================================================
// Replaces: mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite (SM75)
// PTX: mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite
//      {%0,%1}, {%2}, {%3}, {%4,%5};
// Smaller operation: A=1 reg (4 int8), B=1 reg (4 int8), C=2 int32
// Layout for m8n8k16:
//   groupID = lane/4 (0..7), tid = lane%4 (0..3)
//   A: a[0] packs 4 int8: A[groupID][4*tid..4*tid+3] (row=groupID, k=4*tid..+3)
//   B: b[0] packs 4 int8: B[4*tid..4*tid+3][groupID] (k=4*tid..+3, col=groupID)
//   C: c[0] = D[groupID][tid*2], c[1] = D[groupID][tid*2+1]
// Note: This is 8x8 output (8 rows x 8 cols), but only top-half rows 0..7
// ============================================================================
__device__ inline void m8n8k16_s32_s8(uint32_t a_reg, uint32_t b_reg,
                                      int32_t* c) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

#pragma unroll
  for (int kt = 0; kt < 4; kt++) {
    int a_src = g * 4 + kt;
    uint32_t a_val = __shfl_sync(0xffffffff, a_reg, a_src);

    int b_src_c0 = (t * 2) * 4 + kt;
    uint32_t b_val_c0 = __shfl_sync(0xffffffff, b_reg, b_src_c0);

    int b_src_c1 = (t * 2 + 1) * 4 + kt;
    uint32_t b_val_c1 = __shfl_sync(0xffffffff, b_reg, b_src_c1);

    int32_t ia[4], ib_c0[4], ib_c1[4];
    unpack_int8x4(a_val, ia);
    unpack_int8x4(b_val_c0, ib_c0);
    unpack_int8x4(b_val_c1, ib_c1);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      c[0] += ia[i] * ib_c0[i];
      c[1] += ia[i] * ib_c1[i];
    }
  }
}

}  // namespace mma_shfl
}  // namespace MARLIN_NAMESPACE_NAME
