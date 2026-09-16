#pragma once

// Common ldsm helper — include inside namespace MARLIN_NAMESPACE_NAME,
// inside the #else branch of __CUDA_ARCH__ < 750.
// Requires marlin_dtypes.cuh and core/scalar_type.hpp to be included first.
//
// Replaces: ldmatrix.sync.aligned.m8n8.x4.shared.b16 (count=4)
//           ldmatrix.sync.aligned.m8n8.x2.shared.b16 (count=2)
//           ldmatrix.sync.aligned.m8n8.x1.shared.b16 (count=1)
// PTX: ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];
// Loads a 16x16 matrix fragment of operand A from shared memory
// directly into tensor core register layout.

#ifdef USE_SUDA
// Software simulation using __shfl_sync: each thread loads its own 128-bit
// row from shared memory, then redistributes column pairs across the warp to
// match the ldmatrix register assignment.
template <int count, vllm::ScalarTypeId type_id>
__device__ inline void ldsm(typename MarlinScalarType<type_id>::FragA& frag_a,
                            const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);

  // Each thread loads its own row as 4 uint32 (8 x 16-bit elements)
  const uint32_t* row = reinterpret_cast<const uint32_t*>(smem_ptr);
  uint32_t my_col0 = row[0];  // cols 0-1
  uint32_t my_col1 = row[1];  // cols 2-3
  uint32_t my_col2 = row[2];  // cols 4-5
  uint32_t my_col3 = row[3];  // cols 6-7

  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;  // group: 0..7
  const int t = lane & 3;   // thread in group: 0..3

  // For each output register, gather all 4 column pairs from the source
  // thread (g + i*8), then select column pair t for this thread.
  if constexpr (count == 4) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int src = g + i * 8;
      uint32_t c0 = __shfl_sync(0xffffffff, my_col0, src);
      uint32_t c1 = __shfl_sync(0xffffffff, my_col1, src);
      uint32_t c2 = __shfl_sync(0xffffffff, my_col2, src);
      uint32_t c3 = __shfl_sync(0xffffffff, my_col3, src);
      a[i] = (t == 0) ? c0 : (t == 1) ? c1 : (t == 2) ? c2 : c3;
    }
  } else if constexpr (count == 2) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int src = g + i * 8;
      uint32_t c0 = __shfl_sync(0xffffffff, my_col0, src);
      uint32_t c1 = __shfl_sync(0xffffffff, my_col1, src);
      uint32_t c2 = __shfl_sync(0xffffffff, my_col2, src);
      uint32_t c3 = __shfl_sync(0xffffffff, my_col3, src);
      a[i] = (t == 0) ? c0 : (t == 1) ? c1 : (t == 2) ? c2 : c3;
    }
  } else if constexpr (count == 1) {
    int src = g;
    uint32_t c0 = __shfl_sync(0xffffffff, my_col0, src);
    uint32_t c1 = __shfl_sync(0xffffffff, my_col1, src);
    uint32_t c2 = __shfl_sync(0xffffffff, my_col2, src);
    uint32_t c3 = __shfl_sync(0xffffffff, my_col3, src);
    a[0] = (t == 0) ? c0 : (t == 1) ? c1 : (t == 2) ? c2 : c3;
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}
#else
template <int count, vllm::ScalarTypeId type_id>
__device__ inline void ldsm(typename MarlinScalarType<type_id>::FragA& frag_a,
                            const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem));
  } else if constexpr (count == 2) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(smem));
  } else if constexpr (count == 1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(a[0])
                 : "r"(smem));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}
#endif  // USE_SUDA
