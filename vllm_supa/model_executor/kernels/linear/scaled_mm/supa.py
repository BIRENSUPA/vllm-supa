# Copyright (C) 2025-2026 Shanghai Biren Technology Co., Ltd.

"""Dense FP8 block-scaled GEMM implemented by the SUPA extension."""

from typing import cast

import torch

from vllm.utils.math_utils import cdiv
from vllm.model_executor.kernels.linear import CutlassFp8BlockScaledMMKernel


class SupaFp8BlockScaledMMKernel(CutlassFp8BlockScaledMMKernel):
    """Use the SUPA FP8 block-scaled GEMM instead of the CUDA CUTLASS op."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        return supa_scaled_mm(A, B, As, Bs, out_dtype)


def supa_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    `supa_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b).T).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[0] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[0])
    a = a.view(-1, a.shape[-1])

    supa_compatible = (
        scale_a.ndim == a.ndim == 2 and scale_a.shape[0] == a.shape[0] and scale_a.shape[1] == cdiv(a.shape[1], 128)
    ) and (
        scale_b.ndim == b.ndim == 2
        and scale_b.shape[0] == cdiv(b.shape[0], 128)
        and scale_b.shape[1] == cdiv(b.shape[1], 128)
    )
    if not supa_compatible:
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
            triton_scaled_mm,
        )

        # vLLM incorrectly annotates dtype values as type[torch.dtype].
        out = triton_scaled_mm(
            a,
            b,
            scale_a,
            scale_b,
            cast(type[torch.dtype], out_dtype),
            bias,
        )
    else:
        out = torch.empty((a.shape[0], b.shape[0]), dtype=out_dtype, device=a.device)
        torch.ops._supa_C.supa_fp8_blockwise_scaled_mm(out, a, b.T, scale_a, scale_b.T)

    return out.view(*target_shape)
