# Copyright (C) 2025-2026 Shanghai Biren Technology Co., Ltd.

import torch
import torch.nn.functional as F

import vllm.model_executor.models.deepseek_v2 as _deepseek_v2

from vllm_supa.patch_to_with_log import patch_to_with_log

@patch_to_with_log(_deepseek_v2)
def _min_latency_fused_qkv_a_proj_impl(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Provide a SUPA-compatible fallback for the unavailable DSV3 GEMM op."""
    return F.linear(input_, weight)
