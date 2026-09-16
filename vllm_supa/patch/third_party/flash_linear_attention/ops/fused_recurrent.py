# Copyright (c) 2025-2026 Shanghai Biren Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Route the GDN packed decode path through the native SUPA kernel."""

import torch

import vllm.third_party.flash_linear_attention.ops.fused_recurrent as fused_recurrent
import vllm_supa._supa_C  # noqa: F401
from vllm_supa.patch_to_with_log import patch_to_with_log


def _can_use_supa_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> bool:
    data_tensors = (mixed_qkv, a, b, initial_state, out)
    parameter_tensors = (A_log, dt_bias)
    tensors = (*data_tensors, *parameter_tensors, ssm_state_indices)
    return (
        use_qk_l2norm_in_kernel
        and mixed_qkv.device.type == "supa"
        and all(tensor.dtype == torch.bfloat16 for tensor in data_tensors)
        and all(
            tensor.dtype in (torch.float32, torch.bfloat16)
            for tensor in parameter_tensors
        )
        and ssm_state_indices.dtype in (torch.int32, torch.int64)
        and all(tensor.is_contiguous() for tensor in tensors)
        and all(
            tensor.data_ptr() % 16 == 0
            for tensor in (mixed_qkv, initial_state, out)
        )
    )


@patch_to_with_log(fused_recurrent)
def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _can_use_supa_packed_decode(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel,
    ):
        return fused_recurrent._orig_fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            scale,
            initial_state,
            out,
            ssm_state_indices,
            use_qk_l2norm_in_kernel,
        )

    torch.ops._supa_C.fused_recurrent_gated_delta_rule_packed_decode_supa(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        initial_state,
        ssm_state_indices,
        scale,
    )
    return out, initial_state
