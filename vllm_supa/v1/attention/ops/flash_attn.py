# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import sys
import types
from typing import Optional

import suattention as _flash_atten_infer
import torch

from vllm.utils.torch_utils import direct_register_custom_op


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def maybe_fp8_descale(descale, tensor):
    """Return descale if tensor is fp8, otherwise None."""
    if tensor is None or tensor.dtype not in (torch.float8_e4m3fn, ):
        return None
    return maybe_contiguous(descale)


def varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    p_dropout: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    return_softmax: bool,
    num_splits: int,
    gen: Optional[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("FA2 varlen_fwd is not supported by SUPA.")


def fwd_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    out: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    p_dropout: float,
    softmax_scale: float,
    is_causal: bool,
    softcap: float,
    return_softmax: bool,
    gen: Optional[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("FA2 fwd_sparse is not supported by SUPA.")


def varlen_fwd_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    out: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    p_dropout: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    softcap: float,
    return_softmax: bool,
    gen: Optional[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("FA2 varlen_fwd_sparse is not supported by SUPA.")


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor],
    v_new: Optional[torch.Tensor],
    q_v: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cu_seqlens_k_new: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    page_table: Optional[torch.Tensor],
    kv_batch_idx: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    seqlens_rotary: Optional[torch.Tensor],
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    is_rotary_interleaved: bool,
    scheduler_metadata: Optional[torch.Tensor],
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
    s_aux: Optional[torch.Tensor],
    cp_world_size: int,
    cp_rank: int,
    cp_tot_seqused_k: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert _flash_atten_infer is not None, "suattention module not initialized."

    window_size = (window_size_left, window_size_right)

    if page_table is not None:
        q_descale = maybe_fp8_descale(q_descale, q)
        k_descale = maybe_fp8_descale(k_descale, k)
        v_descale = maybe_fp8_descale(v_descale, v)
        output, softmax_lse = _flash_atten_infer.flash_attn_with_kvcache(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_q=max_seqlen_q,
            cache_seqlens=seqused_k,
            softmax_scale=softmax_scale,
            causal=(is_causal and max_seqlen_q != 1),
            window_size=window_size,
            page_table=page_table,
            return_softmax_lse=True,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            # flashattn_train does not accept out here. When restoring that
            # backend, omit this argument and copy the returned output to out.
            out=out,
        )
    else:
        # SUPA kernels require cu_seqlens tensors with storage_offset=0.
        if cu_seqlens_q is not None and cu_seqlens_q.storage_offset() != 0:
            cu_seqlens_q = cu_seqlens_q.clone()
        if cu_seqlens_k is not None and cu_seqlens_k.storage_offset() != 0:
            cu_seqlens_k = cu_seqlens_k.clone()
        q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)
        output, softmax_lse = _flash_atten_infer.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            window_size=window_size,
            causal=is_causal and max_seqlen_q != 1,
            return_attn_probs=True,
        )

    return output, softmax_lse, torch.empty(0, device=output.device), torch.empty(0, device=output.device)


def get_scheduler_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_heads_k: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    seqused_k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cu_seqlens_k_new: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    page_size: Optional[int],
    max_seqlen_k_new: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    has_softcap: bool,
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
) -> torch.Tensor:

    assert _flash_atten_infer is not None, "suattention module not initialized."

    fn = getattr(_flash_atten_infer, "get_scheduler_metadata", None)
    if fn is None:
        return torch.empty(0, device=seqused_k.device)

    return fn(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        headdim=headdim,
        headdim_v=headdim_v,
        qkv_dtype=qkv_dtype,
        seqused_k=seqused_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_k_new=cu_seqlens_k_new,
        seqused_q=seqused_q,
        leftpad_k=leftpad_k,
        page_size=page_size,
        max_seqlen_k_new=max_seqlen_k_new,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        has_softcap=has_softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )

_fa2_lib = torch.library.Library("_vllm_fa2_C", "DEF")
_fa3_lib = torch.library.Library("_vllm_fa3_C", "DEF")

def _register_fa2_custom_ops() -> None:

    sys.modules.setdefault(
        "vllm.vllm_flash_attn._vllm_fa2_C",
        types.ModuleType("vllm.vllm_flash_attn._vllm_fa2_C"),
    )

    direct_register_custom_op(
        op_name="varlen_fwd",
        op_func=varlen_fwd,
        target_lib=_fa2_lib,
        mutates_args=["out"],
    )

    direct_register_custom_op(
        op_name="fwd_sparse",
        op_func=fwd_sparse,
        target_lib=_fa2_lib,
        mutates_args=["out"],
    )

    direct_register_custom_op(
        op_name="varlen_fwd_sparse",
        op_func=varlen_fwd_sparse,
        target_lib=_fa2_lib,
        mutates_args=["out"],
    )


def _register_fa3_custom_ops() -> None:

    direct_register_custom_op(
        op_name="fwd",
        op_func=fwd,
        target_lib=_fa3_lib,
        mutates_args=["out"],
    )

    direct_register_custom_op(
        op_name="get_scheduler_metadata",
        op_func=get_scheduler_metadata,
        target_lib=_fa3_lib,
    )


_register_fa2_custom_ops()
_register_fa3_custom_ops()
