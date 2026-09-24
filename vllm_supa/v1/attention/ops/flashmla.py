# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from typing import Optional, Tuple

import flash_mla as _flashmla_infer
import torch

from vllm.utils.torch_utils import direct_register_custom_op


def sparse_decode_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    tile_scheduler_metadata: Optional[torch.Tensor],
    num_splits: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    head_dim_v: int,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert _flashmla_infer is not None, "flash_mla module not initialized."
    # vLLM supplies dense scheduler tensors through this custom-op schema,
    # while flash_mla builds sparse decode metadata internally.
    sched_meta = _flashmla_infer.FlashMLASchedMeta()
    out, lse = _flashmla_infer.flash_mla_with_kvcache(
        q,
        k_cache,
        None,  # block_table
        None,  # cache_seqlens
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=sched_meta,
        num_splits=None,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=indices_in_kvcache,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        extra_topk_length=extra_topk_length,
    )
    return out, lse, sched_meta.tile_scheduler_metadata, sched_meta.num_splits


def dense_decode_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    tile_scheduler_metadata: Optional[torch.Tensor],
    num_splits: Optional[torch.Tensor],
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert _flashmla_infer is not None, "flash_mla module not initialized."
    assert tile_scheduler_metadata is not None
    assert num_splits is not None
    output, lse = _flashmla_infer.flash_mla_with_kvcache(
        q,
        k_cache,
        head_dim_v=head_dim_v,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        is_fp8_kvcache=False,
    )
    return output, lse, tile_scheduler_metadata, num_splits


def sparse_prefill_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert _flashmla_infer is not None, "flash_mla module not initialized."
    output, max_logits, lse = _flashmla_infer.flash_mla_sparse_fwd(
        q,
        kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    # should be remove
    if out is not None:
        out.copy_(output)
    return output, max_logits, lse


def dense_prefill_fwd(
    workspace_buffer: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_seqlen_q: torch.Tensor,
    cumulative_seqlen_kv: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    mask_mode_code: int,
    softmax_scale: float,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    is_varlen: bool,
) -> None:
    raise NotImplementedError("_flashmla_C::dense_prefill_fwd is not supported by SUPA.")


def dense_prefill_bwd(
    workspace_buffer: torch.Tensor,
    d_o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    cumulative_seqlen_q: torch.Tensor,
    cumulative_seqlen_kv: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    mask_mode_code: int,
    softmax_scale: float,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    is_varlen: bool,
) -> None:
    raise NotImplementedError("_flashmla_C::dense_prefill_bwd is not supported by SUPA.")


_flashmla_lib = torch.library.Library("_flashmla_C", "DEF")
_flashmla_extension_lib = torch.library.Library("_flashmla_extension_C", "DEF")


def _register_flashmla_custom_ops() -> None:

    direct_register_custom_op(
        op_name="sparse_decode_fwd",
        op_func=sparse_decode_fwd,
        target_lib=_flashmla_lib,
    )

    direct_register_custom_op(
        op_name="dense_decode_fwd",
        op_func=dense_decode_fwd,
        target_lib=_flashmla_lib,
    )

    direct_register_custom_op(
        op_name="sparse_prefill_fwd",
        op_func=sparse_prefill_fwd,
        target_lib=_flashmla_lib,
    )

    direct_register_custom_op(
        op_name="dense_prefill_fwd",
        op_func=dense_prefill_fwd,
        target_lib=_flashmla_lib,
    )

    direct_register_custom_op(
        op_name="dense_prefill_bwd",
        op_func=dense_prefill_bwd,
        target_lib=_flashmla_lib,
    )


def fwd_kvcache_mla_fp8(
    q: torch.Tensor,
    kcache: torch.Tensor,
    head_size_v: int,
    seqlens_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    descale_q: Optional[torch.Tensor],
    descale_k: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("_flashmla_extension_C::fwd_kvcache_mla_fp8 is not supported by SUPA.")


def get_mla_decoding_metadata_dense_fp8(
    seqlens_k: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("_flashmla_extension_C::get_mla_decoding_metadata_dense_fp8 is not supported by SUPA.")


def _register_flashmla_extension_custom_ops() -> None:

    direct_register_custom_op(
        op_name="fwd_kvcache_mla_fp8",
        op_func=fwd_kvcache_mla_fp8,
        target_lib=_flashmla_extension_lib,
    )

    direct_register_custom_op(
        op_name="get_mla_decoding_metadata_dense_fp8",
        op_func=get_mla_decoding_metadata_dense_fp8,
        target_lib=_flashmla_extension_lib,
    )


_register_flashmla_custom_ops()
_register_flashmla_extension_custom_ops()
