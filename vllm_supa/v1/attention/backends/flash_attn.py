# Copyright (C) 2025-2026 Shanghai Biren Technology Co., Ltd.

from vllm_supa.v1.attention.ops import flash_attn  # noqa: F401

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

@register_backend(AttentionBackendEnum.FLASH_ATTN)
class SUPAFlashAttentionBackend(FlashAttentionBackend):
    pass
