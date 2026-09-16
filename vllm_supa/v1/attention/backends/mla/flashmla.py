# Copyright (C) 2025-2026 Shanghai Biren Technology Co., Ltd.

from vllm_supa.v1.attention.ops import flashmla as _flashmla  # noqa: F401
from vllm.v1.attention.backends.mla import flashmla
from vllm.v1.attention.backends.mla.flashmla import FlashMLABackend

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

_FLASHMLA_REORDER_BATCH_THRESHOLD = 1

flashmla.FlashMLAMetadataBuilder.reorder_batch_threshold = (
    _FLASHMLA_REORDER_BATCH_THRESHOLD
)

@register_backend(AttentionBackendEnum.FLASHMLA)
class SUPAFlashMLABackend(FlashMLABackend):
    pass
