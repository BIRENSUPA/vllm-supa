# Copyright (C) 2025-2026 Shanghai Biren Technology Co., Ltd.

"""Register the SUPA implementation for dense FP8 block-scaled linear layers."""

from vllm.logger import init_logger
from vllm.model_executor.kernels import linear as linear_kernels
from vllm.platforms import PlatformEnum

from vllm_supa.model_executor.kernels.linear.scaled_mm.supa import (
    SupaFp8BlockScaledMMKernel,
)

logger = init_logger(__name__)

_kernels = linear_kernels._POSSIBLE_FP8_BLOCK_KERNELS.setdefault(
    PlatformEnum.CUDA, []
)

if SupaFp8BlockScaledMMKernel not in _kernels:
    _kernels.insert(0, SupaFp8BlockScaledMMKernel)
