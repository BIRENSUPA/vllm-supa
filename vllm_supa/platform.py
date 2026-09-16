# Copyright (C) 2025-2026 Shanghai Biren Technology Co., Ltd.

from typing import TYPE_CHECKING, Optional
from functools import cache

import torch
from typing_extensions import override
from vllm.logger import init_logger

import vllm.platforms.cuda as cuda_platform
from vllm.platforms.cuda import (
    CudaPlatformBase,
    NvmlCudaPlatform,
    NonNvmlCudaPlatform,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_supa.patch_to_with_log import patch_to_with_log

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


@patch_to_with_log(cuda_platform)
def _get_backend_priorities(
    use_mla: bool,
    device_capability: DeviceCapability,
    num_heads: int | None = None,
    kv_cache_dtype: "CacheDType | None" = None,
    use_non_causal: bool = False,
) -> list[AttentionBackendEnum]:
    @cache
    def _get_backend_priorities_inner(use_mla):
        """Get backend priorities with lazy import to avoid circular dependency."""
        if use_mla:
            return [
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHMLA_SPARSE,
                AttentionBackendEnum.FLASHINFER_MLA,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]

    return _get_backend_priorities_inner(use_mla)


def register_attention_backends() -> None:
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        class_path="vllm_supa.v1.attention.backends.flash_attn.SUPAFlashAttentionBackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA,
        class_path="vllm_supa.v1.attention.backends.mla.flashmla.SUPAFlashMLABackend",
    )


class SUPAPlatformBase(CudaPlatformBase):
    dist_backend: str = "bccl"
    device_control_env_var: str = "SUPA_VISIBLE_DEVICES"

    @classmethod
    @override
    def import_kernels(cls) -> None:
        """Import any platform-specific C kernels."""
        from vllm_supa.utils import import_supa_kernels

        import_supa_kernels()

    @classmethod
    @override
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        super().check_and_update_config(vllm_config)
        # FIXME: need upstream suAttention、FlashMLA、FlashAttention align to open source
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        backend = vllm_config.attention_config.backend
        use_flash_atten = (backend == AttentionBackendEnum.FLASH_ATTN) or (
            backend is None
        )  # If backend is not set, we may choose Flash Attention
        if use_flash_atten:
            # FIXME: limit by suAttention, which requires block size to be 128 for now.
            cache_config.block_size = 128
            logger.info(
                "Forcing kv cache block size to %s for Flash Attention backend on suAttention.",
                cache_config.block_size,
            )

        if model_config is not None and model_config.use_mla and cache_config.block_size is not None:
            backend = vllm_config.attention_config.backend
            use_flashmla = (backend == AttentionBackendEnum.FLASHMLA) or (
                backend is None
            )  # If backend is not set, we may choose FlashMLA
            from vllm.v1.attention.ops.flashmla import is_flashmla_dense_supported

            # FIXME: just temp patch, should remove suattention and flashmla unified backend in the future
            is_dsv4 = "DeepseekV4ForCausalLM" in model_config.architectures
            if use_flashmla and is_dsv4:
                cache_config.block_size = 256
                logger.info("Forcing kv cache block size to 256 for DeepSeek V4 FlashMLA sparse SWA.")
            elif use_flashmla and is_flashmla_dense_supported()[0] and cache_config.block_size % 64 != 0:
                cache_config.block_size = 64
                logger.info("Forcing kv cache block size to 64 for FlashMLA backend.")
            if model_config.get_num_attention_heads(vllm_config.parallel_config) != 128:
                vllm_config.attention_config.backend = AttentionBackendEnum.FLASHMLA
                logger.warning(
                    "Forcing attention backend to FlashMLA since the number of "
                    "attention heads is not 128. not supported by suAttention MLA."
                )

    @classmethod
    @override
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum | None,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        register_attention_backends()
        return super().get_attn_backend_cls(selected_backend, attn_selector_config, num_heads)

    # TODO: need to change
    @classmethod
    @override
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: Optional["AttentionBackendEnum"] = None,
    ) -> "AttentionBackendEnum":
        return AttentionBackendEnum.FLASH_ATTN

    # TODO: implement Punica wrapper for SUPA platform
    @classmethod
    @override
    def get_punica_wrapper(cls) -> str:
        raise NotImplementedError("Punica wrapper is not implemented for SUPAPlatform")

    @override
    def is_sleep_mode_available(self) -> bool:
        return True

    @classmethod
    @override
    def support_deep_gemm(cls) -> bool:
        import vllm_supa.envs as envs

        return envs.VLLM_SUPA_ENABLE_DEEP_GEMM

    @classmethod
    @override
    def is_arch_support_pdl(cls) -> bool:
        # SUPA does not support the gdc_wait instruction required by PDL.
        return False

    @classmethod
    @override
    def pre_register_and_update(cls, parser: Optional[FlexibleArgumentParser] = None) -> None:
        if parser is not None:
            for action in parser._actions:
                opts = action.option_strings
                if opts:
                    if opts[0] == "--block-size":
                        action.choices = [128]
                    elif opts[0] == "--device":
                        action.choices = ["auto", "supa"]
                    elif opts[0] == "--moe-backend" and action.choices is not None and "supa" not in action.choices:
                        action.choices = list(action.choices) + ["supa"]


brml_available = False


class BrmlSupaPlatform(SUPAPlatformBase, NvmlCudaPlatform):
    pass


class NonBrmlSupaPlatform(SUPAPlatformBase, NonNvmlCudaPlatform):
    pass


SUPAPlatform = BrmlSupaPlatform if brml_available else NonBrmlSupaPlatform
