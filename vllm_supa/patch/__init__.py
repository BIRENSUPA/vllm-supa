# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

_PATCHES_APPLIED = False


def apply_patches():
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    from vllm.logger import logger

    from .model_executor.kernels.linear import supa_fp8  # noqa: F401
    from .v1.sample import rejection_sampler  # noqa: F401
    from .v1.worker import gpu_worker  # noqa: F401
    from .v1.worker.gpu.sample import penalties  # noqa: F401

    _PATCHES_APPLIED = True
    logger.info("vllm_supa applied patches successfully.")
