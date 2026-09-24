# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from typing import Literal

def register() -> Literal['vllm_supa.platform.SUPAPlatform']:
    """Register the SUPA platform."""
    from vllm_supa.utils import init_vllm_supa
    init_vllm_supa()
    return "vllm_supa.platform.SUPAPlatform"


def register_patch() -> None:
    from vllm_supa.patch import apply_patches

    apply_patches()


def register_model() -> None:
    pass
