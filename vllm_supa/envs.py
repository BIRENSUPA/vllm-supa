# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    VLLM_SUPA_SKIP_PROFILE_RUN: bool = False
    VLLM_SUPA_SKIP_WARMUP_RUN: bool = False
    VLLM_SUPA_ENABLE_DEEP_GEMM: bool = True
    VLLM_SUPA_USE_LEGACY_ATTENTION_PATCH: bool = False

environment_variables: dict[str, Callable[[], Any]] = {
    # Skip vLLM worker profile_run when kv_cache_memory_bytes is provided.
    "VLLM_SUPA_SKIP_PROFILE_RUN": lambda: bool(
        int(os.getenv("VLLM_SUPA_SKIP_PROFILE_RUN", "0"))),
    # Skip vLLM worker compile/warmup dummy runs.
    "VLLM_SUPA_SKIP_WARMUP_RUN": lambda: bool(
        int(os.getenv("VLLM_SUPA_SKIP_WARMUP_RUN", "0"))),
    # Enable SUPA DeepGEMM selection by default; set to 0 to disable it.
    "VLLM_SUPA_ENABLE_DEEP_GEMM": lambda: bool(
        int(os.getenv("VLLM_SUPA_ENABLE_DEEP_GEMM", "1"))),
    # Deprecated: remove with the legacy attention patch implementation.
    # Use the original attention patch package under vllm_supa.patch.attention.
    # Default 0 uses the refactored operators under vllm_supa.v1.attention.ops.
    "VLLM_SUPA_USE_LEGACY_ATTENTION_PATCH": lambda: bool(
        int(os.getenv("VLLM_SUPA_USE_LEGACY_ATTENTION_PATCH", "0"))),
}


def __getattr__(name: str):
    """
    Gets environment variables lazily.

    NOTE: After enable_envs_cache() invocation (which triggered after service
    initialization), all environment variables will be cached.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
