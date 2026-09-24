# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import sys
import types
from functools import cache

import logging
import os
import typing

import torch
import torch._library.infer_schema as _infer_schema
from torch.library import Library

from .patch_to_with_log import patch_to_with_log

_logger = None


def _get_logger():
    """Lazy singleton for vllm logger to avoid circular import.

    vLLM 0.24.0 triggers platform resolution at import time
    (env_override → torch_utils → platforms), so importing vllm.logger
    at module level causes a circular import when vllm_supa is still
    being initialized.
    """
    global _logger
    if _logger is None:
        from vllm.logger import init_logger

        _logger = init_logger(__name__)
    return _logger


def init_vllm_supa() -> None:
    """Initialize vllm_supa: set module stubs, env vars, logging, and patches."""
    # not change function order
    _init_module_stubs()
    _init_env_vars()
    _init_vllm_supa_logging()


def _init_module_stubs() -> None:
    """Stub out vllm compiled modules to avoid importing them"""

    def disable_module(name: str) -> None:
        sys.modules.setdefault(name, None)  # type: ignore[arg-type]

    # Compiled extensions need module objects so imports can resolve safely.
    sys.modules.setdefault("vllm._C", types.ModuleType("vllm._C"))
    sys.modules.setdefault("vllm._C_stable_libtorch", types.ModuleType("vllm._C_stable_libtorch"))

    sys.modules.setdefault("vllm.vllm_flash_attn._vllm_fa3_C", types.ModuleType("vllm.vllm_flash_attn._vllm_fa3_C"))
    sys.modules.setdefault("vllm._flashmla_C", types.ModuleType("vllm._flashmla_C"))
    sys.modules.setdefault("vllm._flashmla_extension_C", types.ModuleType("vllm._flashmla_extension_C"))

    sys.modules.setdefault("vllm._qutlass_C", types.ModuleType("vllm._qutlass_C"))
    sys.modules.setdefault("vllm._flashkda_C", types.ModuleType("vllm._flashkda_C"))

    # Optional packages are disabled with None to prevent unavailable imports.
    disable_module("vllm.third_party.deep_gemm")
    disable_module("triton_kernels")
    disable_module("vllm.third_party.triton_kernels")


def _init_env_vars() -> None:
    """Set environment variables required by vllm_supa."""
    os.environ["TORCHINDUCTOR_WORKER_START"] = "spawn"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


def _init_vllm_supa_logging() -> None:
    """Propagate VLLM_LOGGING_LEVEL to the vllm_supa namespace.

    vllm configures its own "vllm" logger with propagate=False, so vllm_supa.*
    loggers never reach that handler. Mirror the level and reuse the handler.
    """
    level = os.environ.get("VLLM_LOGGING_LEVEL", "INFO").upper()
    supa_logger = logging.getLogger("vllm_supa")
    supa_logger.setLevel(level)
    supa_logger.propagate = False
    for h in logging.getLogger("vllm").handlers:
        supa_logger.addHandler(h)


# Extend infer_schema to support torch.Generator types
_infer_schema.SUPPORTED_PARAM_TYPES[torch.Generator] = "Generator"
_infer_schema.SUPPORTED_PARAM_TYPES[typing.Optional[torch.Generator]] = "Generator?"


@patch_to_with_log(Library)
def impl(self, op_name, fn, dispatch_key="", *, with_keyset=False, allow_override=False):
    if dispatch_key == "CUDA":
        dispatch_key = "PrivateUse1"
    self._orig_impl(op_name, fn, dispatch_key=dispatch_key, with_keyset=with_keyset, allow_override=allow_override)


if not hasattr(torch.Tensor, "slice"):

    def _tensor_slice(self, dim=0, start=None, end=None, step=1):
        return torch.ops.aten.slice.Tensor(self, dim, start, end, step)

    torch.Tensor.slice = _tensor_slice  # type: ignore[attr-defined]


@cache
def is_flash_attn_available(raise_error: bool = False) -> bool:
    try:
        # Temporarily use suattention until flashattn_train is available.
        import suattention  # noqa: F401

        return True
    except ImportError:
        _get_logger().warning_once(
            "Flash attention is not available. Please ensure that flash attention is correctly installed."
        )
        if raise_error:
            raise
    return False


@cache
def is_flashmla_available(raise_error: bool = False) -> bool:
    try:
        import flash_mla  # noqa: F401

        return True
    except ImportError:
        _get_logger().warning_once("FlashMLA is not available. Please ensure that flash_mla is correctly installed.")
        if raise_error:
            raise
    return False


def import_supa_kernels() -> None:
    """Import SUPA platform-specific C extension modules."""
    import vllm_supa._C  # noqa: F401
    import vllm_supa._C_stable_libtorch  # noqa: F401
    import vllm_supa._moe_C_stable_libtorch  # noqa: F401
    import vllm_supa._supa_C  # noqa: F401
