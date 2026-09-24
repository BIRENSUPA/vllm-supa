# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.
"""Patch vllm.v1.worker.gpu_worker.Worker.determine_available_memory.

When env var VLLM_SUPA_SKIP_PROFILE_RUN=1 AND cache_config.kv_cache_memory_bytes
is provided, skip profile_run() entirely and use the user-supplied byte count
after applying vLLM's frontend multimodal GPU memory reservation.

Rationale:
  profile_run() is a full forward pass at max_num_tokens used by vllm to
  measure peak GPU memory and size the KV cache pool. On Biren cmodel one
  forward at this scale can take 2.5+ hours per call.

  vllm has an existing branch that, when kv_cache_memory_bytes is set, skips
  the memory_profiling math but still calls profile_run() to "compile the model
  for max_num_batched_tokens". On SUPA + enforce_eager, no such compilation
  happens, so that forward is pure waste time.

  This patch makes the skip more thorough: when the env var is set and the user
  has supplied a known-good kv_cache_memory_bytes, profile_run is bypassed
  entirely.

Safety gates:
  - Env var off by default. Off means upstream behavior.
  - kv_cache_memory_bytes must be set. Without it the skip cannot know what KV
    pool size to return.
  - A loud warning is logged so the bypass is visible in run logs.
"""

from vllm.logger import init_logger
from vllm.multimodal.gpu_ipc_memory import reserve_mm_ipc_gpu_memory
from vllm.utils.gpu_sync_debug import enable_gpu_sync_check
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.startup_plan import maybe_apply_startup_plan
from vllm.v1.worker.worker_base import CompilationTimes

import vllm_supa.envs as envs
from vllm_supa.patch_to_with_log import patch_to_with_log

logger = init_logger(__name__)

_SKIP_PROFILE_ENV_VAR = "VLLM_SUPA_SKIP_PROFILE_RUN"
_SKIP_WARMUP_ENV_VAR = "VLLM_SUPA_SKIP_WARMUP_RUN"


@patch_to_with_log(Worker)
def determine_available_memory(self) -> int:
    """vllm-supa patched determine_available_memory."""
    maybe_apply_startup_plan(self)

    skip_env = envs.VLLM_SUPA_SKIP_PROFILE_RUN
    cfg_bytes = self.cache_config.kv_cache_memory_bytes
    if skip_env and cfg_bytes:
        gib = cfg_bytes / (1024 ** 3)
        logger.warning(
            "[vllm-supa] %s=1 + kv_cache_memory_bytes=%.2f GiB set: "
            "skipping profile_run() and memory_profiling entirely. "
            "Using the user-supplied byte count after frontend multimodal "
            "GPU memory reservation. This bypass does not validate memory "
            "usage; only use when you have confirmed the size from a prior "
            "unmodified run.",
            _SKIP_PROFILE_ENV_VAR,
            gib,
        )
        from vllm.utils.torch_utils import current_stream

        # The normal profile_run() executes a full model forward and initializes
        # vLLM's dedicated main stream as a side effect. When this SUPA fast path
        # skips profile_run(), do that explicitly so later MoE shared-expert
        # overlap does not lazily switch streams after routed_input_transform has
        # already queued work on the legacy default stream.
        current_stream()
        return reserve_mm_ipc_gpu_memory(
            cfg_bytes,
            self.model_config.multimodal_config,
            getattr(self.parallel_config, "_api_process_count", 1),
        )

    if skip_env and not cfg_bytes:
        logger.warning(
            "[vllm-supa] %s=1 set but kv_cache_memory_bytes is not provided: "
            "ignoring the skip request and falling through to upstream "
            "profile_run. Pass kv_cache_memory_bytes=N in LLM(...) to "
            "actually take effect.",
            _SKIP_PROFILE_ENV_VAR,
        )
    return self._orig_determine_available_memory()


@patch_to_with_log(Worker)
def compile_or_warm_up_model(self) -> CompilationTimes:
    """vllm-supa patched compile_or_warm_up_model."""
    if envs.VLLM_SUPA_SKIP_WARMUP_RUN:
        logger.warning(
            "[vllm-supa] %s=1: skipping compile/warmup dummy runs, "
            "kernel_warmup, CUDA graph capture, and sampler warmup. "
            "The first real generate request will pay any lazy compilation, "
            "autotune, and workspace allocation costs.",
            _SKIP_WARMUP_ENV_VAR,
        )
        set_random_seed(self.model_config.seed)
        enable_gpu_sync_check()
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    return self._orig_compile_or_warm_up_model()
