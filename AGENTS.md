# AGENTS.md

This file provides guidance to AI coding assistants when working with code in this repository.

## Project Overview

vllm-supa is a **plugin/patch repository** that adds Biren BR200 SUPA GPU support to a specific upstream vLLM release. It does NOT fork vLLM source directly — instead it uses vLLM's plugin system (entry points) and monkey-patching via `fastcore.basics.patch_to` to override behavior at runtime.

**Key concept:** The upstream `vllm` package is installed separately. This repo installs alongside it as `vllm_supa` and registers itself through entry points. The target upstream release or commit is pinned authoritatively in `.env_setup/versions.sh` (`VLLM_VERSION`); `setup.py` resolves a commit ref to the installed wheel's semantic version at build time. Always check that file rather than relying on a version or commit hardcoded elsewhere.

## Virtual Environment

If `.venv/bin/activate` exists, always run `source .venv/bin/activate` before any Python command.

## Build & Development Commands

```bash
# Full environment setup (installs torch, torch_supa, vllm, mounts shared models, editable install)
./.ci/build.sh --setup-env

# Development install (requires torch_supa already installed)
python3 -m pip install -e . --no-build-isolation

# Production wheel build
python3 setup.py bdist_wheel

# Remove compiled artifacts without rebuilding
python3 setup.py clean

# Clean build/dist metadata, then immediately perform a full wheel build
./.ci/build.sh -c
```

**Build prerequisites:** `torch_supa` (Biren PyTorch fork) must be installed before building. The CMake build uses SUDA (Biren's CUDA equivalent) and BRCC compiler for `.su` files.

### Building and Cleaning `csrc`

`csrc/` contains source code and must never be deleted to clean a build. Its
generated object files, build-system state, CMake cache, and linked extensions
live in `build/`; linked `.so` files are also copied into `vllm_supa/`.

Choose the build command by the required output: use the direct CMake `install`
target while iterating on native extensions, and use `bdist_wheel` when a
distributable or final-verification artifact is required.

Before any build, check whether `build/` already contains build-system state:

```bash
if [[ -f build/CMakeCache.txt ]]; then
  rg '^(CMAKE_GENERATOR|CMAKE_BUILD_TYPE|CMAKE_INSTALL_PREFIX|SUPA_ARCH)(:.*)?=' \
    build/CMakeCache.txt
fi
```

When a cache exists, keep using the generator recorded in
`CMAKE_GENERATOR`; do not configure the same `build/` directory with a
different generator. To switch generators, first remove the cached build state
with `python3 setup.py clean`, then configure a new `build/` directory.

### Direct CMake Builds

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSUPA_ARCH=arch_20 \
  -DCMAKE_INSTALL_PREFIX="$PWD"

# Build and install all extension targets into vllm_supa/.
cmake --build build --target install -j16
```

Use `cmake --build build --target install -j16` for local, incremental
development after editing C++ or `.su` files. The build system rebuilds only
stale native sources, and the `install` target updates the compiled extensions
under `vllm_supa/` so they can be tested from the source checkout. The project
defaults `CMAKE_INSTALL_PREFIX` to the repository root and all extension targets
use `vllm_supa` as their install destination.

The direct CMake command does **not** create or refresh `dist/*.whl`, assemble
Python package contents, or validate wheel metadata. It must not be used as the
final packaging verification.

### Production Wheel Builds

Run the production wheel build from the repository root when any of the
following applies: a distributable wheel is required; Python package contents
or packaging metadata changed; or the change is ready for final verification
or handoff.

```bash
source .venv/bin/activate  # when .venv exists
MAX_JOBS=16 python3 setup.py bdist_wheel
```

`bdist_wheel` reuses the build cache and rebuilds stale native sources, copies
the compiled extensions into `vllm_supa/`, and assembles the wheel under
`dist/`. This is the required final verification command; do not substitute
`pip wheel` or another build frontend. It is not necessary for every local
C++/`.su` edit when the incremental CMake `install` target is sufficient.

To discard all cached `csrc` compilation results before rebuilding, run:

```bash
source .venv/bin/activate  # when .venv exists
python3 setup.py clean
MAX_JOBS=16 python3 setup.py bdist_wheel
```

`python3 setup.py clean` removes `build/`, `vllm_supa.egg-info/`, and compiled
`.so`/`.so.*` files copied into `vllm_supa/`. It does not remove `dist/`. If a
fully fresh wheel output directory is required, `./.ci/build.sh -c` removes
`build/`, `dist/`, and top-level `*egg-info`, then continues directly into the
CI-style dependency setup and wheel build. Do not use it as a clean-only
command.

### Skipping the Marlin Kernels

`USE_MARLIN_KERNELS` defaults to `ON`. Set it to `OFF` for local iteration when
the Marlin GEMM kernels are not needed:

```bash
# Direct CMake
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSUPA_ARCH=arch_20 \
  -DCMAKE_INSTALL_PREFIX="$PWD" -DUSE_MARLIN_KERNELS=OFF

# setup.py (env var is forwarded as the CMake option)
USE_MARLIN_KERNELS=OFF MAX_JOBS=16 python3 setup.py bdist_wheel
```

With `OFF`, Marlin GEMM implementations are unavailable and models that dispatch
to them will fail. Use the default `ON` for release wheels.

## Running Tests

```bash
# Run all tests (from test/ directory)
cd test && python3 ./start_test.py -svx

# Run with marker filter
cd test && python3 ./start_test.py -svx -m ci_mini
cd test && python3 ./start_test.py -svx -m sanity
cd test && python3 ./start_test.py -svx -m gcuSanity

# Run a single test file
cd test && python3 ./start_test.py -svx kernels/core/test_layernorm.py

# Run a specific test function
cd test && python3 ./start_test.py -svx kernels/moe/test_fused_topk.py::test_fused_topk
```

Test markers: `ci_mini`, `sanity`, `gcuSanity`, `regression`, `gcuRegression`, `full`

### CPU Reference Pattern

Kernel tests must initialize source tensors and compute reference results on
CPU, then explicitly transfer only the operator inputs to the accelerator.
Follow the existing `test_supa_attn.py` pattern:

```python
cpu_device = torch.device("cpu")
device = torch.device("cuda:0")

x_cpu = torch.randn(shape, dtype=dtype, device=cpu_device)
ref_output = reference_impl(x_cpu)

x = x_cpu.to(device)
output = custom_op(x)
torch.testing.assert_close(output.cpu(), ref_output)
```

Pass `cpu_device` into helpers that create reference tensors, or derive new
reference tensors from an existing CPU tensor with `device=x.device`. Do not
rely on PyTorch's default device for test inputs or reference computation.

## Linting

```bash
# Flake8 (max line length 120)
flake8 vllm_supa/

# Pre-commit hook runs flake8 on staged files automatically
# Fix formatting issues with black:
black <file>
```

## Architecture

### Plugin Registration (setup.py entry points)

```
vllm.platform_plugins:  "biren" → vllm_supa:register()      → returns SUPAPlatform class path
vllm.general_plugins:   "biren_patch" → vllm_supa:register_patch()  → applies monkey patches
                        "biren_enhanced_model" → vllm_supa:register_model()
```

### Directory Layout

```
vllm_supa/              # Python plugin package
├── __init__.py       # Entry points, module stubs, env vars
├── platform.py       # SUPAPlatform (extends CudaPlatformBase)
├── patch/            # Monkey patches applied to upstream vLLM
│   ├── __init__.py   # apply_patches() orchestrator
│   ├── registry.py   # Adds SUPA_ATTN/SUPA_MLA to AttentionBackendEnum
│   ├── _custom_ops.py # cutlass_scaled_mm, fused_add_rms_norm overrides
│   ├── attention/    # Flash attention & FlashMLA custom op registration
│   ├── moe/          # MOE FP8 patches
│   └── ...           # pynvml, pynccl, triton, rejection_sampler patches
├── v1/attention/backends/  # SUPA attention backend implementations
│   ├── supa_attention.py
│   └── supa_mla.py
└── *.so              # Pre-built extension modules

csrc/                 # C++ / SUPA kernel source
├── supa/             # Biren-specific kernels; see csrc/supa/README.md for layout
├── attention/        # Paged attention kernels
├── quantization/     # Quantization schemes (GPTQ, AWQ, Marlin, etc.)
├── moe/              # MOE alignment and topk kernels
└── torch_bindings.cpp # Python↔C++ interface

test/                 # Test suite
├── start_test.py     # Custom pytest runner
├── kernels/          # Kernel unit tests (attention, moe, core)
└── conftest.py       # Fixtures (default_vllm_config, workspace_init)
```

### Compiled Extensions

| Module | Content |
|--------|---------|
| `_C.abi3.so` | Standard vLLM CUDA kernels (attention, quant, cache) |
| `_supa_C.abi3.so` | Biren SUPA kernels (.su files) |
| `_moe_C.abi3.so` | MOE-specific kernels |
| `torch_cuda_stub.so` | Fake libtorch_cuda for compatibility |

### Key Design Patterns

1. **Monkey patching with logging:** All patches use `patch_to_with_log.py` wrapper around `fastcore.patch_to`. Original functions preserved as `_orig_<name>`.

2. **PrivateUse1 dispatch:** SUPA device registers as PyTorch's PrivateUse1 backend. Custom ops use `torch.library` with dispatch key `"PrivateUse1"`.

3. **Module stubs in `__init__.py`:** `vllm._C` and `vllm._C_stable_libtorch` are stubbed to prevent vLLM from loading its own CUDA `.so` files — replaced by vllm_supa's versions.

4. **Attention backend priority:** Platform enforces: `SUPA_MLA → FLASHMLA → TRITON_MLA` (for MLA models) or `SUPA_ATTN → FLASH_ATTN → TRITON_ATTN` (normal models). Block size fixed at 128.

5. **Distributed backend:** Uses BCCL (Biren CCL) instead of NCCL. NVML is stubbed out.

## Git Workflow

- **Main branch:** `develop`
- **Commit format:** `[TYPE] JIRA_ID: description` (e.g., `[ENH] MLFW-10388: Add cache/MLA/MoE ops`)
- **CI environment:** qemu (default), silicon, sucloud
- **CI test level:** sanity (default for MR)

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SUPA_VISIBLE_DEVICES` | Control visible SUPA devices |
| `PATCH_LOG_LEVEL` | Patch logging verbosity (debug/critical) |
| `MAX_JOBS` | Parallel compilation jobs (default: 16) |
| `USE_MARLIN_KERNELS` | Set to `OFF` to drop the Marlin GEMM kernels from the build (default: `ON`) |
| `VLLM_WORKER_MULTIPROC_METHOD` | Forced to "spawn" |

## Adding New Patches

1. Create file in `vllm_supa/patch/` (or appropriate subdirectory)
2. Use `from vllm_supa.patch_to_with_log import patch_to` decorator pattern
3. Import the patch in `vllm_supa/patch/__init__.py` or `vllm_supa/__init__.py`
4. Import order: `os/sys → third-party/torch → vllm → vllm_supa`

## Adding New SUPA Kernels

1. Write reusable `.su` kernels in `csrc/supa/<category>/` and model-specific kernels in `csrc/supa/models/<model>/`; follow `csrc/supa/README.md`
2. Add C++ torch binding in `csrc/supa/torch_bindings.cpp`
3. Register in CMakeLists.txt under the appropriate extension target (`_supa_C`, or `_supa_mira_C` for `moe/fused_moe/`); keep included helpers out of source lists
4. Expose via `vllm_supa/patch/_custom_ops.py` if needed from Python
