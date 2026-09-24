# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# flake8: noqa
import os
import subprocess
import sys

import pytest


@pytest.mark.ci_mini
@pytest.mark.sanity
@pytest.mark.gcuSanity
@pytest.mark.regression
@pytest.mark.gcuRegression
def test_import():
    import torch
    import vllm
    import vllm_supa
    from vllm_supa.utils import import_supa_kernels
    import_supa_kernels()

    from vllm.cumem_allocator import (
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )
