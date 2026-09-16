补丁机制
========

注册流程
--------

vLLM 通过 entry point 加载 ``vllm_supa:register_patch``，该入口导入
``vllm_supa.patch`` 并应用补丁。补丁使用 ``patch_to_with_log``，目标已有同名
符号时，原实现会备份为 ``_orig_<name>``。

使用示例
~~~~~~~~

以下示例来自 ``vllm_supa/patch/torch/nn/modules/normalization.py``，使用
``patch_to_with_log`` 替换 ``nn.LayerNorm.forward``。当输入不满足 SUPA
kernel 条件时，通过装饰器保存的 ``self._orig_forward`` 回退到上游实现：

.. code-block:: python

   import torch
   from torch import nn

   from vllm_supa.patch_to_with_log import patch_to_with_log

   @patch_to_with_log(nn.LayerNorm)
   def forward(self, input: torch.Tensor) -> torch.Tensor:
       # _can_use_supa_layernorm 为该模块中的条件检查辅助函数。
       if _can_use_supa_layernorm(self, input):
           return torch.ops._supa_C.layernorm(
               input, self.weight, self.bias, self.eps)
       return self._orig_forward(input)

补丁模块还必须在 ``vllm_supa/patch/__init__.py`` 的
``apply_patches()`` 中导入，导入模块时装饰器才会执行：

.. code-block:: python

   from .torch.nn.modules import normalization  # noqa: F401

安装 vLLM SUPA 后，vLLM 会通过 entry point 自动调用注册入口。调试或测试时，
也可以显式触发同一个入口：

.. code-block:: python

   import vllm_supa

   vllm_supa.register_patch()

补丁注册只会执行一次；重复调用 ``register_patch()`` 不会重复应用补丁。

导入顺序
--------

源码中的 import 按以下类别分组，并在组之间保留空行：

.. code-block:: python

   import os

   import torch
   import torch_supa

   import vllm

   import vllm_supa

补丁日志
--------

.. code-block:: shell

   PATCH_LOG_LEVEL=debug python3 -c 'import vllm_supa'

日志包含 rank、local rank、目标类/函数和源文件位置。生产环境默认使用
``critical``，排查插件加载问题时再开启 ``debug``。
