扩展边界
========

补丁
----

补丁使用 ``patch_to_with_log`` 封装器，以便保留原函数并记录 rank、目标符号和
源文件位置。补丁模块必须在 ``vllm_supa/patch/__init__.py`` 或初始化路径中导入，
否则 entry point 虽然注册成功，补丁也不会生效。

算子
----

新增 SUPA 内核时，应在 ``csrc/`` 中实现内核，在 C++ binding 中注册，并加入对应的
CMake target。Python 侧通过 ``vllm_supa`` 的 custom op 或 ``torch.ops`` 暴露。

导入顺序
--------

源码统一按以下顺序组织 import，并在类别之间保留空行，以避免 format 工具重排：

.. code-block:: python

   import os

   import torch
   import torch_supa

   import vllm

   import vllm_supa

兼容性
------

平台代码应优先调用上游 vLLM 的公共接口。只有在设备、通信或算子行为确实不同的
情况下才添加补丁，并为补丁路径提供 CPU 参考或回归测试。
