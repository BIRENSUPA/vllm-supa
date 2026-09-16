运行时架构
==========

插件定位
--------

``vllm_supa`` 是与上游 vLLM 并列安装的平台插件，不复制 vLLM 源码。插件通过
entry point 注册 ``SUPAPlatform``，并在导入时加载针对 BR2XX/SUPA 的补丁和算子。

注册入口
--------

.. code-block:: ini

   [vllm.platform_plugins]
   biren = vllm_supa:register

   [vllm.general_plugins]
   biren_patch = vllm_supa:register_patch
   biren_enhanced_model = vllm_supa:register_model

主要模块
--------

* ``vllm_supa/platform.py``：设备类型、后端优先级、Worker 和通信后端。
* ``vllm_supa/patch/``：对上游 vLLM 的函数级补丁。
* ``vllm_supa/v1/``：SUPA 注意力和运行时实现。
* ``csrc/``：C++、CUDA 兼容代码和 SUPA 内核绑定。

设备和通信
----------

SUPA 使用 PyTorch ``PrivateUse1`` dispatch key，设备类型为 ``supa``，分布式通信
后端为 BCCL。平台会根据注意力后端和模型配置自动调整 KV cache block size。
