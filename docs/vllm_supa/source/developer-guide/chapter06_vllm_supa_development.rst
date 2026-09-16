开发规范
========

vLLM 插件系统
--------------------

vLLM 通过 Python package entry point 自动发现和加载外部插件。vLLM SUPA 使用 vLLM 插件系统开发和注册插件，安装后，
``setup.py`` 注册以下入口：

.. code-block:: ini

   [vllm.platform_plugins]
   biren = vllm_supa:register

   [vllm.general_plugins]
   biren_patch = vllm_supa:register_patch
   biren_enhanced_model = vllm_supa:register_model

其中，平台插件用于声明设备平台；
- ``register()`` 返回 ``vllm_supa.platform.SUPAPlatform`` 的导入路径。
- ``register_patch()`` 调用补丁编排器并应用 vLLM/SUPA 补丁，
- ``register_model()`` 预留给模型增强逻辑。

更多插件机制和 entry point 约定请参考 upstream vLLM 的
`Plugin System <https://docs.vllm.ai/en/latest/design/plugin_system/>`_。

新增补丁
--------

1. 在 ``vllm_supa/patch/`` 的对应子目录新增模块。
2. 使用 ``patch_to_with_log``，并在补丁初始化路径中导入模块。
3. 保持上游函数签名和公共接口兼容。
4. 为设备路径补充 PyTorch CPU 参考和最小回归测试。

新增 kernel
------------

1. 在 ``csrc/`` 中添加 C++/SUPA 内核源文件。
2. 在 binding 中注册 Python 接口。
3. 在 CMake target 中加入源文件。
4. 从 ``vllm_supa`` 暴露 custom op，并添加对应测试。

代码 format
------------

.. code-block:: shell

   flake8 vllm_supa/
   black --line-length 120 <file>

导入顺序按标准库、第三方依赖、vLLM、vLLM SUPA 分组。提交前由 pre-commit hook
执行 flake8 检查。
