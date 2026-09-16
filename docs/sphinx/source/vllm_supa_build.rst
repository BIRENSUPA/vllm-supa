编译链路
========

构建入口
--------

仓库使用 ``setuptools`` 调用 CMake 编译扩展。``setup.py bdist_wheel`` 生成发布
wheel，``pip install -e . --no-build-isolation`` 用于开发模式。开发模式和设置
``DEBUG=on`` 时使用 Debug 构建，否则使用 Release 构建。

构建流程
--------

.. code-block:: text

   setup.py
     -> CMake configure (build/)
     -> CMake build
     -> 复制 *.so 到 vllm_supa/
     -> setuptools 打包

构建依赖
--------

* Python 3.10 或更高版本。
* 已安装匹配版本的 TorchSUPA 和上游 ``vllm``。
* 已加载 SUPA SDK/SUDA 环境；CMake 需要 ``cmake``，``.su`` 文件需要 BRCC。
* 依赖版本以 ``.env_setup/versions.sh`` 为准。

并行编译
--------

``MAX_JOBS`` 控制 CMake 并行任务数，默认值为 16。安装 ``ninja`` 后，构建会自动
使用 Ninja 编译池；否则使用 CMake 默认生成器。
