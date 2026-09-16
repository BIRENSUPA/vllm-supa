编译与安装
==========

开发环境安装
------------

该方式依赖已有 BIRENSUPA 环境，并且已经安装 vLLM SUPA 相关依赖。

以 editable 模式安装并编译 vLLM SUPA：

.. code-block:: shell

   [MAX_JOBS=8] python3 -m pip install -e . --no-build-isolation

开发模式会从源码编译 C++/SUPA 扩展；可通过 ``MAX_JOBS`` 设置并行编译任务数。

构建发布 wheel 包：

.. code-block:: shell

   python3 setup.py bdist_wheel
