安装
====

基础要求
--------

* Ubuntu 22.04、Ubuntu 24.04 或兼容的 Linux 环境。
* Python 3.10 及以上。
* BR2XX 硬件、SUPA SDK。
* Docker >= 20.10.7（使用 Docker 镜像安装时）。

版本依赖
--------

以下 upstream vLLM 和 PyTorch（TorchSUPA）版本从交付包中的
``upstream_versoin.txt`` 获取。

.. list-table:: 依赖版本
   :header-rows: 1
   :widths: 40 60

   * - 依赖
     - 版本
   * - vLLM
     - 0.27.1
   * - PyTorch
     - 2.12.0
   * - TorchSUPA
     - 2.12.0

环境准备
--------

安装 vLLM SUPA 有以下三种方式，请根据部署和开发需求选择。

* **方式一：官方 Docker 镜像**

  使用已预装 vLLM SUPA、TorchSUPA 和运行时依赖的官方镜像。

  .. attention::

     当前暂未发镜像，可关注官方发布动态。

* **方式二：wheel 包安装**

  适用于直接在操作系统镜像（如官方 Ubuntu 镜像）、物理机部署或者已有 BIRENSUPA 环境的用户。

  先安装 BIRENSUPA SDK：

  .. code-block:: shell

     sudo bash birensupa-sdk-xxxx.run

     source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh  # 设置 BIRENSUPA 环境
     suda init
     . "$HOME/.gstub/suda.sh"
     suda load

  安装 upstream vLLM：

  .. code-block:: shell

     python3 -m pip install "vllm==0.27.1" --no-deps

  安装 vLLM SUPA 配套算子依赖和 vLLM SUPA 包：

  .. code-block:: shell

     python3 -m pip install \
         flashattn_infer-*.whl \
         suattention-*.whl \
         deep_ep-*.whl \
         flash_mla-*.whl \
         triton-*.whl \
         tilelang-*.whl \
         torch_supa-*.whl \
         vllm_supa-*.whl

  .. attention::

     当前暂未发布配套算子及相关软件包，可关注官方发布动态。

* **方式三：源码构建**

  适用于需要修改代码或重新编译 SUPA 扩展的场景。该方式依赖已有 BIRENSUPA 环境，并且已经安装 vLLM SUPA 相关依赖。

  以 editable 模式安装并编译 vLLM SUPA：

  .. code-block:: shell

     [MAX_JOBS=8] python3 -m pip install -e . --no-build-isolation

  开发模式会从源码编译 C++/SUPA 扩展；可通过 ``MAX_JOBS`` 设置并行编译任务数。

  构建发布 wheel 包：

  .. code-block:: shell

     python3 setup.py bdist_wheel
