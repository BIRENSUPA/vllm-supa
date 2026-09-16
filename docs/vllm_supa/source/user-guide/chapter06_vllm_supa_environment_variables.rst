环境变量
========

环境变量应在启动 Python 进程前设置，至少应在首次导入 ``torch``、``torch_supa``
或 ``vllm_supa`` 前完成设置。未特别说明时，布尔开关使用 ``0`` 表示关闭、``1``
表示开启。

.. note::

   导入 ``vllm_supa`` 时会调用 ``vllm_supa.utils._init_env_vars()``，并强制设置以下环境变量：

   * ``TORCHINDUCTOR_WORKER_START=spawn``
   * ``VLLM_WORKER_MULTIPROC_METHOD=spawn``
   * ``VLLM_USE_FLASHINFER_SAMPLER=0``
   * ``DG_FP8_GEMM_BACKEND=sublas``

   即使用户在导入前设置了其他值，也会被覆盖。

壁仞硬件环境变量
----------------

这组变量控制 SUPA/BR2XX 设备可见性。

.. list-table:: 壁仞硬件环境变量
   :header-rows: 1
   :widths: 42 18 40

   * - 变量
     - 默认值
     - 作用
   * - ``SUPA_VISIBLE_DEVICES``
     - 未设置
     - 控制当前进程可见的 SUPA 设备，例如 ``0`` 或 ``0,1``。

上游 vLLM 环境变量
------------------

这组变量由上游 vLLM 或其依赖运行时定义。其中标记为“初始化时强制设置”的值
由 vLLM SUPA 的 ``_init_env_vars()`` 写入。

.. list-table:: 上游 vLLM 环境变量
   :header-rows: 1
   :widths: 42 18 40

   * - 变量
     - 默认值
     - 作用
   * - ``VLLM_WORKER_MULTIPROC_METHOD``
     - ``spawn`` （初始化时强制设置）
     - 设置 vLLM worker 多进程启动方式。
   * - ``VLLM_LOGGING_LEVEL``
     - ``INFO``
     - 设置 vLLM 及插件日志级别。
   * - ``VLLM_USE_FLASHINFER_SAMPLER``
     - ``0`` （初始化时强制设置）
     - 控制 vLLM FlashInfer sampler 路径；vLLM SUPA 初始化时将其关闭。
   * - ``TORCHINDUCTOR_WORKER_START``
     - ``spawn`` （初始化时强制设置）
     - 设置 TorchInductor worker 启动方式。

DeepGEMM 环境变量
-----------------

这组变量由 DeepGEMM 定义，vLLM SUPA 在初始化时设置适用于 SUPA 的后端。

.. list-table:: DeepGEMM 环境变量
   :header-rows: 1
   :widths: 42 18 40

   * - 变量
     - 默认值
     - 作用
   * - ``DG_FP8_GEMM_BACKEND``
     - ``sublas`` （初始化时强制设置）
     - 选择 DeepGEMM FP8 GEMM 后端；vLLM SUPA 初始化时选择 SubLAS。

vLLM SUPA 环境变量
------------------

这组变量由 ``vllm_supa.envs`` 提供。当前分支只保留以下运行时开关；变量采用惰性
读取，并在服务初始化后缓存。

.. list-table:: vLLM SUPA 环境变量
   :class: longtable
   :header-rows: 1
   :widths: 56 10 34

   * - 变量
     - 默认值
     - 作用
   * - ``VLLM_SUPA_SKIP_PROFILE_RUN``
     - ``0``
     - 跳过 worker 的 profile_run。（即将移除）
   * - ``VLLM_SUPA_SKIP_WARMUP_RUN``
     - ``0``
     - 跳过编译或 warmup dummy run。（即将移除）
   * - ``VLLM_SUPA_ENABLE_DEEP_GEMM``
     - ``1``
     - 启用 SUPA DeepGEMM 选择。
   * - ``VLLM_SUPA_USE_LEGACY_ATTENTION_PATCH``
     - ``0``
     - 临时启用旧版注意力补丁，默认使用重构后的实现；该变量已废弃。
   * - ``PATCH_LOG_LEVEL``
     - ``critical``
     - 设置 SUPA 补丁日志级别；排障时使用 ``debug``。

修改环境变量后必须重新启动 vLLM 进程；运行期间修改通常不会影响已经初始化的组件。
