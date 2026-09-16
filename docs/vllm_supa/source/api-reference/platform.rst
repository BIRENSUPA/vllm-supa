SUPAPlatform
============

``SUPAPlatform`` 将 vLLM 运行时绑定到 BR2XX/SUPA 设备栈。

平台属性
--------

.. list-table:: 平台属性
   :header-rows: 1
   :widths: 35 65

   * - 属性
     - 值
   * - 设备类型
     - ``supa``
   * - Dispatch key
     - ``PrivateUse1``
   * - 分布式后端
     - ``bccl``
   * - 默认 Worker
     - ``vllm_supa.v1.worker.supa_worker.Worker``

配置行为
--------

``check_and_update_config`` 会根据注意力后端调整 KV cache block size，并在必要时
设置 Worker 和多模态调度选项。``import_kernels`` 负责加载 vLLM SUPA 的编译扩展。
