附录
====

术语表
------

.. list-table:: 术语
   :header-rows: 1
   :widths: 25 75

   * - 术语
     - 说明
   * - SUPA
     - 壁仞统一并行计算平台。
   * - BR2XX
     - 本插件支持的壁仞硬件系列。
   * - BIRENSUPA
     - 壁仞软件栈的统称，包含运行时、工具链和配套开发库。
   * - vLLM SUPA
     - 面向 BR2XX/SUPA 设备的 vLLM 插件，为上游 vLLM 提供平台适配和算子补丁。
   * - TorchSUPA
     - 提供 SUPA PyTorch 运行时和 PrivateUse1 设备支持的依赖包。
   * - SUDA
     - 将 CUDA 兼容代码转发到 SUPA 的工具链。
   * - BCCL
     - BIRENSUPA 集合通信库。
   * - BRCC
     - BIRENSUPA 编译器。
   * - PrivateUse1
     - PyTorch 为第三方设备后端预留的 dispatch key。
   * - SubLAS
     - BIRENSUPA 提供的线性代数库，用于 SUPA 上的矩阵乘法和分组矩阵乘法。
