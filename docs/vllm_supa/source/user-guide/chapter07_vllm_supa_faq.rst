常见问题
========

**Q：编译时找不到 CMake、SUDA 或 BRCC，怎么办？**

**A：**

先加载 SUPA SDK 环境，确认 ``cmake``、``suda`` 和 ``brcc`` 在 ``PATH`` 中，再
检查 TorchSUPA 版本是否与 ``.env_setup/versions.sh`` 一致。

----

**Q：安装后插件未生效，怎么办？**

**A：**

确认上游 vLLM 和 ``vllm_supa`` 安装在同一个 Python 环境，并检查 entry point：

.. code-block:: shell

   python3 -c 'import vllm_supa; print(vllm_supa.register())'

如果补丁未出现日志，使用 ``PATCH_LOG_LEVEL=debug`` 重试，并确认目标模块已在
``vllm_supa.patch`` 初始化流程中导入。

----

**Q：运行时设备不可见，怎么办？**

**A：**

检查 ``SUPA_VISIBLE_DEVICES``、驱动权限和 SDK 环境；容器环境还需确认设备已正确
映射到容器内。
