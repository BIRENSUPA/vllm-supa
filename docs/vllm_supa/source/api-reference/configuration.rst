运行时配置
==========

运行时配置
----------

* ``device_type``：由平台自动设置为 ``supa``。
* ``worker_cls``：设为 ``auto`` 时使用 SUPA Worker。
* ``block_size``：由平台结合注意力后端自动检查和调整。
* ``distributed_executor_backend``：BR2XX 分布式通信使用 BCCL。
