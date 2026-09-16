代码注释规范
============

代码注释分为 CPP/SUDA 层和 Python 层。注释应解释实现原因、接口约束和容易被误改的
行为；不要为显而易见的赋值或控制流添加逐行旁白。

CPP/SUDA 层注释
---------------

在 ``.cpp``、``.cu`` 和 ``.su`` 文件中，复杂 kernel 或 binding 前应说明：

* 输入输出的形状、数据类型、布局和量化粒度。
* grid/block 映射、同步要求以及 BR2XX/SUDA 特有的限制。
* 性能相关选择（例如 tile 大小、临时 workspace）及其适用条件。
* 与 Python custom op schema 的对应关系，以及异常输入的处理方式。

使用 ``//`` 编写简短行注释，使用 ``/* ... */`` 说明跨行约束；内核实现的关键步骤
应在逻辑块前集中说明：

.. code-block:: cpp

   // One output row maps to one BR2XX program instance. Keep the layout
   // contiguous because the SUDA kernel assumes unit-stride access.
   void launch_layernorm(const Tensor& input, Tensor& output) {
       // The reduction must complete before the normalization write-back.
       reduce_row(input, output);
   }

Python 层注释
-------------

在 ``vllm_supa/`` Python 模块中，模块或公共函数使用 docstring 说明用途；补丁附近的
注释应说明：

* 被替换的 upstream 行为以及选择该补丁的原因。
* 环境变量、dispatch key、设备能力检查和 fallback 的生效条件。
* monkey patch 的导入顺序、幂等性和兼容性约束。
* Fake/Meta 算子只用于形状/设备传播，不提供真实数值结果。

例如，``fake_supa_ops.py`` 中应解释可选 schema 跳过注册和重复注册容错的原因：

.. code-block:: python

   def _register(name: str, fn) -> None:
       # Optional SUPA builds may omit this schema; leave registration to the
       # available extension instead of failing plugin initialization.
       try:
           torch._C._dispatch_find_schema_or_throw(name, "")
       except RuntimeError:
           return
       torch.library.register_fake(name)(fn)

注释应与代码行为同步；修改算子签名、补丁条件或 fallback 路径时，同时更新对应注释
和 docstring。

Python 文档字符串规范（PEP 257）
--------------------------------

Python 模块、类、函数和方法使用 docstring 描述对外可见的用途和行为，遵循
`PEP 257 <https://peps.python.org/pep-0257/>`_：

* docstring 必须是模块、类或函数体中的第一条语句。
* 简短对象使用单行 docstring，开头概括用途并以句号结尾。
* 多行 docstring 首行写摘要，随后空一行，再补充参数约束、返回值、异常和副作用。
* 使用三重双引号（``\"\"\"``），结束引号与内容保持清晰的缩进。
* 补丁函数应说明被替换的 upstream 接口和兼容性要求；Fake/Meta 函数应说明仅用于
  ``torch.compile`` 的形状/设备传播。

示例：

.. code-block:: python

   def register_patch() -> None:
       """Apply vLLM SUPA runtime patches once.

       The entry point is called by vLLM plugin discovery. Patch imports are
       idempotent so repeated discovery does not register operators twice.
       """
       apply_patches()

不要用行尾注释替代接口文档；当实现逻辑变化时，应同步更新 docstring 和相关注释。
