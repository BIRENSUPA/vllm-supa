快速开始
========

更多通用的 vLLM 使用方法请参考 upstream vLLM 的
`Quickstart <https://docs.vllm.com.cn/en/latest/getting_started/quickstart/>`_。

离线推理
--------

.. code-block:: python

   from vllm import LLM, SamplingParams

   llm = LLM(
       model="Qwen/Qwen3-0.6B",
       enforce_eager=True,
       block_size=128,
       max_model_len=512,
       compilation_config=0,
   )
   outputs = llm.generate(
       ["What is your name?"],
       SamplingParams(temperature=0, max_tokens=128),
   )
   print(outputs[0].outputs[0].text)

在线服务
--------

.. code-block:: shell

   vllm serve Qwen/Qwen3-0.6B \
       --dtype bfloat16 \
       --max-model-len 512 \
       --enforce-eager \
       --attention-backend FLASH_ATTN

服务启动后，可通过 OpenAI 兼容的 ``/v1/chat/completions`` 端点访问。
您也可以使用输入提示查询模型：

.. code-block:: shell

   curl http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "Qwen/Qwen3-0.6B",
           "prompt": "Shanghai is a",
           "max_tokens": 10,
           "temperature": 0
       }'
