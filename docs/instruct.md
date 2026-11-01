Instructing an inbuilt or locally hosted language model requires configuring system prompts, selecting the matching token template for the architecture, and programmatically defining parameters.

**1. Special Token Chat Templates**
Instruction-tuned models use template tokens to isolate system rules from standard user input. Matching the model's exact expected template prevents instruction bleed:

* **ChatML (Qwen, DeepSeek, OpenHermes)**
```text
<|im_start|>system
You are a deterministic system assistant. Adhere strictly to output schemas.
<|im_end|>
<|im_start|>user
Process task payload.
<|im_end|>
<|im_start|>assistant

```


* **Llama 3 / 3.1 / 3.2 Format**
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a specialized backend automation agent.<|eot_id|><|start_header_id|>user<|end_header_id|>

Process task payload.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```


* **Mistral / Instruct Format**
```text
<s>[INST] <<SYS>>
System directives and constraints go here.
<</SYS>>

User instruction here. [/INST]

```



**2. Programmatic System Prompt Injection**

* **Hugging Face `transformers**`
```python
messages = [
    {"role": "system", "content": "You are a precise parsing engine."},
    {"role": "user", "content": "Input text to extract."}
]
formatted_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

```


* **Ollama (Modelfile Definition)**
```dockerfile
FROM llama3.2
SYSTEM """
You are an embedded control agent. 
1. Respond with raw JSON only.
2. Do not include introductory text or markdown wrappers.
"""
PARAMETER temperature 0.1
PARAMETER top_p 0.9

```


* **`llama-cpp-python` / GGUF Runtimes**
```python
from llama_cpp import Llama

llm = Llama(model_path="./model.gguf", chat_format="chatml")
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "Perform low-level string tokenization."},
        {"role": "user", "content": "Target string."}
    ]
)

```



**3. System Prompt Architecture**
To maximize instruction adherence in local models, structure system prompts into four explicit sections:

```^|D|
[ROLE & OBJECTIVE]
Define target persona, identity, and domain scope.

[STRICT CONSTRAINTS]
List negative constraints (e.g., "Do not offer conversational meta-text", "Never fabricate values").

[OPERATIONAL STEPS]
Define step-by-step reasoning or internal validation logic prior to response generation.

[OUTPUT SCHEMA]
Provide exact JSON, XML, or standard format specs.

```
