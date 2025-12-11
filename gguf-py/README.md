---
license: apache-2.0
---
Fairy±i(also named iFairy)
# Abstract
Fairy±i (iFairy) is the first 2-bit complex-valued large language model, where all weights are constrained to {±1, ±i}. By introducing complex-valued architectures and a novel quantization scheme, iFairy achieves efficient compression with minimal accuracy loss. Experiments show that it consistently outperforms existing 2-bit methods (e.g., BitNet b1.58) and approaches full-precision models on language modeling and reasoning benchmarks.
# Evaluation
**Table: Perplexity on WikiText2 and C4 validation sets (lower is better)**

| Size | Model             | WikiText2 | C4    | Avg   |
| :--- | :---------------- | :-------- | :---- | :---- |
| 700M | FP16 LLaMA        | -         | -     | 12.33 |
|      | BitNet b1.58*     | -         | -     | 12.87 |
|      | BitNet b1.58†     | 10.81     | 12.21 | 11.51 |
|      | Fairy ± i°        | 9.41 | 10.75  | 10.08 |
|      | Fairy ± i         | **10.46**     | **11.81** | **11.14** |
| 1.3B | FP16 LLaMA        | -         | -     | 11.25 |
|      | BitNet b1.58*     | -         | -     | 11.29 |
|      | Fairy ± i         | **9.35**      | **10.94** | **10.14** |

\* refers to the reported version in prior work  
† the trained version  
° the full precision Fairy±i  
**Table: Zero-shot Accuracy on Commonsense Reasoning Tasks (%)**
| Model Size | Model          | ARCe  | ARCc  | HS    | BQ    | OQ   | PQ    | WGe   | Avg.  |
| :--------- | :------------- | :---- | :---- | :---- | :---- | :--- | :---- | :---- | :---- |
| 700M       | FP16 LLaMA     | 54.70 | 23.00 | 37.00 | 60.00 | 20.20| 68.90 | 54.80 | 45.51 |
|            | BitNet b1.58 * | 51.80 | 21.40 | 35.10 | 58.20 | 20.00| 68.10 | 55.20 | 44.26 |
|            | BitNet b1.58 † | 51.77 | 22.44 | 35.30 | 58.50 | 20.80| 65.94 | 54.85 | 44.23 |
|            | Fairy±i°       | 55.68 | 24.06 | 37.79 | 60.46 | 20.60| 70.18 | 54.46 | 46.18 |
|            | **Fairy±i**        | **53.45** | **23.04** | **36.04** | **57.31** | **21.00**| **68.01** | **54.06** | **44.70** |
| 1.3B   | FP16 LLaMA | 56.90 | 23.50 | 38.50 | 59.10 | 21.60| 70.00 | 53.90 | 46.21 |
|            | BitNet b1.58 * | 54.90 | 24.20 | 37.70 | 56.70 | 19.60| 68.80 | 55.80 | 45.39 |
|            | **Fairy±i**    | **56.65** | **24.66** | **38.69** | **59.60** | **22.20**| **69.80** | **54.06** | **46.52** |

\* reported in prior work 
† trained version  
° full precision Fairy±i  

# Introduction

The advent of Large Language Models (LLMs) has transformed artificial
intelligence, achieving remarkable performance across a wide range of
natural language
tasks . However,
this success is built upon massive model sizes, often reaching billions
or trillions of parameters, which poses serious deployment challenges
due to immense memory footprints and high computational
costs . To
democratize access to these powerful models, model compression has
become a critical research area, with *quantization* emerging as a
leading technique. Quantization methods are broadly categorized into
Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).
While PTQ  offers simplicity, its
performance often degrades sharply in extremely low-bit scenarios due to
the model's lack of adaptation to quantized representations. In
contrast, QAT integrates quantization into the training loop, allowing
models to learn robust low-bit representations and maintain performance
under aggressive compression. This advantage has motivated recent
research into QAT-based strategies tailored for LLMs.

The pursuit of extremely low-bit quantization, particularly 2-bit
quantization, has become a focal point in efforts to compress Large LLMs
for efficient deployment. Existing approaches, such as
BitNet  and its successors , have
demonstrated that it is possible to retain reasonable accuracy using
ternary quantization schemes with just 1.58 bits per weight. However,
the accuracy of any quantized model is fundamentally limited by the
following equation:
$$\textbf{Accuracy}_\text{quant}=\textbf{Accuracy}_\text{full-precision}-\textbf{Error}_\text{quant}$$
All current quantization research focuses on minimizing quantization
error on full-precision models (e.g., LLaMA), but the quantization error
can never be zero. Therefore, full-precision accuracy becomes the
**ceiling** for quantized accuracy. To date, no existing method has even
attempted to surpass this ceiling.

In this paper, we propose a fundamentally different perspective. Instead
of solely focusing on reducing quantization error, we make the first
attempt to raise the ceiling (the accuracy of the full-precision model),
while still ensuring that the resulting model can be efficiently
quantized to a 2-bit format. Our key insight is that if the
full-precision model becomes more expressive and accurate, the final
2-bit quantized model can achieve higher accuracy as well. Building on
this insight, we propose, for the first time, incorporating
complex-valued neural architectures into LLMs. The complex number
provides a richer representational space with additional phase
information, thereby enhancing the expressiveness of linear
transformations without increasing the parameter count. By
systematically extending the Transformer architecture into the complex
domain, we construct a full-precision complex-valued LLM with superior
modeling capacity.

Building upon this complex-valued foundation, we further design a novel
2-bit quantization scheme tailored for complex weights. Specifically, we
quantize each complex parameter to one of the **fourth roots of unity**
\{ ± 1, ± i\} in the complex plane. This approach---unlike
real-valued quantization---exploits the full 2-bit representational
capacity *without sacrificing symmetry or sparsity*, thereby eliminating
the trade-offs that limit real-valued schemes. The resulting model,
which we name , is perfectly storage-efficient and phase-aware by
design. We propose a quantization function that learns to project
full-precision complex weights onto the target set \{± 1, ± i\}
while preserving both magnitude and phase information. We implement this
within our complex Transformer framework and evaluate its performance
under the same storage and compute constraints as BitNet b1.58.
Experiments show that significantly improves perplexity and downstream
task accuracy, outperforming existing 2-bit baselines and approaching
the performance of full-precision FP16 models.

Our contributions can be summarized as follows:

-   We propose a new perspective on low-bit quantization: improving the
    accuracy of quantized models by raising the ceiling (the full
    precision model).

-   We design a complex-valued LLM architecture that leverages the
    representational benefits of the complex domain without increasing
    parameter storage.

-   We design a 2-bit quantization scheme that maps complex weights to
    the 4th roots of unity \{± 1, ± i\}, fully utilizing bit
    capacity while preserving key properties like symmetry and sparsity.

-   Experimental results show that our quantized model outperforms the
    ceiling of existing 2-bit quantization approaches in terms of both
    PPL and downstream understanding tasks.

# Evalation
**Table: Perplexity on WikiText2 and C4 validation sets (lower is better)**

| Size | Model             | WikiText2 | C4    | Avg   |
| :--- | :---------------- | :-------- | :---- | :---- |
| 700M | FP16 LLaMA        | -         | -     | 12.33 |
|      | BitNet b1.58*     | -         | -     | 12.87 |
|      | BitNet b1.58†     | 10.81     | 12.21 | 11.51 |
|      | Fairy ± i°        | 9.41 | 10.75  | 10.08 |
|      | Fairy ± i         | **10.46**     | **11.81** | **11.14** |
| 1.3B | FP16 LLaMA        | -         | -     | 11.25 |
|      | BitNet b1.58*     | -         | -     | 11.29 |
|      | Fairy ± i         | **9.35**      | **10.94** | **10.14** |

\* refers to the reported version in prior work  
† the trained version  
° the full precision Fairy±i  
**Table: Zero-shot Accuracy on Commonsense Reasoning Tasks (%)**
| Model Size | Model          | ARCe  | ARCc  | HS    | BQ    | OQ   | PQ    | WGe   | Avg.  |
| :--------- | :------------- | :---- | :---- | :---- | :---- | :--- | :---- | :---- | :---- |
| 700M       | FP16 LLaMA     | 54.70 | 23.00 | 37.00 | 60.00 | 20.20| 68.90 | 54.80 | 45.51 |
|            | BitNet b1.58 * | 51.80 | 21.40 | 35.10 | 58.20 | 20.00| 68.10 | 55.20 | 44.26 |
|            | BitNet b1.58 † | 51.77 | 22.44 | 35.30 | 58.50 | 20.80| 65.94 | 54.85 | 44.23 |
|            | Fairy±i°       | 55.68 | 24.06 | 37.79 | 60.46 | 20.60| 70.18 | 54.46 | 46.18 |
|            | **Fairy±i**        | **53.45** | **23.04** | **36.04** | **57.31** | **21.00**| **68.01** | **54.06** | **44.70** |
| 1.3B   | FP16 LLaMA | 56.90 | 23.50 | 38.50 | 59.10 | 21.60| 70.00 | 53.90 | 46.21 |
|            | BitNet b1.58 * | 54.90 | 24.20 | 37.70 | 56.70 | 19.60| 68.80 | 55.80 | 45.39 |
|            | **Fairy±i**    | **56.65** | **24.66** | **38.69** | **59.60** | **22.20**| **69.80** | **54.06** | **46.52** |

\* reported in prior work 
† trained version  
° full precision Fairy±i  
# how to use
## example 
```bash
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "PKU-DS-LAB/Fairy-plus-minus-i-700M"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Apply the chat template
prompt = (
    "System: You are a helpful AI assistant.\n"
    "User: How are you?\n"
    "Assistant:"
)

chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
chat_outputs = model.generate(**chat_input, max_new_tokens=50)
response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True) # Decode only the response part
print("\nAssistant Response:", response)
```
# Links

## Github
- [Fairy±i on github](https://github.com/PKULab1806/Fairy-plus-minus-i)

## ModelScope
- [Fairy±i-700M on ModelScope](https://modelscope.cn/models/PKUDSLAB1806/Fairy-plus-minus-i-700M)
- [Fairy±i-1.3B on ModelScope](https://modelscope.cn/models/PKUDSLAB1806/Fairy-plus-minus-i-1.3B)

## Paper
- [iFairy: the First 2-bit Complex LLM with All Parameters in {±1, ±i}](https://arxiv.org/abs/2508.05571)