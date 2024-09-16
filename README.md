[![LinkedIn][linkedin-shield]][linkedin-url]

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.4](https://img.shields.io/badge/torch-v2.4-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![trl 0.10.1](https://img.shields.io/badge/trl-v0.10.1-violet)](https://huggingface.co/docs/trl/index)
[![Transformers 4.44.2](https://img.shields.io/badge/transformers-v4.44.2-red)](https://huggingface.co/docs/transformers/index)
[![PEFT 0.12.0](https://img.shields.io/badge/peft-v0.12.0-lightblue)](https://huggingface.co/docs/peft/index)
[![datasets 3.0.0](https://img.shields.io/badge/datasets-v2.15.0-orange)](https://huggingface.co/docs/datasets/index)
[![bitsandbytes 0.43.3](https://img.shields.io/badge/bitsandbytes-v0.43.3-green)](https://huggingface.co/blog/hf-bitsandbytes-integration)

## :jigsaw: Objective

- Use [Open Assistant](https://huggingface.co/datasets/OpenAssistant/oasst1) to train LLM model
- Fine tune [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) on using this dataset
- Use QLoRA for fine tuning

## :open_file_folder: Files
- [**dataset.py**](dataset.py)
    - This file contains all methods to preprocess data
- [**finetune.py**](finetune.py)
    - This is the main file of this project
    - It uses function available in `dataset.py`
    - It loads, train, evaluate and save model

## What is PEFT finetuning?
PEFT (Parameter-Efficient Fine-Tuning) is a library designed to adapt large pretrained models for
different tasks without the need to fine-tune all of their parameters, which is often prohibitively
expensive. Instead, PEFT methods focus on fine-tuning only a small subset of additional parameters,
greatly reducing both computational and storage costs while still delivering performance comparable
to a fully fine-tuned model. This approach makes it feasible to train and manage large language
models (LLMs) on consumer-grade hardware.

## LoRA finetuning
LoRA was developed in early 2021 after the release of GPT-3, as Microsoft and OpenAI sought to make
large models commercially viable. They found that one-shot prompting alone was inadequate for
complex tasks, leading to a search for more efficient and cost-effective fine-tuning methods. LoRA
was designed to provide fast, efficient, and affordable fine-tuning for large language models,
enabling better domain specificity and task or user switching. This innovation aimed to address the
limitations of existing methods by reducing computational and storage costs. The research paper on
LoRA was published by Microsoft in October 2021.

## LoRA finetuning with HuggingFace
To implement LoRA finetuning with HuggingFace, you need to use the [PEFT library](https://pypi.org/project/peft/) to inject the LoRA adapters into the model and use them as the update matrices.

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True) # load the model

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1,
    target_modules=['query_key_value'] # optional, you can target specific layers using this
) # create LoRA config for the finetuning

model = get_peft_model(model, peft_config) # create a model ready for LoRA finetuning

model.print_trainable_parameters() 
# trainable params: 9,437,184 || all params: 6,931,162,432 || trainable%: 0.13615586263611604
```

## QLoRA finetuning
QLoRA extends LoRA by compressing the weight parameters of pretrained large language models (LLMs)
to 4-bit precision, reducing their memory footprint and enabling fine-tuning on a single GPU. This
compression makes it possible to run LLMs on less powerful hardware, including consumer GPUs. The
QLoRA paper introduces several innovations: 4-bit NormalFloat for better quantization results,
Double Quantization to save additional memory, and Paged Optimizers to manage memory spikes.
Experimental results show that models trained with QLoRA’s 4-bit NormalFloat outperform those
trained with LoRA and the base LLaMA 2 7B model, though the tradeoff is a slower training time due
to quantization steps. Overall, QLoRA provides significant memory savings but at the cost of
increased training time.

# QLoRA finetuning with HuggingFace
To do QLoRA finetuning with HuggingFace, you need to install both the [BitsandBytes](https://pypi.org/project/bitsandbytes/) library and the PEFT library. The BitsandBytes library takes care of the 4-bit quantization and the whole low-precision storage and high-precision compute part. The PEFT library will be used for the LoRA finetuning part.


```python
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
model_id = "EleutherAI/gpt-neox-20b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
) # setup bits and bytes config

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model) # prepares the whole model for kbit training

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config) # Now you get a model ready for QLoRA training
```

## Installation

1. Clone the repo
```
git clone https://github.com/AkashDataScience/fine_tune_phi_3_oasst1
```
2. Go inside folder
```
 cd fine_tune_phi_3_oasst1
```
3. Install dependencies
```
pip install -r requirements.txt
```

## Training

```
# Start training with:
python finetune.py

# Start training on jupyter notebook
%run finetune.py

```

## Usage
Please refer to [ERA V2 Session 29](https://github.com/AkashDataScience/ERA-V2/tree/master/Week-29)

## Contact

Akash Shah - akashmshah19@gmail.com  
Project Link - [ERA V2](https://github.com/AkashDataScience/ERA-V2/tree/master)

## Acknowledgments
This repo is developed using references listed below:
* [OpenAssistant Conversations - Democratizing Large Language Model Alignment](https://arxiv.org/pdf/2304.07327)
* [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/pdf/2404.14219)
* [Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment](https://arxiv.org/pdf/2312.12148)
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
* [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)


[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/akash-m-shah/
[Python.py]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[PyTorch.tensor]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[torch-url]: https://pytorch.org/
[HuggingFace.transformers]: https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange
[huggingface-url]: https://huggingface.co/