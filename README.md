[![LinkedIn][linkedin-shield]][linkedin-url]

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.4](https://img.shields.io/badge/torch-v2.4-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![trl 0.10.1](https://img.shields.io/badge/trl-v0.10.1-violet)](https://huggingface.co/docs/trl/index)
[![Transformers 4.44.2](https://img.shields.io/badge/transformers-v4.44.2-red)](https://huggingface.co/docs/transformers/index)
[![PEFT 0.12.0](https://img.shields.io/badge/peft-v0.12.0-lightblue)](https://huggingface.co/docs/peft/index)
[![datasets 3.0.0](https://img.shields.io/badge/datasets-v2.15.0-orange)](https://huggingface.co/docs/datasets/index)

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