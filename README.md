# Factuality Enhanced Language Models for Open-Ended Text Generation (Hugging Face Version)
![License: Apache](https://img.shields.io/badge/License-Apache2.0-yellow.svg) 

This code is built on top of [Transformers](https://github.com/huggingface/transformers) v4.20.1 github repository from Huggingface. 

Purpose of this repository is to provide an easy way for researchers to replicate our work:
"[Factuality Enhanced Language Models for Open-Ended Text Generation](https://arxiv.org/pdf/2206.04624.pdf)", _Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Pascale Fung, Mohammad Shoeybi, and Bryan Catanzaro_. 

For factuality evaluation metrics used in this paper, please refer to <https://github.com/nayeon7lee/FactualityPrompt>.

## 1. Setup
All the requirements needed to run Transformers v4.20.1 codebase. Please refer to their github for setup.

## 2. Factual Decoding

```sh
MODEL_NAME=model_name
P_VAL=0.9
P_DECAY_RATE=0.9
P_LOWERBOUND=0.3
RESET_PATIENCE=5

python run_generation.py   \
    --model_type=${MODEL_NAME} \
    --model_name_or_path=${MODEL_NAME} \
    --p ${P_VAL} \
    --p_decay_rate ${P_DECAY_RATE} \
    --p_lower_cap ${P_LOWERBOUND} \
    --reset_patience ${RESET_PATIENCE}
```

You can check the implementation of factual nucleus decoding in MegatronLM codebase [here](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/text_generation/generation.py#L207)

You can check the implementation of factual nucleus decoding in BlenderBot3 [here](https://github.com/facebookresearch/ParlAI/blob/main/parlai/core/torch_generator_agent.py)


## Citation
If you use this code, please cite both of the papers listed below:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2206.04624,
  doi = {10.48550/ARXIV.2206.04624},
  url = {https://arxiv.org/abs/2206.04624},
  author = {Lee, Nayeon and Ping, Wei and Xu, Peng and Patwary, Mostofa and Fung, Pascale and Shoeybi, Mohammad and Catanzaro, Bryan},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Computers and Society (cs.CY), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Factuality Enhanced Language Models for Open-Ended Text Generation},
  publisher = {arXiv},
  year = {2022},  
  copyright = {Creative Commons Attribution 4.0 International}
}
```


```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
