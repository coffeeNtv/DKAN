# Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics - AAAI 2026 Oral

<p align="center">
  <a href="https://arxiv.org/abs/2511.17685" target="_blank"><img src="https://img.shields.io/badge/arXiv-2511.17685-red"></a>
  <a href="https://huggingface.co/datasets/wzhang472/dkan" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a>
  <a href="https://huggingface.co/papers/2505.21497" target="_blank"><img 
</p>

Thank you for your attention. This is the official codebase for DKAN. 

<img src="./figure/framework.png" title="DKAN"/>

## Environment

``````
python=3.11.11
pytorch-lightning==2.4.0
torch==2.4.1+cu121
torchvision==0.19.1+cu121
``````

More details of our environments are provided in dkan.yaml.

## Data Preparation

All data used in this study is available at [Huggingface](https://huggingface.co/datasets/wzhang472/dkan/tree/main). Please feel free to reach me if you have any trouble in downloading the data. After downloading all data, you will obtain below data directory:

```
	./data/
	├── her2st
    │ 		├── gt_features_224
    │ 		├── n_features_5_224
    │ 		├── ST-cnts
    │ 		├── ST-imgs
    │ 		├── ST-spotfiles
    │ 		├── ST-text
    				└── her2st_ncbi_biobert.pt
    │ 		└── ...
	├── skin
    │ 		└── ...
	└── stnet
    		└── ...
```

## Image Feature Extraction

If you wish to use other foundation model alternatives, such as Conch, take HER2+ dataset as an example, the command for feature extraction can be found below:

```
python extract_features.py --config her2st/her2st --test_mode internal --extract_mode target --encoder conch
python extract_features.py --config her2st/her2st --test_mode internal --extract_mode neighbor --encoder conch
```



## Gene Feature Extraction

Our gene feature extraction have multiple steps:

1. Gene summary retrieval from external public gene databases, such as NCBI and GO-Term
2. Gene summary refinement by LLM, such as GPT-4o, DeepSeek-V3, DeepSeek-R1, and Llama-2
3. Gene embedding, such as Conch, Plip, BioBERT and BioGPT 

To provide more gene representation alternatives and ensure reproducibility, more details about using other free accessed LLMs can be found at [gene_embedding_notes](./Gene_Embedding/gene_embedding.md). All codes, gene summaries and gene features are also released in this repository.



## Training and Evaluation

Before starting training, please make sure the directories in ./config/\*/\*.yaml are updated accordingly to your work environments.

```
# Training
python main.py --config_name skin/skin --mode cv --gpu 0
python main.py --config_name her2st/her2st --mode cv --gpu 0
python main.py --config_name stnet/stnet --mode cv --gpu 0

# Evaluation
python test.py --dataset skin --model_name your_name --gpu 0
```



## Reproducing Reported Results

Please see ./evaluation/metric.ipynb to reproduce the results reported in our paper.



## TODO

- [x] Release our ST data on [Hugging Face]()

- [x] Release our gene summary and embeddings

- [x] Release our prompts we used in this study

- [x] Release code for training and evaluation

- [x] Release code for reproducing our results 

- [x] Release code for image features with foundation models

- [x] Release code for gene summary from external gene databases 

- [x] Release code for gene summary refinement via LLMs 

- [x] Release code for gene feature extraction with text encoders

- [x] Release paper on [Arxiv](https://arxiv.org/abs/2511.17685)

  

## Acknowledgement

Our study builds upon previous studies: [TRIPLEX](https://github.com/NEXGEM/TRIPLEX), [HisToGene](https://github.com/maxpmx/HisToGene). Many thanks for their opensource and the contributions to this community.



## Citation 

Please kindly cite our paper if our work could be helpful for your study. Thank you.

```
@misc{zhang2025dkan,
      title={Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics}, 
      author={Wei Zhang and Jiajun Chu and Xinci Liu and Chen Tong and Xinyue Li},
      year={2025},
      eprint={2511.17685},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2511.17685}, 
}
```

