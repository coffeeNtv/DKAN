## Gene Feature Extraction

Our gene feature extraction have multiple steps:

1. Gene summary retrieval from external public gene databases, such as NCBI and GO-Term

2. Gene summary refinement by LLM, such as GPT-4o, DeepSeek-V3, DeepSeek-R1, and Llama-2

3. Gene embedding, such as Conch, Plip, BioBERT and BioGPT 

   

## Environment

``````
transformers==4.49.0
torch==2.6.0
numpy==1.26.4
requests==2.32.4
openai==1.63.0
huggingface-hub==0.29.1
biopython==1.85
goatools==1.4.12
``````



## Datasets

We use three gene groups: stnet, her2st, and skin.

Their gene names can be found at ./evaluation/* or [Huggingface](https://huggingface.co/datasets/wzhang472/dkan/tree/main):

- `genes_her2st.npy`  

- `genes_skin.npy`  

- `genes_stnet.npy`  

  

## Configuration

Configuration files for the three gene groups are located in:  

`./config/her2st.json`, `./config/skin.json`, and `./config/stnet.json`.  

Key configurable parameters:  
- `gene_summary_path`: Directory to store gene information extracted from databases.  

- `gene_database`: Public database for information extraction (supports `NCBI` and `GO-Term`).  

- `gene_description_filename`: Output file for LLM-generated gene summaries.  

- `LLM_type`: LLM model for summary generation (supports `gpt-4o`, `deepseek_v3`, `deepseek_r`, `llama2`).  

- `gene_encoding_type`: Embedding tool for processing summaries (supports `biogpt` and `biobert`).  

- `gene_describe_encoding_file`: Output file for embedding results.  

  

## Command

Please refer to below commands for different stages:

```
# extract gene information
python main.py --mode=info_extract

# refine gene summaries via LLM
python main.py --mode=llm_describe

# embed gene summaries
python main.py --mode=embedding  

# run all steps sequentially
python main.py --mode=all  
```

- Note that in our paper, we use NCBI as external gene database, GPT4-o as gene summary refiner, Biobert as text encoder.
- For text encoder with conch and plip, please refer to "text_encoder_conch.py" and "text_encoder_plip.py".
- For extracting gene information from GO-Term database, please download necessary files from [Hugging face](https://huggingface.co/datasets/wzhang472/dkan/tree/main/Go-term) before running the script.

## Reproducibility

- All our prompts we used in our paper are provided in ./Gene_Embedding/prompt/*

- All our raw data retrieved from NCBI database is provided in ./Gene_Embedding/ncbi_raw_information/*

- All our gene summaries we used in our paper are provided in ./Gene_Embedding/gene_features/*.pt

- All our gene features we used in our paper are provided in  ./Gene_Embedding/gene_summaries/*.txt

- All code for gene retrieval, summary refinement, text embedding for different LLMs and text encoders are provided in ./Gene_Embedding/*.py