import numpy as np
import os
import torch

from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize, get_tokenizer
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="checkpoints/conch/pytorch_model.bin")

descriptions = []
path = '/home/wzhang/st/text/herst_ncbi.txt'
with open(path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if not line:  # skip blank row
            continue

        parts = line.split('\t')
        if len(parts) < 2:  # gene name and descriptions
            continue

        description = parts[1]  
        descriptions.append(description)

tokenizer = get_tokenizer() # load tokenizer

text_tokens = tokenize(texts=descriptions, tokenizer=tokenizer) # tokenize the text
text_embs = model.encode_text(text_tokens)
gene_names = np.load("/home/wzhang/st/data/her2st/genes_her2st.npy",allow_pickle=True)

torch.save({"gene_names": gene_names, "gene_features": torch.tensor(text_embs)}, "her2st_ncbi_conch.pt")

