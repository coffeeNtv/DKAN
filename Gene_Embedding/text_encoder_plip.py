from plip import PLIP
import numpy as np
import os
import torch

plip = PLIP('vinid/plip')
descriptions = []
path = '/home/wzhang/st/text/her2st_ncbi_result.txt'
with open(path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if not line: 
            continue

        parts = line.split('\t')
        if len(parts) < 2:  
            continue

        description = parts[1]  
        descriptions.append(description)

text_emb = plip.encode_text(descriptions, batch_size=1)
gene_names = np.load("/home/wzhang/st/data/her2st/genes_her2st.npy",allow_pickle=True)
torch.save({"gene_names": gene_names, "gene_features": torch.tensor(text_emb)}, "her2st_ncbi_plip.pt")
