'''
This file encode the LLM description for genes with different Embedding Models, we now 
'''
from transformers import GPT2Tokenizer, GPT2Model
from transformers import BertTokenizer, BertModel
import torch
import sys
import numpy as np
# loading biogpt
biogpt_tokenizer = GPT2Tokenizer.from_pretrained("microsoft/BioGPT")
biogpt_model = GPT2Model.from_pretrained("microsoft/BioGPT")
# loading biobert
biobert_tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1")
biobert_model = BertModel.from_pretrained("dmis-lab/biobert-large-cased-v1.1")

def encoding_gene_description(filename, model, tokenizer, gene_file, fileoutput):
    '''
    for biogpt and biobert
    '''
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gene_embed_list = []
    gene_names = np.load(gene_file,allow_pickle=True)
    fileobj = open(filename, "r")
    while True:
        line = fileobj.readline()
        if line == "":
            break
        line = line.strip("\n").split("\t")
        gene, text = line
        text = text.replace(":", " ").replace("_", " ").replace(";", ".").replace("/", " ").replace("(", " ").replace(")", " ").replace("+", " ").replace("'", " ").replace("’", " ")
        ###prompt embedding
        #tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        last_token_embedding = last_hidden_state[:, -1, :]
        new_emb = last_token_embedding.numpy()[0]
        gene_embed_list.append(new_emb)
    torch.save({"gene_names": gene_names, "gene_features": torch.tensor(gene_embed_list)}, fileoutput)
    return gene_embed_list


if __name__ == "__main__":
    gene_description_file = sys.argv[1]
    choice_model = sys.argv[2]
    gene_file = sys.argv[3]
    fileoutput_path = sys.argv[4]
    while True:
        if choice_model == "biogpt":
            model = biogpt_model
            tokenizer = biogpt_tokenizer
            encoding_gene_description(gene_description_file, model, tokenizer, gene_file, fileoutput_path)
            break
        elif choice_model == "biobert":
            model = biobert_model
            tokenizer = biobert_tokenizer
            encoding_gene_description(gene_description_file, model, tokenizer, gene_file, fileoutput_path)
            break
        else:
            print("Sorry, we don't support this type of encoding method")
            break
        
