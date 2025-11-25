"""
This file extracted the Gene Summary from public databases. Curretly we have NCBI database.
"""
import numpy as np
from Bio import Entrez
import json
import sys
import os
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools import obo_parser


def run_ncbi(gene_name):
    """
    This function input the gene name and out put the summary of gene
    """
    gene_dict = {"gene_name": gene_name}
    species = "Homo sapiens"  #give species
    handle = Entrez.esearch(db="gene", term=f"{gene_name}[Gene Name] AND {species}[Organism]")
    record = Entrez.read(handle)
    try:
        id_list = record["IdList"]
        gene_id = record["IdList"][0]  # select the first gene id that matches the gene name
        gene_dict["id_list"] = id_list
        gene_dict["gene_id"] = gene_id
    except:
        print("no_" + gene_name)
        return gene_dict
    #通过找到的gene_id去得到gene summary
    handle = Entrez.efetch(db="gene", id=[gene_id], rettype="xml", retmode="xml")
    records = Entrez.read(handle)
    for i in ["Entrezgene_gene", "Entrezgene_location", "Entrezgene_summary", "Entrezgene_commentaries"]:
        if i in records[0]:
            gene_dict[i] = records[0][i]
    return gene_dict

def run_go(gene_filename):
    go_obo = obo_parser.GODag("go.obo")
    gene_dict = {}
    with open("goa_human.gaf", "r") as f:
        lines = [line for line in f if not line.startswith("!")]  # 跳过注释行
    gene_list = np.load(gene_filename, allow_pickle=True)
    ###read the go term data for target genes
    for line in lines:
        line = line.strip("\n").split("\t")
        protein = line[1]
        gene_name = line[2]
        Qualifier = line[3]
        go_id = line[4]
        go_term = go_obo.query_term(go_id)
        go_namespace = go_term.namespace
        go_name = go_term.name
        if gene_name not in gene_list:
            continue
        if gene_name not in gene_dict:
            gene_dict[gene_name] = []
        now = (protein, Qualifier, go_name, go_namespace)
        if now not in gene_dict[gene_name]:
            gene_dict[gene_name].append(now)
    ### generate file for each gene in gene_filepath
    gene_summary_dict = {}
    for gene in gene_dict:
        info_lis = []
        for value in gene_dict[gene]:
            protein, fuct, fuct_name, fuct_id = value
            sentence = "The protein encoded by " + protein + " " + fuct.replace("_", " ") + " " + fuct_name
            info_lis.append(sentence)
        if len(info_lis)>10:
            info_lis = info_lis[:10]
        summary = ". ".join(info_lis)
        gene_summary_dict[gene] = {"gene_name": gene, "Entrezgene_summary": summary}
    return gene_summary_dict


def running_all_genes(filepath, filename, database_type):
    """
    this method input gene name file, named filename, which is .npy and contains all gene name
    this method output files that includes summary dict for each gene, each gene summary is stored in a separate file at filepath/gene_name.
    """
    fileobj = open(filename, "r")
    data = np.load(filename, allow_pickle=True)
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    if database_type == "go":
        gene_summary_dict = run_go(filename)
    for gene_name in data:
        try:
            if database_type == "ncbi":
                gene_dict = run_ncbi(gene_name)
            elif database_type == "go":
                if gene_name in gene_summary_dict:
                    gene_dict = gene_summary_dict[gene_name]
                else:
                    gene_dict = {"gene_name": gene_name}
            else:
                print("Sorry, We do not support this database now")
                gene_dict = {"gene_name": gene_name}
        except:
            gene_dict = {}
        out = filepath + gene_name
        with open(out, "w") as k:
            json.dump(gene_dict, k, ensure_ascii=False)

if __name__ == "__main__":
    gene_filename = sys.argv[1]
    output_path = sys.argv[2]
    database_type = sys.argv[3]
    running_all_genes(output_path, gene_filename, database_type)
