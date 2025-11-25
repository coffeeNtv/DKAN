import os 
import argparse
import json

def main(args):
    """
    main function 
    """
    config = args.config_name
    mode = args.mode
    ###load the config path
    with open(config, "r") as f:
        config_data = json.load(f)
    print(config_data)
    
    ##extract datapath from config data
    gene_name = config_data.get("gene_name", "")
    gene_file = config_data.get("gene_file", "")
    gene_summary_path = config_data.get("gene_summary_path", "")
    gene_database = config_data.get("gene_database", "")
    gene_description_filename = config_data.get("gene_description_filename", "")
    LLM_type = config_data.get("LLM_type", "")
    gene_encoding_type = config_data.get("gene_encoding_type", "")
    gene_describe_encoding_file = config_data.get("gene_describe_encoding_file", "")
    if mode == "all":
        os.system("python extract_gen_info.py " + gene_file + " " + gene_summary_path + " " + gene_database)
        os.system("python generating_llm_description.py " + gene_file + " " + gene_summary_path + " " + gene_description_filename + " " + LLM_type + " " + gene_name)
        os.system("python generating_descibe_encoding.py " + gene_description_filename + " " + gene_encoding_type + " " + gene_file + " " + gene_describe_encoding_file)
    elif mode == "info_extract": 
        os.system("python extract_gen_info.py " + gene_file + " " + gene_summary_path + " " + gene_database)
    elif mode == "llm_describe": 
        os.system("python generating_llm_description.py " + gene_file + " " + gene_summary_path + " " + gene_description_filename + " " + LLM_type  + " " + gene_name)
    elif mode == "embedding":
        os.system("python generating_descibe_encoding.py " + gene_description_filename + " " + gene_encoding_type + " " + gene_file + " " + gene_describe_encoding_file)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='./config/her2st.json', help='Path to the configuration file.')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'info_extract', 'llm_describe', 'embedding'], help='Execution mode.')
    args, unknown = parser.parse_known_args()

    main(args)