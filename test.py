import os
import torch
import numpy as np
import argparse
import pandas as pd
from scipy.stats import pearsonr


def list_sorted_subfolders(folder_path):
    """List all subfolders in a directory and sort them by name."""
    subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    return [os.path.join(folder_path, i) for i in sorted(subfolders)]

def patient_results(path):
    """Load MAE, MSE, and correlation results from a given path."""
    mae = torch.load(os.path.join(path, 'MAE'))
    mse = torch.load(os.path.join(path, 'MSE'))
    cor = torch.load(os.path.join(path, 'cor'))  # shape: (250,)
    return mae, mse, cor

def load_all_label_data(label_dir):
    data_list = []
    for file in sorted(os.listdir(label_dir)):
        if file.endswith('.npy'):
            data = np.load(os.path.join(label_dir, file))  # shape: (n, 250)
            data_list.append(data)
    return np.concatenate(data_list, axis=0)  # shape: (N_total, 250)

def get_top_heg_indices(data, top_k=50):
    gene_means = np.mean(data, axis=0)
    return np.argsort(gene_means)[-top_k:][::-1]


def get_top_hvg_indices(data, top_k=50):
    data = data / np.sum(data, axis=1, keepdims=True)  # normalize each spot
    gene_vars = np.var(data, axis=0)
    return np.argsort(gene_vars)[-top_k:][::-1]

def metrics(dataset, result_dir, label_dir, gene_names_file, output_csv, top_k=50):
    """Calculate and save evaluation metrics."""

    # Get all subfolders in the output directory
    res_subfolders = list_sorted_subfolders(result_dir)

    mae_list, mse_list, pcc_all = [], [], []

    for path in res_subfolders:
        mae, mse, cor = patient_results(path)
        mae_list.append(mae)
        mse_list.append(mse)
        pcc_all.append(cor.numpy() if torch.is_tensor(cor) else cor)

    pcc_all = np.stack(pcc_all)  # shape: (num_patients, 250)

    # HPG: highly predictive genes
    rank_sum = np.zeros(250)
    for cor in pcc_all:
        ranks = np.argsort(cor)
        for i, idx in enumerate(ranks):
            rank_sum[idx] += i
    hpg_indices = np.argsort(rank_sum)[-top_k:]

    # HEG: highly expressed genes
    all_label_data = load_all_label_data(label_dir)
    heg_indices = get_top_heg_indices(all_label_data, top_k=top_k)

    # HVG: highly variable genes
    hvg_indices = get_top_hvg_indices(all_label_data, top_k=top_k)

    # Compute means and stds
    mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)
    mse_mean, mse_std = np.mean(mse_list), np.std(mse_list)

    pcc_all_mean = np.mean(np.nanmean(pcc_all, axis=1))
    pcc_all_std = np.std(np.nanmean(pcc_all, axis=1))

    pcc_hpg_mean = np.mean(np.nanmean(pcc_all[:, hpg_indices], axis=1))
    pcc_hpg_std = np.std(np.nanmean(pcc_all[:, hpg_indices], axis=1))

    pcc_heg_mean = np.mean(np.nanmean(pcc_all[:, heg_indices], axis=1))
    pcc_heg_std = np.std(np.nanmean(pcc_all[:, heg_indices], axis=1))

    pcc_hvg_mean = np.mean(np.nanmean(pcc_all[:, hvg_indices], axis=1))
    pcc_hvg_std = np.std(np.nanmean(pcc_all[:, hvg_indices], axis=1))

    
    df = pd.DataFrame([{
        "mae_mean": round(mae_mean, 3),
        "mae_std": round(mae_std, 2),
        "mse_mean": round(mse_mean, 3),
        "mse_std": round(mse_std, 2),
        "pcc_all_mean": round(pcc_all_mean, 3),
        "pcc_all_std": round(pcc_all_std, 2),
        "pcc_hpg_mean": round(pcc_hpg_mean, 3),
        "pcc_hpg_std": round(pcc_hpg_std, 2),
        "pcc_heg_mean": round(pcc_heg_mean, 3),
        "pcc_heg_std": round(pcc_heg_std, 2),
        "pcc_hvg_mean": round(pcc_hvg_mean, 3),
        "pcc_hvg_std": round(pcc_hvg_std, 2),
    }])
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV saved to: {output_csv}")

    print(f'dataset: {dataset}')
    print("MAE mean: %.3f, std: %.2f" % (mae_mean, mae_std))
    print("MSE mean: %.3f, std: %.2f" % (mse_mean, mse_std))
    print("PCC ALL mean: %.3f, std: %.2f" % (pcc_all_mean, pcc_all_std))
    print("PCC HPG mean: %.3f, std: %.2f" % (pcc_hpg_mean, pcc_hpg_std))
    print("PCC HEG mean: %.3f, std: %.2f" % (pcc_heg_mean, pcc_heg_std))
    print("PCC HVG mean: %.3f, std: %.2f" % (pcc_hvg_mean, pcc_hvg_std))


def eval_model(dataset, gpu=3, model_name="test", output_dir=""):
    """Evaluate the model on the given dataset."""
    FOLD_DICT = {'her2st': 8, 'skin': 4, 'stnet': 8}

    for fold in range(FOLD_DICT[dataset]):
        # the directory to save ckpt, for example, 2025-11-17 is the date you start training.
        cur_dir = os.path.join('logs', '2025-11-18', f'{fold}-{model_name}-{dataset}-2021-10')   
        ckpt_files = [f for f in os.listdir(cur_dir) if f.endswith('.ckpt')]
        if not ckpt_files:
            print(f"Error: No .ckpt files found in {cur_dir}!")
            continue
        for f in ckpt_files:
            cmd = f"python main.py --config_name skin/skin --mode test --fold {fold} --model_path " + \
                    f"{os.path.join(cur_dir, f)}  --gpu {gpu}"                                      ####### config
            print(cmd)
            os.system(cmd)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate model by given parameters')
    parser.add_argument('--dataset', type=str, required=True, choices=['her2st', 'skin', 'stnet'], help='Dataset name')
    parser.add_argument('--gpu', type=int, default=1, help='GPU ID')
    parser.add_argument('--model_name', type=str, help='Model name')
    args = parser.parse_args()

    # the directory to save test results, for example, 2025-11-18 is the date you start testing
    output_dir = os.path.join('final', args.model_name, args.dataset, '2025-11-19')  

    # Evaluate the model
    eval_model(dataset=args.dataset, gpu=args.gpu, model_name=args.model_name, output_dir=output_dir)
    
    # ------------------------- metric calculation ------------------------- #
    # the processed ground truth, they will be save during training, you could also found them in our hugging face
    label_dir = f"/home/wzhang/data/skin/label"

    # the npy file that store the gene name of interest
    gene_name_file = f"/home/wzhang/data_uni/skin/genes_skin.npy"

    # the csv file that save all results
    output_csv = f"{args.dataset}_{args.model_name}.csv" 

    # Calculate and save metrics
    metrics(dataset=args.dataset,
            result_dir=output_dir,
            label_dir=label_dir,
            gene_names_file=gene_name_file,
            output_csv=output_csv,
            top_k=50)
