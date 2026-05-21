# open files in paper_results directory, following the naming {model_name}_{dataset_name}_{number_seed}_test_results.json, get the data, compute mean and std
import os
import json
import numpy as np  

# models = ['gru', 'grugcn', 'mlp', 'transformer']
models = ['transformer_spatial']    
datasets = ['peakweather', 'ostrinia']
seeds = [42, 43, 44, 45, 46]
metrics = ['test_mae', 'test_mre', 'test_mse']
use_node_embeddings = [False, True]

results = {}
for model in models:
    results[model] = {}
    for dataset in datasets:
        results[model][dataset] = {}
        for use_embd in use_node_embeddings:
            all_metrics = {metric: [] for metric in metrics}
            for seed in seeds:
                file_name = f'paper_results/{model}_{dataset}_nodes_embd_{use_embd}/{seed}/test_results.json'
                if os.path.exists(file_name):
                    with open(file_name, 'r') as f:
                        data = json.load(f)
                        for metric in metrics:
                            if metric in data:
                                all_metrics[metric].append(data[metric])
                            else:
                                print(f"Warning: {metric} not found in {file_name}")
                else:
                    print(f"Warning: File {file_name} does not exist.")
            
            results[model][dataset][f'nodes_embd_{use_embd}'] = {}
            for metric in metrics:
                if all_metrics[metric]:
                    mean_val = np.mean(all_metrics[metric])
                    std_val = np.std(all_metrics[metric])
                    results[model][dataset][f'nodes_embd_{use_embd}'][metric] = {
                        'mean': mean_val,
                        'std': std_val
                    }
                else:
                    results[model][dataset][f'nodes_embd_{use_embd}'][metric] = {
                        'mean': None,
                        'std': None
                    }

print(json.dumps(results, indent=4))

# Save results to a JSON file
with open('final_results_summary.json', 'w') as f:
    json.dump(results, f, indent=4)

# make a latex table with the results
def format_metric(metric_dict):
    if metric_dict['mean'] is None or metric_dict['std'] is None:
        return "N/A"
    return f"{metric_dict['mean']:.4f} ± {metric_dict['std']:.4f}"
latex_table = "\\begin{tabular}{l l l " + " ".join(["c" for _ in metrics]) + "}\n"
latex_table += "Model & Dataset & Node Embd & " + " & ".join(metrics) + " \\\\ \n"
latex_table += "\\hline\n"
for model in models:
    for dataset in datasets:
        for use_embd in use_node_embeddings:
            embd_key = f'nodes_embd_{use_embd}'
            if embd_key in results[model][dataset]:
                latex_table += f"{model} & {dataset} & {use_embd} & " + " & ".join([format_metric(results[model][dataset][embd_key][metric]) for metric in metrics]) + " \\\\ \n"
latex_table += "\\end{tabular}\n"

with open('final_results_table.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table saved to final_results_table.tex")