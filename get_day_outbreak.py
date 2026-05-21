import numpy as np
import matplotlib.pyplot as plt




def get_curves(npz_paths_list):

    
    # Load data from all configurations
    y_pred_means = []
    y_true = None
    
    for npz_paths in npz_paths_list:
        y_preds_all = []
        for npz_path in npz_paths:
            data = np.load(npz_path)
            if y_true is None:
                y_true = data['y_true']
            y_preds_all.append(data['y_pred'])
        
        # Compute mean across seeds for this configuration
        y_preds_stacked = np.stack(y_preds_all, axis=0)
        y_pred_mean = np.mean(y_preds_stacked, axis=0)
        y_pred_means.append(y_pred_mean)

    # Clip real values to be non-negative
    y_true = np.clip(y_true, a_min=0, a_max=None)
    
    
    return {"y_true" : y_true, 
            "y_pred_all" : y_preds_stacked}
    

# create function which plots true vs predicted for a single node
def plot_true_vs_predicted(node_id, preds, seeds, day_real_pred, day_pred_pred, i):

    day_real_pred = day_real_pred[i]
    day_pred_pred = day_pred_pred[:, i]

    delay = 175 if dataset == 'peakweather' else 0

    # only plot from time step 150 onwards
    y_true_node = preds['y_true'][delay:250, :, node_id]
    y_pred_node = preds['y_pred_all'][:, delay:250, :, node_id]

    plt.figure(figsize=(10, 6))
    plt.plot(y_true_node, label='True', color='black', linewidth=2)

    for seed in range(y_pred_node.shape[0]):
        y_pred_node_seed = np.clip(y_pred_node[seed], a_min=0, a_max=None)
        plt.plot(y_pred_node_seed, label=f'Predicted (seed {seeds[seed]})', alpha=0.5)
        # mark the predicted day of outbreak
        plt.axvline(x=day_pred_pred[seed]-delay, color='red', linestyle='--', alpha=0.5)

    # mark the true day of outbreak
    plt.axvline(x=day_real_pred-delay, color='green', linestyle='--', label='True Outbreak Day')

    plt.title(f'Node {node_id}: True vs Predicted Outbreaks')
    plt.xlabel('Days')
    plt.ylabel('Outbreak Value')
    plt.legend()
    plt.grid()
    plt.savefig("./test")

def plot_true_vs_predicted_ensemable(node_id, preds, day_real_pred, day_pred_pred, i):
    
    day_real_pred = day_real_pred[i]
    day_pred_pred = day_pred_pred[i]

    delay = 175 if dataset == 'peakweather' else 0

    # only plot from time step 150 onwards
    y_true_node = preds['y_true'][delay:250, :, node_id]
    y_pred_node = np.mean(preds['y_pred_all'], axis=0)[delay:250, :, node_id]

    plt.figure(figsize=(10, 6))
    plt.plot(y_true_node, label='True', color='black', linewidth=2)

    y_pred_node_seed = np.clip(y_pred_node, a_min=0, a_max=None)
    # mean of y_pred_node_seed
    plt.plot(y_pred_node_seed, label=f'Predicted (ensemable)', alpha=0.8)
    # mark the predicted day of outbreak
    plt.axvline(x=day_pred_pred-delay, color='red', linestyle='--', alpha=0.8)

    # mark the true day of outbreak
    plt.axvline(x=day_real_pred-delay, color='green', linestyle='--', label='True Outbreak Day')

    plt.title(f'Node {node_id}: True vs Predicted Outbreaks (Ensemable)')
    plt.xlabel('Days')
    plt.ylabel('Outbreak Value')
    plt.legend()
    plt.grid()
    plt.savefig("./test_ensemable")


def get_outbreak_from_predictions(y_pred_node_seed, strategy='max_diff', threshold=10, dataset='peakweather'):

    # available strategies: ['max_diff', 'first_of_two_maxes', 'max_after_day_150', 'third_of_three_maxes',
    #                        'first_of_two_maxes_after_day_150', 'first_over_threshold', 'first_fixed_max_th']

    if strategy == 'max_diff':
        daily_diffs = np.diff(y_pred_node_seed, axis=0)
        day_outbreak = np.argmax(daily_diffs)

    elif strategy == 'first_of_two_maxes':
        daily_diffs = np.diff(y_pred_node_seed, axis=0)
        # indices of the two largest values
        top2_idx = np.argsort(daily_diffs, axis=0)[-2:]
        # choose the one with lower index (earlier day)
        day_outbreak = np.min(top2_idx)

    elif strategy == 'max_after_day_th':
        daily_diffs = np.diff(y_pred_node_seed, axis=0)
        # consider only days after day th
        daily_diffs_after_th = daily_diffs[threshold:]
        if len(daily_diffs_after_th) == 0:
            day_outbreak = -1  # no data after day th
        else:
            day_outbreak = np.argmax(daily_diffs_after_th) + threshold  # adjust index back to original

    elif strategy == 'third_of_three_maxes':
        daily_diffs = np.diff(y_pred_node_seed, axis=0)
        # indices of the three largest values
        top3_idx = np.argsort(daily_diffs, axis=0)[-3:]
        if len(top3_idx) < 3:
            day_outbreak = -1  # not enough data
        else:
            day_outbreak = np.sort(top3_idx)[0]  # choose the earliest day among the three

    elif strategy == 'first_of_two_maxes_after_day_150':
        daily_diffs = np.diff(y_pred_node_seed, axis=0)
        # consider only days after day 150
        daily_diffs_after_150 = daily_diffs[150:]
        top2_idx = np.argsort(daily_diffs_after_150, axis=0)[-2:]
        if len(top2_idx) < 2:
            day_outbreak = -1  # not enough data
        else:
            day_outbreak = np.min(top2_idx) + 150  # adjust index back to original

    elif strategy == 'first_over_threshold':
        daily_diffs = np.diff(y_pred_node_seed, axis=0)
        over_threshold_indices = np.where(daily_diffs > threshold)[0]
        if len(over_threshold_indices) > 0:
            day_outbreak = over_threshold_indices[0]
        else:
            day_outbreak = -1  # no outbreak detected

    elif strategy == 'first_fixed_max_th':
        over_threshold_indices = np.where(y_pred_node_seed > threshold)[0]
        if len(over_threshold_indices) > 0:
            day_outbreak = over_threshold_indices[0]
        else:
            day_outbreak = -1  # no outbreak detected

    elif strategy == 'dd50':
        if dataset == 'ostrinia':
            # check max value
            max_value = np.max(y_pred_node_seed)
            dd50_value = max_value / 2
            over_dd50_indices = np.where(y_pred_node_seed >= dd50_value)[0]
            if len(over_dd50_indices) > 0:
                day_outbreak = over_dd50_indices[0]
            else:
                day_outbreak = -1  # no outbreak detected
        else:
            max_value = np.sum(y_pred_node_seed)
            dd50_value = max_value / 2
            cumulative_sum = np.cumsum(y_pred_node_seed)
            over_dd50_indices = np.where(cumulative_sum >= dd50_value)[0]
            if len(over_dd50_indices) > 0:
                day_outbreak = over_dd50_indices[0]
            else:
                day_outbreak = -1  # no outbreak detected
    elif strategy == 'dd10':
        if dataset == 'ostrinia':
            # check max value
            max_value = np.max(y_pred_node_seed)
            dd10_value = max_value / 10
            over_dd10_indices = np.where(y_pred_node_seed >= dd10_value)[0]
            if len(over_dd10_indices) > 0:
                day_outbreak = over_dd10_indices[0]
            else:
                day_outbreak = -1  # no outbreak detected
        else:
            max_value = np.sum(y_pred_node_seed)
            dd10_value = max_value / 10
            cumulative_sum = np.cumsum(y_pred_node_seed)
            over_dd10_indices = np.where(cumulative_sum >= dd10_value)[0]
            if len(over_dd10_indices) > 0:
                day_outbreak = over_dd10_indices[0]
            else:
                day_outbreak = -1  # no outbreak detected

    elif strategy == 'dd25':
        if dataset == 'ostrinia':
            max_value = np.max(y_pred_node_seed)
            dd25_value = max_value / 4
            over_dd25_indices = np.where(y_pred_node_seed >= dd25_value)[0]
            if len(over_dd25_indices) > 0:
                day_outbreak = over_dd25_indices[0]
            else:
                day_outbreak = -1  # no outbreak detected
        else:
            max_value = np.sum(y_pred_node_seed)
            dd25_value = max_value / 4
            cumulative_sum = np.cumsum(y_pred_node_seed)
            over_dd25_indices = np.where(cumulative_sum >= dd25_value)[0]
            if len(over_dd25_indices) > 0:
                day_outbreak = over_dd25_indices[0]
            else:
                day_outbreak = -1  # no outbreak detected

    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")

    return day_outbreak

import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(data, title="Distribution", bins=30, kde=True):
    """
    Plots a histogram and optional KDE for a given dataset.
    
    Parameters
    ----------
    data : array-like
        Input numerical data.
    title : str, optional
        Title of the plot.
    bins : int, optional
        Number of bins for the histogram.
    kde : bool, optional
        Whether to plot a Kernel Density Estimate (KDE).
    """
    plt.figure(figsize=(7, 4))
    sns.histplot(data, bins=bins, kde=kde, color="steelblue", edgecolor="black")
    plt.title(title, fontsize=13)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("./test_distribution.png")


def filter_nodes(dataset, preds, th):

    if dataset == 'peakweather':

        nodes_id = np.arange(0, 160)  # Assuming 160 nodes for example

        # filter out nodes without outbreaks
        outbreak_nodes = []
        for node_id in nodes_id:
            y_true_node = preds['y_true'][:, :, node_id]
            if np.max(y_true_node) > th:
                outbreak_nodes.append(node_id)
        # print(f'Number of outbreak nodes: {len(outbreak_nodes)}/{len(nodes_id)}')

        #filter out node 130, 149, 19, 108
        if 130 in outbreak_nodes:
            outbreak_nodes.remove(130)
        if 149 in outbreak_nodes:
            outbreak_nodes.remove(149)
        if 19 in outbreak_nodes:
            outbreak_nodes.remove(19)
        if 108 in outbreak_nodes:
            outbreak_nodes.remove(108)

        # Compute day of outbreak for each node as the max diference between following days
        day_outbreaks = []
        for node_id in outbreak_nodes:
            y_true_node = preds['y_true'][:, :, node_id]
            daily_diffs = np.diff(y_true_node, axis=0)
            day_outbreak = np.argmax(daily_diffs, axis=0)  # Day of max increase
            day_outbreaks.append(day_outbreak)

        return outbreak_nodes, day_outbreaks
    
    if dataset == 'ostrinia':

        nodes_id = np.arange(0, 13)  # Assuming 13 nodes for example

        # filter out nodes without outbreaks
        outbreak_nodes = []
        for node_id in nodes_id:
            y_true_node = preds['y_true'][:, :, node_id]
            if np.max(y_true_node) > th:
                outbreak_nodes.append(node_id)
        # print(f'Number of outbreak nodes: {len(outbreak_nodes)}/{len(nodes_id)}')

        # Compute day of outbreak for each node as the max diference between following days
        day_outbreaks = []
        for node_id in outbreak_nodes:
            y_true_node = preds['y_true'][:, :, node_id]
            daily_diffs = np.diff(y_true_node, axis=0)
            day_outbreak = np.argmax(daily_diffs, axis=0)  # Day of max increase
            day_outbreaks.append(day_outbreak)

        return outbreak_nodes, day_outbreaks
    
def get_days_outbreaks(y_pred_node, outbreak_nodes, seeds, strategy='max_diff', threshold=10):

        # do the same for predicted values.
        day_outbreaks_pred_per_seed = np.zeros((len(seeds), len(outbreak_nodes)), dtype=int)
        for i, node_id in enumerate(outbreak_nodes):
            for seed in range(y_pred_node.shape[0]):
                y_pred_node_seed = np.clip(y_pred_node[seed], a_min=0, a_max=None)

                day_outbreak = get_outbreak_from_predictions(y_pred_node_seed[:,:,node_id], strategy=strategy, threshold=threshold)

                day_outbreaks_pred_per_seed[seed, i] = day_outbreak

        return day_outbreaks_pred_per_seed

def compute_errors(outbreak_nodes, day_outbreaks, day_outbreaks_pred_per_seed, seeds):
    
    errors = []
    count_fails = 0
    # compute average error per node and print it
    for i, node_id in enumerate(outbreak_nodes):
        true_day = day_outbreaks[i]
        pred_day = day_outbreaks_pred_per_seed[:, i]
        # if there is a -1 in pred_day, ignore it in the error computation
        if -1 in pred_day:
            count_fails += np.sum(pred_day == -1)
        pred_day = pred_day[pred_day != -1]
        if len(pred_day) == 0:
            continue
        error = np.abs(true_day - pred_day)
        errors.append(np.mean(error))
        mean_error = np.mean(error)
        # print(f'Node {node_id}: Mean error in day of outbreak prediction: {mean_error} days')
        
    errors = np.array(errors)

    return errors, errors.mean(), count_fails

if __name__ == '__main__':

    # Configuration 
    seeds = [42, 43, 44, 45, 46]
    model = 'gru'
    dataset = 'ostrinia'
    embedding = True
    base_path = f'paper_results/{model}_{dataset}_nodes_embd_{embedding}'
    npz_files = [f'{base_path}/{seed}/predictions.npz' for seed in seeds]


    # Example usage
    preds = get_curves([npz_files])

    if dataset == 'peakweather':

        outbreak_nodes, day_outbreaks = filter_nodes(dataset, preds, 20)

        # print('Day of outbreak for each outbreak node:')
        # for node_id, day_outbreak in zip(outbreak_nodes, day_outbreaks):
        #     print(f'Node {node_id}: Days {day_outbreak}')

        day_outbreaks_pred_per_seed = get_days_outbreaks(preds['y_pred_all'], outbreak_nodes, seeds, strategy='first_of_two_maxes', threshold=10)

        # print('Day of outbreak for each predicted node:')
        # for i, node_id in enumerate(outbreak_nodes):
        #     day_outbreak = day_outbreaks_pred_per_seed[:, i]
        #     print(f'Node {node_id}: Days {day_outbreak}')

        per_node_error, average_error, count_fails = compute_errors(outbreak_nodes, day_outbreaks, day_outbreaks_pred_per_seed, seeds)

        # make them mean of the seeds (from the real series) not the days
        y_pred_node_mean = np.mean(preds['y_pred_all'], axis=0)  # mean over seeds

        day_outbreaks_pred_per_seed = get_days_outbreaks(y_pred_node_mean[None,:], outbreak_nodes, [0], strategy='first_of_two_maxes', threshold=10)

        # compute day agains with the mean values
        per_node_error_ensamble, average_error_ensamble, count_fails_ensamble = compute_errors(outbreak_nodes, day_outbreaks, day_outbreaks_pred_per_seed, [0])

        # print avg error and avg error ensemable6
        print(f'Average error in day of outbreak prediction: {average_error} days')
        print(f'Average error in day of outbreak prediction (mean of seeds): {average_error_ensamble} days')

        for th in range(0,25):
            day_outbreaks_pred_per_seed = get_days_outbreaks(y_pred_node_mean[None,:], outbreak_nodes, [0], strategy='first_over_threshold', threshold=th)

            # compute day agains with the mean values
            per_node_error_ensamble, average_error_ensamble, count_fails = compute_errors(outbreak_nodes, day_outbreaks, day_outbreaks_pred_per_seed, [0])

            print(f'Average error in day of outbreak prediction (mean of seeds): {average_error_ensamble} days with threshold {th} with {count_fails} fails')

    if dataset == 'ostrinia':


        outbreak_nodes, day_outbreaks = filter_nodes(dataset, preds, 20)
        
        # print('Day of outbreak for each outbreak node:')
        # for node_id, day_outbreak in zip(outbreak_nodes, day_outbreaks):
        #     print(f'Node {node_id}: Days {day_outbreak}')

        # do the same for predicted values.
        day_outbreaks_pred_per_seed = get_days_outbreaks(preds['y_pred_all'], outbreak_nodes, seeds, strategy='first_of_two_maxes', threshold=10)

        # print('Day of outbreak for each predicted node:')
        # for i, node_id in enumerate(outbreak_nodes):
        #     day_outbreak = day_outbreaks_pred_per_seed[:, i]
        #     print(f'Node {node_id}: Days {day_outbreak}')

        # compute average error per node and print it
        per_node_error, average_error, count_fails = compute_errors(outbreak_nodes, day_outbreaks, day_outbreaks_pred_per_seed, seeds)

        # make them mean of the seeds (from the real series) not the days
        y_pred_node_mean = np.mean(preds['y_pred_all'], axis=0)  # mean over seeds

        # compute day agains with the mean values
        day_outbreaks_pred_per_seed_ensamble = get_days_outbreaks(y_pred_node_mean[None,:], outbreak_nodes, [0], strategy='first_of_two_maxes', threshold=10)

        per_node_error_ensamble, average_error_ensamble, count_fails_ensamble = compute_errors(outbreak_nodes, day_outbreaks, day_outbreaks_pred_per_seed_ensamble, [0])

        # print avg error and avg error ensemable6
        print(f'Average error in day of outbreak prediction: {average_error} days')
        print(f'Average error in day of outbreak prediction (mean of seeds): {average_error_ensamble} days')
        
        for th in range(40,0,-1):
            day_outbreaks_pred_per_seed = get_days_outbreaks(y_pred_node_mean[None,:], outbreak_nodes, [0], strategy='first_fixed_max_th', threshold=th)

            # compute day agains with the mean values
            per_node_error_ensamble, average_error_ensamble, count_fails = compute_errors(outbreak_nodes, day_outbreaks, day_outbreaks_pred_per_seed, [0])

            print(f'Average error in day of outbreak prediction (mean of seeds): {average_error_ensamble} days with threshold {th} with {count_fails} fails')
