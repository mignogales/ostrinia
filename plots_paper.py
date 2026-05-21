import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from cycler import cycler

# Configure publication-quality aesthetics
def setup_plot_style():
    """Configure matplotlib for high-quality, publication-ready plots."""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman']
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 13
    rcParams['axes.titlesize'] = 14
    rcParams['axes.titleweight'] = 'bold'
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 16
    rcParams['axes.linewidth'] = 1.2
    rcParams['grid.linewidth'] = 0.5
    rcParams['lines.linewidth'] = 2.0
    rcParams['lines.markersize'] = 6
    rcParams['xtick.major.width'] = 1.2
    rcParams['ytick.major.width'] = 1.2
    rcParams['xtick.minor.width'] = 0.8
    rcParams['ytick.minor.width'] = 0.8
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False

def get_color_palette():
    """Return a sophisticated color palette for different models."""
    return {
        'ground_truth': '#2E4057',  # Deep blue-gray
        'model_1': '#E63946',        # Vibrant red
        'model_2': '#06A77D',        # Teal green
        'model_3': '#F77F00',        # Orange
        'model_4': '#6A4C93',        # Purple
        'persistent': '#95A5A6',     # Light gray
        'grid': '#E8E8E8',           # Very light gray
        'background': '#FAFAFA'      # Off-white
    }

def plot_node_predictions_ensemble(npz_paths_list, node_id, ax, data_plot_list=None, 
                                   normalize=False, add_persistent=False, delay_persistent=1, 
                                   start_step=0, end_step=None):
    """
    Load predictions from multiple .npz files across different configurations, compute mean predictions,
    and plot real vs ensemble predicted for a specified node on a given axis.
    
    Parameters:
    -----------
    npz_paths_list : list of lists of str
        List containing lists of paths to .npz files. Each sublist represents a different configuration.
    node_id : int
        Index of the node to visualize
    ax : matplotlib.axes.Axes
        Axis object to plot on
    data_plot_list : list of dict or None
        List of metadata dictionaries for each configuration
    normalize : bool
        If True, normalize predictions by their max and scale by ground truth max
    add_persistent : bool
        If True, add persistent model baseline (previous timestep prediction)
    delay_persistent : int
        Number of time steps to delay for persistent model (default: 1 for t-1 prediction)
    start_step : int
        Starting time step for plotting
    end_step : int or None
        Ending time step for plotting (None for all remaining steps)
    """
    colors = get_color_palette()
    
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
    
    # Validate node_id
    n_nodes = y_true.shape[2]
    if node_id >= n_nodes or node_id < 0:
        raise ValueError(f"node_id must be between 0 and {n_nodes-1}")
    
    # Extract node-specific data
    node_true = y_true[:, 0, node_id]
    node_preds = [y_pred[:, 0, node_id] for y_pred in y_pred_means]
    
    # Compute persistent model (t-delay prediction)
    if add_persistent:
        if delay_persistent <= 0:
            raise ValueError("delay_persistent must be a positive integer")
        
        if delay_persistent >= len(node_true):
            raise ValueError(f"delay_persistent ({delay_persistent}) must be less than sequence length ({len(node_true)})")
        
        node_persistent = np.concatenate([
            np.repeat(node_true[0], delay_persistent),
            node_true[:-delay_persistent]
        ])
    
    # Apply time window
    if end_step is None:
        end_step = len(node_true)
    
    node_true = node_true[start_step:end_step]
    node_preds = [pred[start_step:end_step] for pred in node_preds]
    if add_persistent:
        node_persistent = node_persistent[start_step:end_step]
    
    # Normalization
    if normalize:
        true_max = np.max(node_true)
        node_preds = [(pred / np.max(pred)) * true_max for pred in node_preds]
        if add_persistent:
            node_persistent = (node_persistent / np.max(node_persistent)) * true_max
    
    # Set background color
    ax.set_facecolor(colors['background'])
    
    # Plot on provided axis
    time_steps = np.arange(start_step, start_step + len(node_true))
    
    # Plot ground truth with enhanced styling
    ax.plot(time_steps, node_true, label='Ground Truth', 
            color=colors['ground_truth'], linewidth=2.5, alpha=0.9, zorder=10)
    
    # Model colors and styles
    model_colors = [colors['model_1'], colors['model_2'], colors['model_3'], colors['model_4']]
    linestyles = ['-', '-', '-', '-']
    alphas = [0.85, 0.85, 0.85, 0.85]
    
    metrics_list = []
    for idx, node_pred in enumerate(node_preds):
        label_suffix = f'Model {idx+1}'
        if data_plot_list and idx < len(data_plot_list):
            model = data_plot_list[idx].get('model', '').upper()
            label_suffix = f'{model}'
        
        color = model_colors[idx % len(model_colors)]
        
        # Plot prediction line
        ax.plot(time_steps, node_pred, 
                label=label_suffix, 
                color=color, 
                linewidth=2.0, 
                alpha=alphas[idx % len(alphas)], 
                linestyle=linestyles[idx % len(linestyles)],
                zorder=8-idx)
        
        mse = np.mean((node_true - node_pred) ** 2)
        mae = np.mean(np.abs(node_true - node_pred))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((node_true - node_pred) / (node_true + 1e-8))) * 100
        metrics_list.append({'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
    
    # Add persistent model
    if add_persistent:
        ax.plot(time_steps, node_persistent, 
                label=f'Persistent (t−{delay_persistent})', 
                color=colors['persistent'], 
                linewidth=2.0, 
                alpha=0.7, 
                linestyle='--',
                zorder=5)
        
        mse_persistent = np.mean((node_true - node_persistent) ** 2)
        mae_persistent = np.mean(np.abs(node_true - node_persistent))
        rmse_persistent = np.sqrt(mse_persistent)
        mape_persistent = np.mean(np.abs((node_true - node_persistent) / (node_true + 1e-8))) * 100
        metrics_list.append({'MSE': mse_persistent, 'MAE': mae_persistent, 
                           'RMSE': rmse_persistent, 'MAPE': mape_persistent, 'type': 'persistent'})
    
    # Enhanced styling
    ax.set_xlabel('Time Step', fontsize=13, fontweight='semibold')
    ax.set_ylabel('Cumulative captures', fontsize=13, fontweight='semibold')
    
    title = f'Node {node_id}: Ground Truth vs. Predictions'
    if normalize:
        title += ' (Normalized)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Enhanced legend
    legend = ax.legend(loc='best', fontsize=10, frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.95,
                      edgecolor='lightgray', borderpad=0.8)
    legend.get_frame().set_facecolor('white')
    
    # Enhanced grid
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color=colors['grid'])
    ax.set_axisbelow(True)
    
    # Add minor ticks
    ax.minorticks_on()
    ax.tick_params(which='minor', length=3, width=0.8, direction='in')
    ax.tick_params(which='major', length=5, width=1.2, direction='in')
    
    return metrics_list

def plot_multiple_nodes_ensemble(npz_paths_list, node_ids, output_dir='plots', data_plot_list=None, 
                                 normalize=False, add_persistent=False, delay_persistent=1, 
                                 start_step=0, end_step=None, plot_both_versions=False):
    """
    Generate ensemble comparison plots for multiple nodes in subplots.
    
    Parameters:
    -----------
    npz_paths_list : list of lists of str
        List containing lists of paths to .npz files for each configuration
    node_ids : list of int
        List of node indices to visualize
    output_dir : str
        Directory to save the plots
    data_plot_list : list of dict or None
        List of metadata dictionaries for each configuration
    normalize : bool
        If True, normalize predictions by their max and scale by ground truth max
    add_persistent : bool
        If True, add persistent model baseline
    delay_persistent : int
        Number of time steps to delay for persistent model
    start_step : int
        Starting time step for plotting
    end_step : int or None
        Ending time step for plotting
    plot_both_versions : bool
        If True, generate both normalized and non-normalized versions (overrides normalize parameter)
    """
    setup_plot_style()
    
    n_nodes = len(node_ids)
    
    # Handle plot_both_versions flag - create side-by-side subplots
    if plot_both_versions:
        # Plot both versions in the same frame for each node
        n_nodes = len(node_ids)
        fig, axes = plt.subplots(n_nodes, 1, figsize=(14, 6 * n_nodes), 
                                facecolor='white', constrained_layout=True)
        
        # Handle single node case
        if n_nodes == 1:
            axes = [axes]
        
        all_metrics = {'non_normalized': {}, 'normalized': {}}
        
        for idx, node_id in enumerate(node_ids):
            # Plot both normalized and non-normalized in the same axis
            metrics_both = _plot_node_both_versions_ensemble(
                npz_paths_list, node_id, axes[idx], data_plot_list,
                add_persistent=add_persistent, delay_persistent=delay_persistent,
                start_step=start_step, end_step=end_step
            )
            all_metrics['non_normalized'][node_id] = metrics_both['non_normalized']
            all_metrics['normalized'][node_id] = metrics_both['normalized']
            
            # Print metrics for both versions
            print(f"\n{'='*70}")
            print(f"Node {node_id} Performance Metrics")
            print(f"{'='*70}")
            
            print("\nNON-NORMALIZED:")
            print(f"{'Model':<25} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
            print(f"{'-'*70}")
            _print_metrics(metrics_both['non_normalized'], data_plot_list)
            
            print("\nNORMALIZED:")
            print(f"{'Model':<25} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
            print(f"{'-'*70}")
            _print_metrics(metrics_both['normalized'], data_plot_list)
            print(f"{'='*70}\n")
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = _generate_filename(output_dir, data_plot_list, node_ids, 
                                        normalize=None, add_persistent=add_persistent, 
                                        start_step=start_step, end_step=end_step,
                                        both_versions=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"\n{'='*70}")
        print(f"Plot saved to: {output_path}")
        print(f"{'='*70}\n")
        
        return all_metrics
    
    else:
        # Single version plot
        n_nodes = len(node_ids)
        # Single version plot
        fig, axes = plt.subplots(n_nodes, 1, figsize=(14, 6 * n_nodes), 
                                facecolor='white', constrained_layout=True)
        
        # Handle single node case
        if n_nodes == 1:
            axes = [axes]
        
        all_metrics = {}
        for idx, node_id in enumerate(node_ids):
            metrics = plot_node_predictions_ensemble(
                npz_paths_list, node_id, axes[idx], data_plot_list, 
                normalize, add_persistent, delay_persistent, start_step, end_step
            )
            all_metrics[node_id] = metrics
            
            # Print metrics in formatted table
            print(f"\n{'='*70}")
            print(f"Node {node_id} Performance Metrics")
            print(f"{'='*70}")
            print(f"{'Model':<25} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
            print(f"{'-'*70}")
            _print_metrics(metrics, data_plot_list)
            print(f"{'='*70}\n")
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = _generate_filename(output_dir, data_plot_list, node_ids, 
                                        normalize=normalize, add_persistent=add_persistent, 
                                        start_step=start_step, end_step=end_step,
                                        both_versions=False)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"\n{'='*70}")
        print(f"Plot saved to: {output_path}")
        print(f"{'='*70}\n")
        
        return all_metrics

def _plot_node_both_versions_ensemble(npz_paths_list, node_id, ax, data_plot_list=None,
                                       add_persistent=False, delay_persistent=1,
                                       start_step=0, end_step=None):
    """
    Plot both normalized and non-normalized predictions in the same frame.
    
    Returns a dictionary with metrics for both versions.
    """
    colors = get_color_palette()
    
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
        
        y_preds_stacked = np.stack(y_preds_all, axis=0)
        y_pred_mean = np.mean(y_preds_stacked, axis=0)
        y_pred_means.append(y_pred_mean)

    y_true = np.clip(y_true, a_min=0, a_max=None)
    
    n_nodes = y_true.shape[2]
    if node_id >= n_nodes or node_id < 0:
        raise ValueError(f"node_id must be between 0 and {n_nodes-1}")
    
    # Extract node-specific data
    node_true = y_true[:, 0, node_id]
    node_preds = [y_pred[:, 0, node_id] for y_pred in y_pred_means]
    
    # Compute persistent model
    if add_persistent:
        if delay_persistent <= 0:
            raise ValueError("delay_persistent must be a positive integer")
        if delay_persistent >= len(node_true):
            raise ValueError(f"delay_persistent ({delay_persistent}) must be less than sequence length ({len(node_true)})")
        
        node_persistent = np.concatenate([
            np.repeat(node_true[0], delay_persistent),
            node_true[:-delay_persistent]
        ])
    
    # Apply time window
    if end_step is None:
        end_step = len(node_true)
    
    node_true_windowed = node_true[start_step:end_step]
    node_preds_windowed = [pred[start_step:end_step] for pred in node_preds]
    if add_persistent:
        node_persistent_windowed = node_persistent[start_step:end_step]
    
    # Prepare normalized versions
    true_max = np.max(node_true_windowed)
    node_preds_normalized = [(pred / np.max(pred)) * true_max for pred in node_preds_windowed]
    if add_persistent:
        node_persistent_normalized = (node_persistent_windowed / np.max(node_persistent_windowed)) * true_max
    
    # Set background
    ax.set_facecolor(colors['background'])
    time_steps = np.arange(start_step, start_step + len(node_true_windowed))
    
    # Plot ground truth
    ax.plot(time_steps, node_true_windowed, label='Ground Truth', 
            color=colors['ground_truth'], linewidth=2.5, alpha=0.9, zorder=10)
    
    # Model colors
    model_colors = [colors['model_1'], colors['model_2'], colors['model_3'], colors['model_4']]
    
    metrics_non_normalized = []
    metrics_normalized = []
    
    # Plot all predictions
    for idx, (node_pred, node_pred_norm) in enumerate(zip(node_preds_windowed, node_preds_normalized)):
        label_suffix = f'Model {idx+1}'
        if data_plot_list and idx < len(data_plot_list):
            model = data_plot_list[idx].get('model', '').upper()
            label_suffix = f'{model}'
        
        color = model_colors[idx % len(model_colors)]
        
        # Non-normalized (solid line)
        ax.plot(time_steps, node_pred, 
                label=f'{label_suffix}', 
                color=color, 
                linewidth=2.0, 
                alpha=0.85, 
                linestyle='-',
                zorder=8-idx)
        
        # Normalized (dashed line)
        ax.plot(time_steps, node_pred_norm, 
                label=f'{label_suffix} (Norm)', 
                color=color, 
                linewidth=2.0, 
                alpha=0.65, 
                linestyle='--',
                zorder=7-idx)
        
        # Compute metrics for non-normalized
        mse = np.mean((node_true_windowed - node_pred) ** 2)
        mae = np.mean(np.abs(node_true_windowed - node_pred))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((node_true_windowed - node_pred) / (node_true_windowed + 1e-8))) * 100
        metrics_non_normalized.append({'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
        
        # Compute metrics for normalized
        mse_norm = np.mean((node_true_windowed - node_pred_norm) ** 2)
        mae_norm = np.mean(np.abs(node_true_windowed - node_pred_norm))
        rmse_norm = np.sqrt(mse_norm)
        mape_norm = np.mean(np.abs((node_true_windowed - node_pred_norm) / (node_true_windowed + 1e-8))) * 100
        metrics_normalized.append({'MSE': mse_norm, 'MAE': mae_norm, 'RMSE': rmse_norm, 'MAPE': mape_norm})
    
    # Add persistent model if requested
    if add_persistent:
        ax.plot(time_steps, node_persistent_windowed, 
                label=f'Persistent (t−{delay_persistent})', 
                color=colors['persistent'], 
                linewidth=2.0, 
                alpha=0.7, 
                linestyle='-',
                zorder=5)
        
        mse_p = np.mean((node_true_windowed - node_persistent_windowed) ** 2)
        mae_p = np.mean(np.abs(node_true_windowed - node_persistent_windowed))
        rmse_p = np.sqrt(mse_p)
        mape_p = np.mean(np.abs((node_true_windowed - node_persistent_windowed) / (node_true_windowed + 1e-8))) * 100
        metrics_non_normalized.append({'MSE': mse_p, 'MAE': mae_p, 'RMSE': rmse_p, 'MAPE': mape_p, 'type': 'persistent'})
    
    # Styling
    ax.set_xlabel('Time Step', fontsize=13, fontweight='semibold')
    ax.set_ylabel('Cumulative captures', fontsize=13, fontweight='semibold')
    ax.set_title(f'Node {node_id}: Ground Truth vs. Predictions (Both Versions)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, linestyle=':', alpha=0.4, color=colors['grid'], zorder=0)
    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='gray', 
              fancybox=True, shadow=True, ncol=2)
    ax.tick_params(direction='out', length=6, width=1.2)
    
    return {'non_normalized': metrics_non_normalized, 'normalized': metrics_normalized}

def _print_metrics(metrics, data_plot_list):
    """Helper function to print metrics in formatted table."""
    config_idx = 0
    for m_idx, m in enumerate(metrics):
        if m.get('type') == 'persistent':
            model_name = 'Persistent Model'
        else:
            if data_plot_list and config_idx < len(data_plot_list):
                model_name = data_plot_list[config_idx].get('model', f'Model {config_idx+1}').upper()
            else:
                model_name = f'Model {config_idx+1}'
            config_idx += 1
        
        print(f"{model_name:<25} {m['MSE']:>10.4f} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['MAPE']:>9.2f}%")

def _generate_filename(output_dir, data_plot_list, node_ids, normalize, add_persistent, 
                      start_step, end_step, both_versions=False):
    """Helper function to generate output filename."""
    if data_plot_list and len(data_plot_list) >= 2:
        filename_parts = []
        for dp in data_plot_list:
            model = dp.get('model', 'model')
            embd = 'embd' if dp.get('embedding', False) else 'no_embd'
            filename_parts.append(f"{model}_{embd}")
        
        if both_versions:
            norm_str = '_both_versions'
        else:
            norm_str = '_normalized' if normalize else ''
        
        pers_str = '_with_persistent' if add_persistent else ''
        time_str = f'_steps_{start_step}_{end_step-1}' if end_step else f'_steps_{start_step}_end'
        nodes_str = '_'.join(map(str, node_ids))
        return os.path.join(output_dir, 
            f"comparison_{'_vs_'.join(filename_parts)}_nodes_{nodes_str}{norm_str}{pers_str}{time_str}.png")
    else:
        if both_versions:
            norm_str = '_both_versions'
        else:
            norm_str = '_normalized' if normalize else ''
        
        pers_str = '_with_persistent' if add_persistent else ''
        time_str = f'_steps_{start_step}_{end_step-1}' if end_step else f'_steps_{start_step}_end'
        nodes_str = '_'.join(map(str, node_ids))
        return os.path.join(output_dir, 
            f'nodes_{nodes_str}_ensemble_comparison{norm_str}{pers_str}{time_str}.png')

# Usage example:
if __name__ == "__main__":

    dataset = "ostrinia"  # Options: 'ostrinia', 'peakweather', etc.

    # Configuration 1
    seeds1 = [42, 43, 44, 45, 46]
    model1 = 'gru'
    dataset1 = dataset
    embedding1 = True if dataset == 'peakweather' else False
    data_plot1 = {'seeds': seeds1, 'model': model1, 'dataset': dataset1, 'embedding': embedding1}
    base_path1 = f'paper_results/{model1}_{dataset1}_nodes_embd_{embedding1}'
    npz_files1 = [f'{base_path1}/{seed}/predictions.npz' for seed in seeds1]
    
    # Configuration 2
    seeds2 = [42, 43, 44, 45, 46]
    model2 = 'grugcn'
    dataset2 = dataset
    embedding2 = True if dataset == 'peakweather' else False
    data_plot2 = {'seeds': seeds2, 'model': model2, 'dataset': dataset2, 'embedding': embedding2}
    base_path2 = f'paper_results/{model2}_{dataset2}_nodes_embd_{embedding2}'
    npz_files2 = [f'{base_path2}/{seed}/predictions.npz' for seed in seeds2]

    # configuration 3
    seeds3 = [42, 43, 44, 45, 46]
    model3 = 'mlp'
    dataset3 = dataset
    embedding3 = True if dataset == 'peakweather' else False
    data_plot3 = {'seeds': seeds3, 'model': model3, 'dataset': dataset3, 'embedding': embedding3}
    base_path3 = f'paper_results/{model3}_{dataset3}_nodes_embd_{embedding3}'
    npz_files3 = [f'{base_path3}/{seed}/predictions.npz' for seed in seeds3]

    # configuration 4
    seeds4 = [42, 43, 44, 45, 46]
    model4 = 'transformer'
    dataset4 = dataset
    embedding4 = True if dataset == 'peakweather' else False
    data_plot4 = {'seeds': seeds4, 'model': model4, 'dataset': dataset4, 'embedding': embedding4}
    base_path4 = f'paper_results/{model4}_{dataset4}_nodes_embd_{embedding4}'
    npz_files4 = [f'{base_path4}/{seed}/predictions.npz' for seed in seeds4]

    # configuration 5
    seeds5 = [42, 43, 44, 45, 46]
    model5 = 'degree_day'
    dataset5 = dataset
    embedding5 = "linear"
    data_plot5 = {'seeds': seeds5, 'model': model5, 'dataset': dataset5, 'embedding': embedding5}
    base_path5 = f'paper_results/{model5}_{embedding5}_pooledTrue_{dataset5}'
    npz_files5 = [f'{base_path5}/{seed}/predictions.npz' for seed in seeds5]

    # Plot multiple nodes with both configurations in subplots
    # Option 1: Plot both normalized and non-normalized versions
    plot_multiple_nodes_ensemble(
        npz_paths_list=[npz_files1, npz_files2, npz_files3, npz_files4, npz_files5],
        node_ids=[40, 56] if dataset == 'peakweather' else [10],
        output_dir='plots',
        data_plot_list=[data_plot1, data_plot2, data_plot3, data_plot4, data_plot5],
        add_persistent=True,
        delay_persistent=14,
        start_step=0 if dataset1 == 'ostrinia' else 180,
        end_step=200 if dataset1 == 'ostrinia' else 300,
        plot_both_versions=False  # Generate both normalized and non-normalized plots
    )
    
    # Option 2: Plot only one version
    # plot_multiple_nodes_ensemble(
    #     npz_paths_list=[npz_files1, npz_files2],
    #     node_ids=[40, 56],
    #     output_dir='plots',
    #     data_plot_list=[data_plot1, data_plot2],
    #     normalize=False,  # or True for normalized only
    #     add_persistent=True,
    #     delay_persistent=14,
    #     start_step=30 if dataset1 == 'ostrinia' else 150,
    #     end_step=200 if dataset1 == 'ostrinia' else 300
    # )