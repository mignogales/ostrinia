import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
import pandas as pd
import os


def plot_flag_increment(predictor, data_module, save_path='flag_increment_plot.png'):

    # Get the test dataset
    test_data = data_module.test_dataloader()

    # Get the predictions and true values
    flag_true = []
    flag_pred = []
    y_true = []
    y_pred = []
    predictor.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            batch = batch.to(predictor.device)
            predictions = predictor.predict_step(batch, i)
            flag_true.append(batch['second_target'].numpy())
            flag_pred.append(predictions['y_hat'][...,1:])
            y_true.append(batch.target.y[...,0].numpy())
            y_pred.append(predictions['y_hat'][...,0])

    flag_true = np.concatenate(flag_true, axis=0)
    flag_pred = np.concatenate(flag_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    flag_pred = torch.nn.functional.sigmoid(torch.tensor(flag_pred)).numpy()[...,0]

    for th in np.arange(0, 1, 0.1):
        print(f"Threshold: {th}")
        print(classification_report(flag_true.flatten(), (flag_pred > th).astype(int).flatten(), zero_division=0))

        flag_pred_binary = (flag_pred > th)

        # Convert predictions to binary using a threshold of 0.5
        flag_pred_binary = flag_pred_binary.astype(int)
        flag_true_binary = flag_true.astype(int)

        # Analyze results
        results = analyze_boolean_predictions(flag_true_binary, flag_pred_binary)
        fig = create_comprehensive_plots(results, flag_true_binary, flag_pred_binary)
        # create the directory if it does not exist
        os.makedirs(os.path.dirname(save_path)+f"/{round(th,1)}", exist_ok=True)
        # Save the figure
        fig.savefig(os.path.join(os.path.dirname(save_path), f"{round(th,1)}", "flag_increment_plot.png"))
        plt.close(fig)

    plot_predictions(y_true, y_pred, save_dir=os.path.dirname(save_path),
                prediction_horizon=14, node_subset=6)

    # Print results
    # for key, value in results.items():
    #     print(f"{key}: {value}")




def analyze_boolean_predictions(flag_true, flag_pred):
    """
    Comprehensive analysis of boolean prediction arrays.
    
    Parameters:
    -----------
    flag_true : np.ndarray
        Ground truth boolean array, shape (N, 1, nodes)
    flag_pred : np.ndarray  
        Predicted boolean array, shape (N, 1, nodes)
    
    Returns:
    --------
    dict: Comprehensive results dictionary
    """
    
    # Flatten arrays for analysis
    y_true = flag_true.flatten()
    y_pred = flag_pred.flatten()
    
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # ROC-AUC (if applicable)
    try:
        auc_roc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc_roc = None
    
    # Average Precision Score
    try:
        avg_precision = average_precision_score(y_true, y_pred)
    except ValueError:
        avg_precision = None
    
    # First event analysis
    first_event_results = analyze_first_event_timing(flag_true, flag_pred)
    
    results = {
        'classification_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'negative_predictive_value': npv,
            'auc_roc': auc_roc,
            'average_precision': avg_precision
        },
        'confusion_matrix': cm,
        'confusion_components': {
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        },
        'first_event_analysis': first_event_results,
        'data_info': {
            'total_samples': len(y_true),
            'positive_class_ratio': np.mean(y_true),
            'predicted_positive_ratio': np.mean(y_pred)
        }
    }
    
    return results

def analyze_first_event_timing(flag_true, flag_pred):
    """
    Analyze timing differences for first event detection across nodes.
    
    Parameters:
    -----------
    flag_true : np.ndarray, shape (N, 1, nodes)
    flag_pred : np.ndarray, shape (N, 1, nodes)
    
    Returns:
    --------
    dict: First event timing analysis results
    """
    N, _, nodes = flag_true.shape
    
    first_true_times = []
    first_pred_times = []
    time_differences = []
    detection_success = []
    
    for node in range(nodes):
        # Extract time series for current node
        true_series = flag_true[:, 0, node]
        pred_series = flag_pred[:, 0, node]
        
        # Find first true event
        true_indices = np.where(true_series)[0]
        first_true = true_indices[0] if len(true_indices) > 0 else None
        
        # Find first predicted event
        pred_indices = np.where(pred_series)[0]
        first_pred = pred_indices[0] if len(pred_indices) > 0 else None
        
        if first_true is not None:
            first_true_times.append(first_true)
            
            if first_pred is not None:
                first_pred_times.append(first_pred)
                time_diff = first_pred - first_true
                time_differences.append(time_diff)
                detection_success.append(1)
            else:
                # No prediction made
                detection_success.append(0)
        
    # Statistical analysis of time differences
    if time_differences:
        time_diffs_array = np.array(time_differences)
        
        timing_stats = {
            'abs_mean_time_difference': np.mean(np.abs(time_diffs_array)),
            'median_time_difference': np.median(time_diffs_array),
            'std_time_difference': np.std(time_diffs_array),
            'min_time_difference': np.min(time_diffs_array),
            'max_time_difference': np.max(time_diffs_array),
            'early_detection_rate': np.mean(time_diffs_array < 0),  # Predicted before true
            'late_detection_rate': np.mean(time_diffs_array > 0),   # Predicted after true
            'exact_timing_rate': np.mean(time_diffs_array == 0)     # Exact timing match
        }
    else:
        timing_stats = {}
    
    return {
        'nodes_with_true_events': len(first_true_times),
        'nodes_with_predicted_events': len(first_pred_times),
        'detection_success_rate': np.mean(detection_success) if detection_success else 0,
        'first_true_times': first_true_times,
        'first_pred_times': first_pred_times,
        'time_differences': time_differences,
        'timing_statistics': timing_stats
    }

def create_comprehensive_plots(results, flag_true, flag_pred):
    """
    Create comprehensive visualization plots.
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Confusion Matrix
    plt.subplot(3, 4, 1)
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted False', 'Predicted True'],
                yticklabels=['Actual False', 'Actual True'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. Metrics Bar Plot
    plt.subplot(3, 4, 2)
    metrics = results['classification_metrics']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'NPV']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                    metrics['specificity'], metrics['f1_score'], metrics['negative_predictive_value']]
    
    bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 
                                                      'gold', 'plum', 'orange'])
    plt.ylim(0, 1.1)
    plt.title('Classification Metrics')
    plt.xticks(rotation=45)
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. First Event Time Differences Histogram
    plt.subplot(3, 4, 3)
    time_diffs = results['first_event_analysis']['time_differences']
    if time_diffs:
        plt.hist(time_diffs, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='Perfect Timing')
        plt.xlabel('Time Difference (Predicted - True)')
        plt.ylabel('Frequency')
        plt.title('First Event Detection Timing')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No valid time differences', ha='center', va='center', 
                transform=plt.gca().transAxes)
        plt.title('First Event Detection Timing')
    
    # 4. Detection Success Rate by Node (sample of first 50 nodes)
    plt.subplot(3, 4, 4)
    N, _, nodes = flag_true.shape
    sample_nodes = min(50, nodes)
    
    success_rates = []
    for node in range(sample_nodes):
        true_series = flag_true[:, 0, node]
        pred_series = flag_pred[:, 0, node]
        
        if np.any(true_series):
            success = 1 if np.any(pred_series) else 0
            success_rates.append(success)
        else:
            success_rates.append(np.nan)  # No true events to detect
    
    valid_rates = [r for r in success_rates if not np.isnan(r)]
    if valid_rates:
        plt.bar(range(len(valid_rates)), valid_rates, alpha=0.7, color='mediumseagreen')
        plt.xlabel('Node Index')
        plt.ylabel('Detection Success (0/1)')
        plt.title(f'First Event Detection Success\n(First {len(valid_rates)} nodes with events)')
    else:
        plt.text(0.5, 0.5, 'No nodes with events', ha='center', va='center',
                transform=plt.gca().transAxes)
        plt.title('First Event Detection Success')
    
    # 5. Time Series Sample (first node with events)
    plt.subplot(3, 4, 5)
    sample_node = None
    for node in range(min(10, nodes)):  # Check first 10 nodes
        if np.any(flag_true[:, 0, node]):
            sample_node = node
            break
    
    if sample_node is not None:
        time_steps = np.arange(N)
        plt.plot(time_steps, flag_true[:, 0, sample_node], 'o-', label='True', 
                markersize=4, linewidth=2, color='blue')
        plt.plot(time_steps, flag_pred[:, 0, sample_node], 's-', label='Predicted', 
                markersize=4, linewidth=2, color='red', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Boolean Value')
        plt.title(f'Time Series Comparison (Node {sample_node})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No nodes with events found', ha='center', va='center',
                transform=plt.gca().transAxes)
        plt.title('Time Series Comparison')
    
    # 6. Class Distribution
    plt.subplot(3, 4, 6)
    y_true_flat = flag_true.flatten()
    y_pred_flat = flag_pred.flatten()
    
    categories = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    values = [results['confusion_components']['true_negatives'],
              results['confusion_components']['false_positives'],
              results['confusion_components']['false_negatives'],
              results['confusion_components']['true_positives']]
    colors = ['lightblue', 'lightcoral', 'orange', 'lightgreen']
    
    plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Prediction Distribution')
    
    # 7-8. Timing Analysis Subplots
    if time_diffs:
        # Box plot of time differences
        plt.subplot(3, 4, 7)
        plt.boxplot(time_diffs, patch_artist=True, 
                   boxprops=dict(facecolor='lightsteelblue', alpha=0.7))
        plt.ylabel('Time Difference')
        plt.title('Time Difference Distribution')
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution of time differences
        plt.subplot(3, 4, 8)
        sorted_diffs = np.sort(time_diffs)
        cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
        plt.plot(sorted_diffs, cumulative, linewidth=2, color='darkblue')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Perfect Timing')
        plt.xlabel('Time Difference')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution of Time Differences')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # 9. Precision-Recall Curve (if applicable)
    plt.subplot(3, 4, 9)
    y_true_flat = flag_true.flatten()
    y_pred_flat = flag_pred.flatten()
    
    if len(np.unique(y_true_flat)) > 1 and len(np.unique(y_pred_flat)) > 1:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_flat, y_pred_flat)
        plt.plot(recall_vals, precision_vals, linewidth=2, color='purple')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Insufficient class variation\nfor PR curve', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Precision-Recall Curve')
    
    # 10. Node-wise Event Frequency
    plt.subplot(3, 4, 10)
    true_freq = np.sum(flag_true[:, 0, :], axis=0)  # Events per node
    pred_freq = np.sum(flag_pred[:, 0, :], axis=0)
    
    sample_nodes = min(20, nodes)
    node_indices = np.arange(sample_nodes)
    
    width = 0.35
    plt.bar(node_indices - width/2, true_freq[:sample_nodes], width, 
           label='True Events', alpha=0.8, color='blue')
    plt.bar(node_indices + width/2, pred_freq[:sample_nodes], width,
           label='Predicted Events', alpha=0.8, color='red')
    
    plt.xlabel('Node Index')
    plt.ylabel('Total Events')
    plt.title(f'Event Frequency by Node (First {sample_nodes})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 11. Early vs Late Detection Analysis
    plt.subplot(3, 4, 11)
    if time_diffs:
        early_count = np.sum(np.array(time_diffs) < 0)
        exact_count = np.sum(np.array(time_diffs) == 0)
        late_count = np.sum(np.array(time_diffs) > 0)
        
        categories = ['Early\nDetection', 'Exact\nTiming', 'Late\nDetection']
        counts = [early_count, exact_count, late_count]
        colors = ['green', 'gold', 'red']
        
        bars = plt.bar(categories, counts, color=colors, alpha=0.7)
        plt.ylabel('Count')
        plt.title('Detection Timing Categories')
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No timing data available', ha='center', va='center',
                transform=plt.gca().transAxes)
        plt.title('Detection Timing Categories')
    
    # 12. Summary Statistics Table
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Create summary text
    summary_text = f"""
    CLASSIFICATION SUMMARY
    ═══════════════════════
    Accuracy: {results['classification_metrics']['accuracy']:.3f}
    F1-Score: {results['classification_metrics']['f1_score']:.3f}
    Precision: {results['classification_metrics']['precision']:.3f}
    Recall: {results['classification_metrics']['recall']:.3f}
    
    FIRST EVENT ANALYSIS
    ══════════════════════
    Detection Success: {results['first_event_analysis']['detection_success_rate']:.3f}
    Nodes with Events: {results['first_event_analysis']['nodes_with_true_events']}
    """
    
    if results['first_event_analysis']['timing_statistics']:
        timing_stats = results['first_event_analysis']['timing_statistics']
        summary_text += f"""
    Abs Mean Time Diff: {timing_stats['abs_mean_time_difference']:.2f}
    Early Detection: {timing_stats['early_detection_rate']:.3f}
    Late Detection: {timing_stats['late_detection_rate']:.3f}
        """
    
    plt.text(0.05, 0.95, summary_text, fontsize=10, fontfamily='monospace',
             verticalalignment='top', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
    
    return fig

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Union, Optional, Tuple
import warnings

def plot_predictions(y_true: np.ndarray, 
                    y_pred: np.ndarray,
                    save_dir: Union[str, Path],
                    prediction_horizon: int = 14,
                    node_subset: Optional[int] = None,
                    figsize_per_node: Tuple[int, int] = (12, 4),
                    max_nodes_single_plot: int = 6,
                    dpi: int = 300,
                    format: str = 'png',
                    title_prefix: str = 'Node') -> None:
    """
    Visualize time series predictions with persistence model baseline.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values with shape (N, 1, nodes)
    y_pred : np.ndarray  
        Predicted values with shape (N, 1, nodes)
    save_dir : str or Path
        Directory to save plots
    prediction_horizon : int, default=14
        Number of days ahead for predictions (for persistence model)
    node_subset : int, optional
        Number of nodes to plot (randomly selected if None)
    figsize_per_node : tuple, default=(12, 4)
        Figure size per node subplot
    max_nodes_single_plot : int, default=6
        Maximum nodes in single plot before creating separate files
    dpi : int, default=300
        Resolution for saved figures
    format : str, default='png'
        File format for saved plots
    title_prefix : str, default='Node'
        Prefix for node titles
    """
    
    # Input validation
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have identical shapes"
    assert len(y_true.shape) == 3, "Input arrays must be 3D (N, 1, nodes)"
    assert y_true.shape[1] == 1, "Second dimension must be 1"
    
    # Extract dimensions
    N, _, n_nodes = y_true.shape
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Handle node selection
    if node_subset is not None and node_subset < n_nodes:
        selected_nodes = np.random.choice(n_nodes, size=node_subset, replace=False)
        nodes_to_plot = sorted(selected_nodes)
    else:
        nodes_to_plot = list(range(n_nodes))
    
    # Reshape data for easier handling
    y_true_reshaped = y_true.squeeze(axis=1)  # (N, nodes)
    y_pred_reshaped = y_pred.squeeze(axis=1)  # (N, nodes)
    
    # Create persistence model (14 days lag)
    y_persistence = np.full_like(y_true_reshaped, np.nan)
    if N > prediction_horizon:
        y_persistence[prediction_horizon:] = y_true_reshaped[:-prediction_horizon]
    
    # Time axis
    time_steps = np.arange(N)
    
    # Determine plotting strategy based on number of nodes
    n_nodes_to_plot = len(nodes_to_plot)
    
    if n_nodes_to_plot <= max_nodes_single_plot:
        # Single plot with subplots
        _create_single_plot(y_true_reshaped, y_pred_reshaped, y_persistence, 
                          time_steps, nodes_to_plot, save_path, 
                          figsize_per_node, dpi, format, title_prefix)
    else:
        # Multiple plots
        _create_multiple_plots(y_true_reshaped, y_pred_reshaped, y_persistence,
                             time_steps, nodes_to_plot, save_path,
                             figsize_per_node, max_nodes_single_plot, 
                             dpi, format, title_prefix)
    
    print(f"Plots saved to: {save_path}")
    print(f"Total nodes plotted: {n_nodes_to_plot}")

def _create_single_plot(y_true: np.ndarray, y_pred: np.ndarray, y_persistence: np.ndarray,
                       time_steps: np.ndarray, nodes_to_plot: list, save_path: Path,
                       figsize_per_node: Tuple[int, int], dpi: int, format: str,
                       title_prefix: str) -> None:
    """Create a single plot with multiple subplots."""
    
    n_nodes = len(nodes_to_plot)
    fig, axes = plt.subplots(n_nodes, 1, 
                            figsize=(figsize_per_node[0], figsize_per_node[1] * n_nodes),
                            squeeze=False)
    
    for i, node_idx in enumerate(nodes_to_plot):
        ax = axes[i, 0] if n_nodes > 1 else axes[0, 0]
        
        # Plot data
        ax.plot(time_steps, y_true[:, node_idx], 'b-', label='True', linewidth=1.5, alpha=0.8)
        ax.plot(time_steps, y_pred[:, node_idx], 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
        ax.plot(time_steps, y_persistence[:, node_idx], 'g:', label='Persistence (14d)', linewidth=1.5, alpha=0.7)
        
        # Styling
        ax.set_title(f'{title_prefix} {node_idx}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Calculate and display metrics
        _add_metrics_text(ax, y_true[:, node_idx], y_pred[:, node_idx], y_persistence[:, node_idx])
    
    plt.tight_layout()
    plt.savefig(save_path / f'predictions_all_nodes.{format}', dpi=dpi, bbox_inches='tight')
    plt.close()

def _create_multiple_plots(y_true: np.ndarray, y_pred: np.ndarray, y_persistence: np.ndarray,
                          time_steps: np.ndarray, nodes_to_plot: list, save_path: Path,
                          figsize_per_node: Tuple[int, int], max_nodes_single_plot: int,
                          dpi: int, format: str, title_prefix: str) -> None:
    """Create multiple separate plot files."""
    
    # Group nodes into batches
    node_batches = [nodes_to_plot[i:i + max_nodes_single_plot] 
                   for i in range(0, len(nodes_to_plot), max_nodes_single_plot)]
    
    for batch_idx, batch_nodes in enumerate(node_batches):
        n_nodes_batch = len(batch_nodes)
        
        fig, axes = plt.subplots(n_nodes_batch, 1,
                                figsize=(figsize_per_node[0], figsize_per_node[1] * n_nodes_batch),
                                squeeze=False)
        
        for i, node_idx in enumerate(batch_nodes):
            ax = axes[i, 0] if n_nodes_batch > 1 else axes[0, 0]
            
            # Plot data
            ax.plot(time_steps, y_true[:, node_idx], 'b-', label='True', linewidth=1.5, alpha=0.8)
            ax.plot(time_steps, y_pred[:, node_idx], 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
            ax.plot(time_steps, y_persistence[:, node_idx], 'g:', label='Persistence (14d)', linewidth=1.5, alpha=0.7)
            
            # Styling
            ax.set_title(f'{title_prefix} {node_idx}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            # Calculate and display metrics
            _add_metrics_text(ax, y_true[:, node_idx], y_pred[:, node_idx], y_persistence[:, node_idx])
        
        plt.tight_layout()
        plt.savefig(save_path / f'predictions_batch_{batch_idx + 1}.{format}', dpi=dpi, bbox_inches='tight')
        plt.close()

def _add_metrics_text(ax, y_true: np.ndarray, y_pred: np.ndarray, y_persistence: np.ndarray) -> None:
    """Add performance metrics as text on the plot."""
    
    # Calculate RMSE
    def rmse(true, pred):
        mask = ~(np.isnan(true) | np.isnan(pred))
        if mask.sum() == 0:
            return np.nan
        return np.sqrt(np.mean((true[mask] - pred[mask]) ** 2))
    
    # Calculate MAE
    def mae(true, pred):
        mask = ~(np.isnan(true) | np.isnan(pred))
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs(true[mask] - pred[mask]))
    
    rmse_pred = rmse(y_true, y_pred)
    rmse_pers = rmse(y_true, y_persistence)
    mae_pred = mae(y_true, y_pred)
    mae_pers = mae(y_true, y_persistence)
    
    # Create metrics text
    metrics_text = f'RMSE - Pred: {rmse_pred:.3f}, Pers: {rmse_pers:.3f}\n'
    metrics_text += f'MAE - Pred: {mae_pred:.3f}, Pers: {mae_pers:.3f}'
    
    # Add text box
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
           fontsize=10, fontfamily='monospace')

