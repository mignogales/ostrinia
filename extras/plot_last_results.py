import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import pickle

class ResultsAggregator:
    """Aggregates predictions and metrics across multiple runs."""
    
    def __init__(self, save_dir: Path, dataset_name: str):
        self.dataset_name = dataset_name
        self.save_dir = Path(save_dir) / dataset_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.save_dir / "predictions.pkl"
        self.metrics_file = self.save_dir / "metrics.pkl"
        
        # Initialize storage
        if self.predictions_file.exists():
            with open(self.predictions_file, 'rb') as f:
                self.all_predictions = pickle.load(f)
        else:
            self.all_predictions = {'y_true': None, 'runs': {}}
        
        if self.metrics_file.exists():
            with open(self.metrics_file, 'rb') as f:
                self.all_metrics = pickle.load(f)
        else:
            self.all_metrics = {}
    
    def add_run(self, seed: int, y_true: np.ndarray, y_pred: np.ndarray, 
                metrics: Dict[str, float]):
        """Store predictions and metrics for a single run."""
        if self.all_predictions['y_true'] is None:
            self.all_predictions['y_true'] = y_true
        
        self.all_predictions['runs'][seed] = y_pred
        self.all_metrics[seed] = metrics
        
        with open(self.predictions_file, 'wb') as f:
            pickle.dump(self.all_predictions, f)
        
        with open(self.metrics_file, 'wb') as f:
            pickle.dump(self.all_metrics, f)
    
    def get_aggregated_results(self):
        """Compute statistics across runs."""
        if not self.all_predictions['runs']:
            return None
        
        y_true = self.all_predictions['y_true']
        predictions = np.array([self.all_predictions['runs'][seed] 
                               for seed in sorted(self.all_predictions['runs'].keys())])
        
        return {
            'y_true': y_true,
            'y_pred_mean': predictions.mean(axis=0),
            'y_pred_std': predictions.std(axis=0),
            'y_pred_individual': predictions,
            'seeds': sorted(self.all_predictions['runs'].keys())
        }
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Generate summary statistics for metrics."""
        if not self.all_metrics:
            return None
        
        df = pd.DataFrame(self.all_metrics).T
        summary = pd.DataFrame({
            'mean': df.mean(),
            'std': df.std(),
            'min': df.min(),
            'max': df.max()
        })
        return df, summary


def plot_results_over_runs(
    predictor=None,
    data_module=None,
    run_dir=None,
    model_type=None,
    log_metrics=None,
    delay=None,
    seed=None,
    dataset_name=None,
):
    """
    Plot results over multiple runs.
    
    Args:
        predictor: Model predictor instance
        data_module: Data module for the model
        run_dir: Directory for saving results
        model_type: Type of model
        log_metrics: List of metrics to log
        delay: Delay parameter from dataset config
        seed: Current random seed
        dataset_name: Name of the dataset being used
    """
    run_dir = Path(run_dir)
    
    if dataset_name is None:
        dataset_name = "default_dataset"
    
    aggregator = ResultsAggregator(run_dir / "aggregated_results", dataset_name)
    
    # Generate predictions for current run
    y_true, y_pred = _generate_predictions(predictor, data_module)
    
    # Compute metrics for current run
    metrics = _compute_metrics(y_true, y_pred, log_metrics)
    
    # Store results
    aggregator.add_run(seed, y_true, y_pred, metrics)
    
    # Log current run metrics to wandb
    if wandb.run is not None:
        wandb.log({f"{dataset_name}/seed_{seed}/{k}": v for k, v in metrics.items()})
    
    # Check if this is the final run (5 runs completed)
    results = aggregator.get_aggregated_results()
    if results and len(results['seeds']) == 5:
        _create_final_plots(results, aggregator, run_dir, model_type, delay, dataset_name)
        _log_final_metrics(aggregator, dataset_name)


def _generate_predictions(predictor, data_module):
    """Generate predictions using the model."""
    predictor.eval()
    y_true_list, y_pred_list = [], []
    
    with torch.no_grad():
        for batch in data_module.test_dataloader():

            y = batch['y']

            pred = predictor.predict_batch(batch)
            y_pred_list.append(pred.cpu().numpy())
            y_true_list.append(y.numpy() if isinstance(y, torch.Tensor) else y)
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    return y_true.flatten(), y_pred.flatten()


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     log_metrics: List[str]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    
    for metric_name in log_metrics:
        if metric_name.lower() in ['mse', 'mean_squared_error']:
            metrics['MSE'] = np.mean((y_true - y_pred) ** 2)
        elif metric_name.lower() in ['mae', 'mean_absolute_error']:
            metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
        elif metric_name.lower() in ['rmse', 'root_mean_squared_error']:
            metrics['RMSE'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif metric_name.lower() in ['r2', 'r_squared']:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['R2'] = 1 - (ss_res / ss_tot)
        elif metric_name.lower() == 'mape':
            metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return metrics


def _create_final_plots(results: Dict, aggregator: ResultsAggregator, 
                        run_dir: Path, model_type: str, delay: int, 
                        dataset_name: str):
    """Generate individual visualization plots."""
    y_true = results['y_true']
    y_pred_mean = results['y_pred_mean']
    y_pred_std = results['y_pred_std']
    y_pred_individual = results['y_pred_individual']
    seeds = results['seeds']
    time_steps = np.arange(len(y_true))
    
    plots_dir = aggregator.save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Mean predictions with confidence interval
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_steps, y_true, 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
    ax.plot(time_steps, y_pred_mean, 'b-', label='Mean Prediction', linewidth=2)
    ax.fill_between(time_steps, 
                     y_pred_mean - 2 * y_pred_std,
                     y_pred_mean + 2 * y_pred_std,
                     alpha=0.3, label='±2σ Confidence')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Mean Predictions with Uncertainty - {dataset_name}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot1_path = plots_dir / "mean_predictions.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    if wandb.run is not None:
        wandb.log({f"{dataset_name}/mean_predictions": wandb.Image(str(plot1_path))})
    plt.close()
    
    # Plot 2: Individual runs
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_steps, y_true, 'k-', label='Ground Truth', linewidth=2.5, alpha=0.8)
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    for idx, seed in enumerate(seeds):
        ax.plot(time_steps, y_pred_individual[idx], '--', 
                label=f'Seed {seed}', alpha=0.6, color=colors[idx], linewidth=1.5)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Individual Run Predictions - {dataset_name}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2_path = plots_dir / "individual_runs.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    if wandb.run is not None:
        wandb.log({f"{dataset_name}/individual_runs": wandb.Image(str(plot2_path))})
    plt.close()
    
    # Plot 3: Residuals distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    residuals = y_true - y_pred_mean
    ax.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Residual', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Residual Distribution - {dataset_name} (μ={residuals.mean():.3f}, σ={residuals.std():.3f})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot3_path = plots_dir / "residuals_distribution.png"
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    if wandb.run is not None:
        wandb.log({f"{dataset_name}/residuals_distribution": wandb.Image(str(plot3_path))})
    plt.close()
    
    # Plot 4: Metrics comparison across seeds
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df, _ = aggregator.get_metrics_summary()
    metrics_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(f'Metrics Across Seeds - {dataset_name}', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Metrics', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plot4_path = plots_dir / "metrics_comparison.png"
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    if wandb.run is not None:
        wandb.log({f"{dataset_name}/metrics_comparison": wandb.Image(str(plot4_path))})
    plt.close()
    
    # Plot 5: Prediction uncertainty (std) over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_steps, y_pred_std, 'r-', linewidth=2)
    ax.fill_between(time_steps, 0, y_pred_std, alpha=0.3, color='red')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Standard Deviation', fontsize=12)
    ax.set_title(f'Prediction Uncertainty Over Time - {dataset_name}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot5_path = plots_dir / "uncertainty_over_time.png"
    plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
    if wandb.run is not None:
        wandb.log({f"{dataset_name}/uncertainty_over_time": wandb.Image(str(plot5_path))})
    plt.close()
    
    # Plot 6: Scatter plot (predicted vs actual)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred_mean, alpha=0.5, s=20, color='steelblue')
    min_val = min(y_true.min(), y_pred_mean.min())
    max_val = max(y_true.max(), y_pred_mean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Mean Prediction', fontsize=12)
    ax.set_title(f'Predicted vs Actual Values - {dataset_name}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plot6_path = plots_dir / "scatter_pred_vs_actual.png"
    plt.savefig(plot6_path, dpi=300, bbox_inches='tight')
    if wandb.run is not None:
        wandb.log({f"{dataset_name}/scatter_pred_vs_actual": wandb.Image(str(plot6_path))})
    plt.close()


def _log_final_metrics(aggregator: ResultsAggregator, dataset_name: str):
    """Log aggregated metrics to wandb."""
    metrics_df, summary = aggregator.get_metrics_summary()
    
    if wandb.run is not None:
        for metric in summary.index:
            wandb.log({
                f"{dataset_name}/final/{metric}_mean": summary.loc[metric, 'mean'],
                f"{dataset_name}/final/{metric}_std": summary.loc[metric, 'std'],
                f"{dataset_name}/final/{metric}_min": summary.loc[metric, 'min'],
                f"{dataset_name}/final/{metric}_max": summary.loc[metric, 'max']
            })
        
        wandb.log({
            f"{dataset_name}/metrics_table": wandb.Table(dataframe=metrics_df),
            f"{dataset_name}/metrics_summary": wandb.Table(dataframe=summary)
        })
    
    save_dir = aggregator.save_dir
    metrics_df.to_csv(save_dir / "metrics_per_seed.csv")
    summary.to_csv(save_dir / "metrics_summary.csv")