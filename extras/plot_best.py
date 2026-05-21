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
    
    def get_best_run(self):
        """Get the best run based on lowest MAE."""
        if not self.all_metrics:
            return None
        
        # Find seed with lowest MAE
        best_seed = None
        best_mae = float('inf')
        
        for seed, metrics in self.all_metrics.items():
            if 'MAE' in metrics and metrics['MAE'] < best_mae:
                best_mae = metrics['MAE']
                best_seed = seed
        
        if best_seed is None:
            # If MAE not available, use MSE as fallback
            for seed, metrics in self.all_metrics.items():
                if 'MSE' in metrics:
                    mse = metrics['MSE']
                    if mse < best_mae:
                        best_mae = mse
                        best_seed = seed
        
        if best_seed is not None:
            return {
                'seed': best_seed,
                'y_true': self.all_predictions['y_true'],
                'y_pred': self.all_predictions['runs'][best_seed],
                'metrics': self.all_metrics[best_seed]
            }
        return None
    
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
    Plot results for the best run only (lowest MAE).
    
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
    
    # Compute metrics for current run (ensure MAE is calculated)
    metrics = _compute_metrics(y_true, y_pred, log_metrics)
    
    # Store results
    aggregator.add_run(seed, y_true, y_pred, metrics)
    
    # Log current run metrics to wandb
    if wandb.run is not None:
        wandb.log({f"{dataset_name}/seed_{seed}/{k}": v for k, v in metrics.items()})
    
    # Check if this is the final run (5 runs completed)
    if len(aggregator.all_predictions['runs']) == 5:
        best_run = aggregator.get_best_run()
        if best_run:
            _create_best_run_plot(best_run, aggregator, run_dir, model_type, 
                                 delay, dataset_name)
            _log_final_metrics(aggregator, best_run, dataset_name)


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
    """Compute evaluation metrics. Always computes MAE for best run selection."""
    metrics = {}
    
    # Always compute MAE for best run selection
    metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
    
    for metric_name in log_metrics:
        if metric_name.lower() in ['mse', 'mean_squared_error']:
            metrics['MSE'] = np.mean((y_true - y_pred) ** 2)
        elif metric_name.lower() in ['mae', 'mean_absolute_error']:
            # Already computed above
            pass
        elif metric_name.lower() in ['rmse', 'root_mean_squared_error']:
            metrics['RMSE'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif metric_name.lower() in ['r2', 'r_squared']:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['R2'] = 1 - (ss_res / ss_tot)
        elif metric_name.lower() == 'mape':
            metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return metrics


def _create_best_run_plot(best_run: Dict, aggregator: ResultsAggregator, 
                          run_dir: Path, model_type: str, delay: int, 
                          dataset_name: str):
    """Generate comprehensive visualization for the best run only."""
    y_true = best_run['y_true']
    y_pred = best_run['y_pred']
    best_seed = best_run['seed']
    metrics = best_run['metrics']
    time_steps = np.arange(len(y_true))
    
    plots_dir = aggregator.save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Subplot 1: Time series prediction
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(time_steps, y_true, 'k-', label='Ground Truth', linewidth=2, alpha=0.8)
    ax1.plot(time_steps, y_pred, 'b-', label=f'Prediction (Seed {best_seed})', linewidth=2)
    ax1.set_xlabel('Time Step', fontsize=11)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title(f'Best Run Prediction (MAE={metrics.get("MAE", 0):.4f})', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Scatter plot
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction')
    ax2.set_xlabel('Ground Truth', fontsize=11)
    ax2.set_ylabel('Prediction', fontsize=11)
    ax2.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Subplot 3: Residuals over time
    ax3 = plt.subplot(2, 3, 3)
    residuals = y_true - y_pred
    ax3.plot(time_steps, residuals, 'g-', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax3.fill_between(time_steps, 0, residuals, alpha=0.3, color='green')
    ax3.set_xlabel('Time Step', fontsize=11)
    ax3.set_ylabel('Residual', fontsize=11)
    ax3.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Residuals histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax4.set_xlabel('Residual', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title(f'Residual Distribution (μ={residuals.mean():.3f}, σ={residuals.std():.3f})',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Subplot 5: Metrics comparison across all runs
    ax5 = plt.subplot(2, 3, 5)
    metrics_df, _ = aggregator.get_metrics_summary()
    if metrics_df is not None and not metrics_df.empty:
        metrics_df.plot(kind='bar', ax=ax5, width=0.8)
        # Highlight best run
        best_idx = list(metrics_df.index).index(best_seed)
        for container in ax5.containers:
            container[best_idx].set_color('red')
            container[best_idx].set_alpha(0.8)
        ax5.set_xlabel('Seed', fontsize=11)
        ax5.set_ylabel('Metric Value', fontsize=11)
        ax5.set_title(f'All Runs Comparison (Best: Seed {best_seed})', 
                      fontsize=12, fontweight='bold')
        ax5.legend(title='Metrics', fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
    
    # Subplot 6: Absolute error distribution
    ax6 = plt.subplot(2, 3, 6)
    abs_errors = np.abs(residuals)
    ax6.hist(abs_errors, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax6.axvline(metrics.get('MAE', np.mean(abs_errors)), color='red', 
                linestyle='--', linewidth=2, label=f'MAE={metrics.get("MAE", 0):.4f}')
    ax6.set_xlabel('Absolute Error', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Best Run Analysis - {dataset_name} (Model: {model_type}, Delay: {delay})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_path = plots_dir / "best_run_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    if wandb.run is not None:
        wandb.log({f"{dataset_name}/best_run_analysis": wandb.Image(str(plot_path))})
    plt.close()
    
    # Save best run information
    best_run_info = {
        'seed': best_seed,
        'metrics': metrics,
        'residuals_mean': residuals.mean(),
        'residuals_std': residuals.std()
    }
    with open(plots_dir / 'best_run_info.pkl', 'wb') as f:
        pickle.dump(best_run_info, f)


def _log_final_metrics(aggregator: ResultsAggregator, best_run: Dict, 
                       dataset_name: str):
    """Log aggregated metrics and best run information to wandb."""
    metrics_df, summary = aggregator.get_metrics_summary()
    
    if wandb.run is not None:
        # Log best run metrics
        wandb.log({
            f"{dataset_name}/best_run/seed": best_run['seed'],
            **{f"{dataset_name}/best_run/{k}": v 
               for k, v in best_run['metrics'].items()}
        })
        
        # Log summary statistics
        for metric in summary.index:
            wandb.log({
                f"{dataset_name}/final/{metric}_mean": summary.loc[metric, 'mean'],
                f"{dataset_name}/final/{metric}_std": summary.loc[metric, 'std'],
                f"{dataset_name}/final/{metric}_min": summary.loc[metric, 'min'],
                f"{dataset_name}/final/{metric}_max": summary.loc[metric, 'max'],
                f"{dataset_name}/final/{metric}_best": best_run['metrics'].get(metric, 0)
            })
        
        wandb.log({
            f"{dataset_name}/metrics_table": wandb.Table(dataframe=metrics_df),
            f"{dataset_name}/metrics_summary": wandb.Table(dataframe=summary)
        })
    
    save_dir = aggregator.save_dir
    metrics_df.to_csv(save_dir / "metrics_per_seed.csv")
    summary.to_csv(save_dir / "metrics_summary.csv")
    
    # Save best run details
    with open(save_dir / "best_run_details.txt", 'w') as f:
        f.write(f"Best Run: Seed {best_run['seed']}\n")
        f.write(f"{'='*40}\n")
        for metric, value in best_run['metrics'].items():
            f.write(f"{metric}: {value:.6f}\n")