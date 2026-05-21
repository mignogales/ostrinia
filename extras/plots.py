import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Callable
from tsl.data import StaticBatch

def get_dl_predictions(predictor, data_module):
    """
    Get predictions from a deep learning model that processes batched data.
    
    Args:
        predictor: Deep learning model with predict_step method
        data_module: Data module with test_dataloader method
    
    Returns:
        Dictionary containing y (true values), y_hat (predictions), inputs_plot (input data)
    """
    predictor.eval()
    predictor.to('cuda')
    
    # Get test data
    test_data = data_module.test_dataloader()
    
    inputs = []
    pred = []
    # Get predictions
    for i, batch in enumerate(test_data):
        batch = batch.to(predictor.device)
        predictions = predictor.predict_step(batch, i)
        pred.append(predictions)
        inputs.append(batch)
    
    # Stack list of tensors to a single tensor
    y, y_hat = [], []
    inputs_plot = []
    if predictor.model_kwargs['horizon'] != pred[0]['y_hat'].shape[-1]:
        flag_increment = []
    for element in pred:
        y.append(element['y'].cpu().numpy())
        y_hat.append(element['y_hat'].cpu().numpy())
        # if predictor.model_kwargs['horizon'] != element['y_hat'].shape[-1]:
        #     flag_increment.append(element['y_hat'][:, :, :, :predictor.model_kwargs['horizon']].cpu().numpy())
    for element in inputs:
        inputs_plot.append(element['x'].cpu().numpy())
        if "second_target" in element.keys():
            flag_increment.append(element['second_target'].cpu().numpy())

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    inputs_plot = np.concatenate(inputs_plot)

    if predictor.model_kwargs['horizon'] != y_hat.shape[-1]:
        flag_increment = np.concatenate(flag_increment)
    
    # Get only one feature dimension if present
    if predictor.model_kwargs['horizon'] != y_hat.shape[-1]:
        flag_increment_hat = y_hat[:, :, :, predictor.model_kwargs['horizon']:]
        # get the class with the highest probability
        flag_increment_hat = np.argmax(flag_increment_hat, axis=-1)
        # add a new axis to match the shape of y and y_hat
        flag_increment_hat = flag_increment_hat[:, :, :, np.newaxis]
    
    y = y[:, :, :, 0]
    y_hat = y_hat[:, :, :, 0]

    return {
        'y': y,
        'y_hat': y_hat,
        'inputs_plot': inputs_plot,
        'flag_increment': flag_increment if 'flag_increment' in locals() else None,
        'flag_increment_hat': flag_increment_hat if 'flag_increment_hat' in locals() else None
    }

def get_arimax_predictions(predictor, data_loader):
    """
    Get predictions from an ARIMAX model using a PyTorch DataLoader.
    
    Args:
        predictor: ARIMAX model with predict method
        data_loader: PyTorch DataLoader providing batches of data
    
    Returns:
        Dictionary containing y (true values), y_hat (predictions), inputs_plot (input features)
    """
    # Lists to collect results
    all_y = []
    all_y_hat = []
    all_x = []
    
    # Iterate through the data loader
    for batch in data_loader:
        # Extract data from batch (handling both dictionary and tuple formats)
        if isinstance(batch, StaticBatch):
            x = batch.input['x'].cpu().numpy()
            u = batch.input.get('u', None)
            if u is not None:
                u = u.cpu().numpy()
                # concatenate x and u along the feature dimension
                x = np.concatenate([x, u], axis=-1)
            y = batch.target['y'].cpu().numpy()

        # Generate predictions 
        batch_y_hat = predictor.predict(y, exog=x, steps=predictor.delay + predictor.horizon - 1)
        
        # Store batch results
        all_y.append(y)
        all_y_hat.append(batch_y_hat)
        all_x.append(x)
    
    # Concatenate all batches
    y = np.concatenate(all_y, axis=0)
    y_hat = np.concatenate(all_y_hat, axis=0)
    x = np.concatenate(all_x, axis=0)
    
    # Reshape if needed
    if len(y_hat.shape) == 2:
        y_hat = y_hat[:, np.newaxis, :]  # Add horizon dimension
    
    return {
        'y': y,
        'y_hat': y_hat,
        'inputs_plot': x
    }

def plot_predictions_common(prediction_data, run_dir, important_nodes=None, log_metrics=None, delay=14, wandb_run=None):
    """
    Common plotting function for predictions.
    
    Args:
        prediction_data: Dictionary containing y, y_hat, and inputs_plot
        run_dir: Directory to save plots
        important_nodes: List of important node indices
        log_metrics: Metrics to log
        delay: Delay parameter
        wandb_run: WandB run object for logging
    """
    # Create directory for plots
    plot_dir = Path(run_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    y = prediction_data['y']
    y_hat = prediction_data['y_hat']
    inputs_plot = prediction_data.get('inputs_plot', None)
    
    # Take nodes which y[:, 0, n] sum is not zero
    non_zero_nodes = [i for i in range(y.shape[2]) if np.sum(y[:, 0, i]) != 0]
    
    # Important nodes (use provided or default)
    if important_nodes is None:
        important_nodes = [4, 9, 10]
    
    horizon = y.shape[1]
    
    if horizon > 6:
        horizons = range(0, horizon, 2)
    else:
        horizons = range(horizon)
    
    # Plot predictions
    index_diffs = []
    index_diffs_2 = []
    important_index_diffs = []
    important_index_diffs_2 = []
    
    for h in horizons:
        for n in non_zero_nodes:
            plt.figure(figsize=(10, 5))

            # Create time indexes. The signal should spawn from the first non zero value to the first zero value after that
            # For the first index, find the first non-zero value
            start_index = np.where(y[:, h, n] != 0)[0][0] if np.any(y[:, h, n] != 0) else 0
            # For the last index, find the first zero value after the first non-zero value
            end_index = np.where(y[start_index:, h, n] == 0)[0]

            if end_index.size > 0:
                end_index = end_index[0] + start_index
            else:
                end_index = y.shape[0]

            if start_index != 0:
                start_index = start_index - 10
                delayed_index_start = start_index - h - delay - 1
                delayed_index_end = end_index - h - delay - 1
                adjust = 0
            else:
                delayed_index_start = 0
                delayed_index_end = end_index - h - delay - 1
                adjust = delay + 1

            # Compute steepest slope of y and y_hat
            index_max_slope_y = np.argmax(slope_max(y[start_index:end_index, h, n], delay)) + delay//2 + start_index
            index_max_slope_y_hat = np.argmax(slope_max(y_hat[start_index:end_index, h, n], delay)) + delay//2 + start_index

            # Compute the second biggest slope from 30 days after the biggest slope
            if index_max_slope_y + 30 < end_index - start_index:
                index_max_slope_y_2 = np.argmax(slope_max(y[start_index + index_max_slope_y + 30:end_index, h, n], delay)) + index_max_slope_y + 30
            else:
                index_max_slope_y_2 = index_max_slope_y

            if index_max_slope_y_hat + 30 < end_index - start_index and len(y_hat[start_index + index_max_slope_y_hat + 30:end_index, h, n]) > delay:
                index_max_slope_y_hat_2 = np.argmax(slope_max(y_hat[start_index + index_max_slope_y_hat + 30:end_index, h, n], delay)) + index_max_slope_y_hat + 30
            else:
                index_max_slope_y_hat_2 = index_max_slope_y_hat

            # Plot predictions with filtering
            N = 4
            y_hat_filtered = np.convolve(y_hat[start_index-N//2:end_index, h, n], np.ones(N)/N**2, mode='valid')
            factor_conv = y_hat_filtered.max() / y[start_index:end_index, h, n].max() if y[start_index:end_index, h, n].max() != 0 else 1

            factor = y_hat[start_index:end_index, h, n].max() / y[start_index:end_index, h, n].max() if y[start_index:end_index, h, n].max() != 0 else 1
            if factor < 0.:
                factor = 1.
            
            plt.plot(range(start_index, end_index), y[start_index:end_index, h, n], label='True')
            plt.plot(range(start_index, end_index), y_hat[start_index:end_index, h, n]/factor, label='Predicted')
            
            # Add dotted plot for inputs if available
            if inputs_plot is not None:
                plt.plot(range(start_index + adjust, end_index), y[delayed_index_start:delayed_index_end, 0, n], label='Input', linestyle='dotted')
            
            # Add filtered moved one step to the left
            plt.plot(range(start_index + adjust, end_index + 1 - N//2), y_hat_filtered/factor_conv, label='Predicted Filtered', linestyle='dashdot')
            
            plt.xlim(start_index, end_index)
            # Add vertical lines at the index of the steepest slope
            plt.axvline(x=index_max_slope_y, color='red', linestyle='--', label='Max Slope True')
            plt.axvline(x=index_max_slope_y_hat, color='green', linestyle='--', linewidth=4, label='Max Slope Predicted')

            # Add vertical lines at the second biggest slope
            plt.axvline(x=index_max_slope_y_2, color='orange', linestyle='--', label='Second Max Slope True')
            plt.axvline(x=index_max_slope_y_hat_2, color='purple', linestyle='--', linewidth=4, label='Second Max Slope Predicted')

            plt.xticks(range(start_index, end_index, 10))
            plt.grid()
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()

            # Log difference between predicted and true index of steepest slope
            if wandb_run is not None:
                index_diffs.append(abs((index_max_slope_y) - (index_max_slope_y_hat)))
                index_diffs_2.append(abs((index_max_slope_y_2) - (index_max_slope_y_hat_2)))

                if n in important_nodes:
                    important_index_diffs.append(abs((index_max_slope_y) - (index_max_slope_y_hat)))
                    important_index_diffs_2.append(abs((index_max_slope_y_2) - (index_max_slope_y_hat_2)))

            if n in important_nodes:
                plt.title(f'Important Node {n} - Horizon {h} - Delay {delay} - Factor: {factor:.2f}')
                plt.tight_layout()
                plt.savefig(plot_dir / f'important_node_{n}_horizon_{h}.png')
            else:
                plt.title(f'Node {n} - Horizon {h} - Delay {delay} - Factor: {factor:.2f}')
                plt.tight_layout()
                plt.savefig(plot_dir / f'node_{n}_horizon_{h}.png')
            print(f'Saved plot for Node {n} - Horizon {h}')
            plt.close()

            # Repeat for the whole time range
            factor = y_hat[:, h, n].max() / y[:, h, n].max() if y[:, h, n].max() != 0 else 1
            if factor < 0.:
                factor = 1.

            # Compute filtered prediction
            N = 4
            y_hat_filtered = np.convolve(y_hat[:, h, n], np.ones(N)/N**2, mode='valid')[N//2:]
            factor_conv = y_hat_filtered.max() / y[:, h, n].max() if y[:, h, n].max() != 0 else 1

            # Compute the slope for the whole time range
            index_max_slope_y = np.argmax(slope_max(y[:, h, n], delay)) + delay//2
            index_max_slope_y_hat = np.argmax(slope_max(y_hat[:, h, n], delay)) + delay//2

            if y_hat.shape[0] - index_max_slope_y_hat - 30 > 0:
                index_max_slope_y_2 = np.argmax(slope_max(y[index_max_slope_y + 30:, h, n], delay)) + index_max_slope_y + 30
                index_max_slope_y_hat_2 = np.argmax(slope_max(y_hat[index_max_slope_y_hat + 30:, h, n], delay)) + index_max_slope_y_hat + 30

            plt.figure(figsize=(10, 5))
            plt.plot(range(y.shape[0]), y[:, h, n], label='True')
            plt.plot(range(y.shape[0]), y_hat[:, h, n]/factor, label='Predicted')
            
            if inputs_plot is not None:
                plt.plot(range(delay + 1, y.shape[0]), y[:-delay-1, 0, n], label='Input', linestyle='dotted')
                
            plt.plot(range(N, y_hat_filtered.shape[0] + N), y_hat_filtered/factor_conv, label='Predicted Filtered', linestyle='dashdot')
            plt.axvline(x=index_max_slope_y, color='red', linestyle='--', label='Max Slope True')
            plt.axvline(x=index_max_slope_y_hat, color='green', linestyle='--', label='Max Slope Predicted')
            
            if y.shape[0] - index_max_slope_y_hat - 30 > 0:
                plt.axvline(x=index_max_slope_y_2, color='orange', linestyle='--', label='Second Max Slope True')
                plt.axvline(x=index_max_slope_y_hat_2, color='purple', linestyle='--', label='Second Max Slope Predicted')
                
            plt.xlim(0, y.shape[0])
            plt.xticks(range(0, y.shape[0], 10))
            plt.grid()
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            
            if n in important_nodes:    
                plt.title(f'Important Node {n} - Horizon {h} - Delay {delay} - Factor: {factor:.2f}')
                plt.tight_layout()
                plt.savefig(plot_dir / f'important_node_{n}_horizon_{h}_full.png')
            else:
                plt.title(f'Node {n} - Horizon {h} - Delay {delay} - Factor: {factor:.2f}')
                plt.tight_layout()
                plt.savefig(plot_dir / f'node_{n}_horizon_{h}_full.png')
            print(f'Saved full plot for Node {n} - Horizon {h}')
            plt.close()

    if "flag_increment" in prediction_data.keys():
        plot_binary_flags(prediction_data, run_dir, important_nodes=important_nodes, delay=delay)

    print(f'Plots saved to {plot_dir}')
    
    # Return metrics for logging
    return {
        'index_diffs': np.mean(index_diffs) if index_diffs else None,
        'index_diffs_2': np.mean(index_diffs_2) if index_diffs_2 else None,
        'important_index_diffs': np.mean(important_index_diffs) if important_index_diffs else None,
        'important_index_diffs_2': np.mean(important_index_diffs_2) if important_index_diffs_2 else None
    }

def plot_predictions_test(predictor, data_module, run_dir, model_type='dl', arimax_test_data=None, 
                         important_nodes=None, log_metrics=None, delay=14, wandb_run=None):
    """
    Plot predictions on the test set for different model types.
    
    Args:
        predictor: Model (DL or ARIMAX)
        data_module: Data module for DL models (ignored for ARIMAX)
        run_dir: Directory to save plots
        model_type: 'dl' for deep learning models, 'arimax' for ARIMAX models
        arimax_test_data: Test data for ARIMAX models (ignored for DL)
        important_nodes: List of important node indices
        log_metrics: Metrics to log
        delay: Delay parameter
        wandb_run: WandB run object for logging
    """

    if arimax_test_data is None and model_type.lower() == 'arimax':
        arimax_test_data = data_module.test_dataloader()

    # Get predictions based on model type
    if model_type.lower() == 'dl':
        prediction_data = get_dl_predictions(predictor, data_module)
    elif model_type.lower() == 'arimax':
        prediction_data = get_arimax_predictions(predictor, arimax_test_data)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'dl' or 'arimax'")
    
    # Plot predictions using common function
    metrics = plot_predictions_common(
        prediction_data, run_dir, important_nodes=important_nodes, 
        log_metrics=log_metrics, delay=delay, wandb_run=wandb_run
    )
    
    # Log metrics if needed
    if wandb_run is not None:
        wandb_run.log(metrics)

def slope_max(y, delay):
    """Calculate the maximum slope."""
    y = ((y[delay:] - y[delay//2:-delay//2]) / (y[delay//2:-delay//2] - y[:-delay]))
    # Change nan to zero
    y[np.isnan(y)] = 0
    return y


def plot_binary_flags(prediction_data, run_dir, important_nodes=None, delay=14):
    """
    Plot binary flags (flag_increment and flag_increment_hat) alongside real signal.
    
    Args:
        prediction_data: Dictionary containing y, flag_increment, flag_increment_hat
        run_dir: Directory to save plots
        important_nodes: List of important node indices
        delay: Delay parameter
    """
    # Create directory for binary flag plots
    plot_dir = Path(run_dir) / 'plots' / 'binary_flags'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    y = prediction_data['y']
    flag_increment = prediction_data['flag_increment']
    flag_increment_hat = prediction_data['flag_increment_hat']
    
    # Take nodes which y[:, 0, n] sum is not zero
    non_zero_nodes = [i for i in range(y.shape[2]) if np.sum(y[:, 0, i]) != 0]
    
    # Important nodes (use provided or default)
    if important_nodes is None:
        important_nodes = [4, 9, 10]
    
    horizon = y.shape[1]
    
    if horizon > 6:
        horizons = range(0, horizon, 2)
    else:
        horizons = range(horizon)
    
    # Plot binary flags for each horizon and node
    for h in horizons:
        for n in non_zero_nodes:
            # Find start and end indices (same logic as original function)
            start_index = np.where(y[:, h, n] != 0)[0][0] if np.any(y[:, h, n] != 0) else 0
            end_index = np.where(y[start_index:, h, n] == 0)[0]
            
            if end_index.size > 0:
                end_index = end_index[0] + start_index
            else:
                end_index = y.shape[0]
            
            if start_index != 0:
                start_index = start_index - 10
            
            # Extract data for the time range
            time_indices = np.arange(start_index, end_index)
            y_signal = y[start_index:end_index, h, n]
            true_flags = flag_increment[start_index:end_index, h, n].astype(int)
            pred_flags = flag_increment_hat[start_index:end_index, h, n].astype(int)[:,0]
            
            # Calculate metrics
            accuracy = np.mean(true_flags == pred_flags)
            precision = np.sum((pred_flags == 1) & (true_flags == 1)) / np.sum(pred_flags == 1) if np.sum(pred_flags == 1) > 0 else 0
            recall = np.sum((pred_flags == 1) & (true_flags == 1)) / np.sum(true_flags == 1) if np.sum(true_flags == 1) > 0 else 0
            
            # Create subplot figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Top subplot: Real signal
            ax1.plot(time_indices, y_signal, 'b-', label='Real Signal (y)', linewidth=2)
            ax1.set_ylabel('Signal Value')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_title(f'Real Signal - Node {n}, Horizon {h}')
            
            # Bottom subplot: Binary flags
            ax2.step(time_indices, true_flags, 'r-', where='post', label='True Flag', linewidth=2)
            ax2.step(time_indices, pred_flags, 'g--', where='post', label='Predicted Flag', linewidth=2)
            
            # # Add filled areas for better visualization
            # for i in range(len(time_indices)):
            #     if true_flags[i] == 1:
            #         ax2.axvspan(time_indices[i], time_indices[i]+1 if i < len(time_indices)-1 else time_indices[i]+1, 
            #                    alpha=0.2, color='red', zorder=0)
            #     if pred_flags[i] == 1:
            #         ax2.axvspan(time_indices[i], time_indices[i]+1 if i < len(time_indices)-1 else time_indices[i]+1, 
            #                    alpha=0.2, color='green', zorder=1)

            # Add filled areas for full range - only when flags are active
            active_indices = np.where((true_flags == 1) | (pred_flags == 1))[0]
            
            for idx in active_indices:
                if true_flags[idx] == pred_flags[idx]:  # Agreement (both 1)
                    ax2.axvspan(idx + start_index, idx+1 + start_index, alpha=0.2, color='green', zorder=0)
                else:  # Disagreement (one is 1, other is 0)
                    ax2.axvspan(idx + start_index, idx+1 + start_index, alpha=0.2, color='red', zorder=1)


            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Flag Value')
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_title(f'Binary Flags - Acc: {accuracy:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}')
            
            plt.tight_layout()
            
            # Save plot based on importance
            if n in important_nodes:
                plt.savefig(plot_dir / f'importantnode{n}horizon{h}_flags.png', dpi=300, bbox_inches='tight')
                print(f'Saved binary flag plot for Important Node {n} - Horizon {h}')
            else:
                plt.savefig(plot_dir / f'node{n}horizon{h}_flags.png', dpi=300, bbox_inches='tight')
                print(f'Saved binary flag plot for Node {n} - Horizon {h}')
            
            plt.close()
            
            # Create full range plot
            time_indices_full = np.arange(y.shape[0])
            y_signal_full = y[:, h, n]
            true_flags_full = flag_increment[:, h, n].astype(int)
            pred_flags_full = flag_increment_hat[:, h, n].astype(int)[:,0]
            
            # Calculate full range metrics
            accuracy_full = np.mean(true_flags_full == pred_flags_full)
            precision_full = np.sum((pred_flags_full == 1) & (true_flags_full == 1)) / np.sum(pred_flags_full == 1) if np.sum(pred_flags_full == 1) > 0 else 0
            recall_full = np.sum((pred_flags_full == 1) & (true_flags_full == 1)) / np.sum(true_flags_full == 1) if np.sum(true_flags_full == 1) > 0 else 0
            
            # Create full range subplot figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Top subplot: Real signal
            ax1.plot(time_indices_full, y_signal_full, 'b-', label='Real Signal (y)', linewidth=1.5)
            ax1.set_ylabel('Signal Value')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_title(f'Full Range Real Signal - Node {n}, Horizon {h}')
            
            # Bottom subplot: Binary flags
            ax2.step(time_indices_full, true_flags_full, 'r-', where='post', label='True Flag', linewidth=1.5)
            ax2.step(time_indices_full, pred_flags_full, 'g--', where='post', label='Predicted Flag', linewidth=1.5)
            
            # Add filled areas for full range - only when flags are active
            active_indices = np.where((true_flags_full == 1) | (pred_flags_full == 1))[0]
            
            for idx in active_indices:
                if true_flags_full[idx] == pred_flags_full[idx]:  # Agreement (both 1)
                    ax2.axvspan(idx, idx+1, alpha=0.2, color='green', zorder=0)
                else:  # Disagreement (one is 1, other is 0)
                    ax2.axvspan(idx, idx+1, alpha=0.2, color='red', zorder=1)
            
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Flag Value')
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_title(f'Full Range Binary Flags - Acc: {accuracy_full:.3f}, Prec: {precision_full:.3f}, Rec: {recall_full:.3f}')
            
            plt.tight_layout()
            
            # Save full range plot
            if n in important_nodes:
                plt.savefig(plot_dir / f'importantnode{n}horizon{h}_flags_full.png', dpi=300, bbox_inches='tight')
                print(f'Saved full binary flag plot for Important Node {n} - Horizon {h}')
            else:
                plt.savefig(plot_dir / f'node{n}horizon{h}_flags_full.png', dpi=300, bbox_inches='tight')
                print(f'Saved full binary flag plot for Node {n} - Horizon {h}')
            
            plt.close()