
import numpy as np

def plot_predictions_test(predictor, data_module, run_dir, log_metrics=None, delay=14, wandb_run=None):
    """
    Plot predictions on the test set.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Create directory for plots
    plot_dir = Path(run_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

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

    # stack list of tensors to a single tensor
    y, y_hat = [], []
    inputs_plot = []
    for element in pred:
        y.append(element['y'].cpu().numpy())
        y_hat.append(element['y_hat'].cpu().numpy())
    for element in inputs:
        inputs_plot.append(element['x'].cpu().numpy())
    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    inputs_plot = np.concatenate(inputs_plot)

    # take nodes which y[:, 0, n, :] sum is not zero
    non_zero_nodes = [i for i in range(y.shape[2]) if np.sum(y[:, 0, i, :]) != 0]

    # important nodes are the 3 ones with the highest sum across all time steps
    important_nodes = [4, 9, 10]

    # get indices of non-zero nodes
    horizon = y.shape[1]

    # get only one node
    y = y[:, :, :, 0]
    y_hat = y_hat[:, :, :, 0]

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

            # create time indexes. the signal should spawn from the first non zero value to the first zero value after that
            # for the first index, find the first non-zero value
            start_index = np.where(y[:, h, n] != 0)[0][0]
            # for the last index, find the first zero value after the first non-zero value
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

            # compute steepest slope of y and y_hat:
            #     - slope = (y_{t+h} - y_{t})/(y_{t} - y_{t-h}) with h = delay

            index_max_slope_y = np.argmax(slope_max(y[start_index:end_index, h, n], delay)) + delay//2 + start_index

            index_max_slope_y_hat = np.argmax(slope_max(y_hat[start_index:end_index, h, n], delay)) + delay//2 + start_index

            # compute the second biggest slope from 30 days after the biggest slope
            if index_max_slope_y + 30 < end_index - start_index:
                index_max_slope_y_2 = np.argmax(slope_max(y[start_index + index_max_slope_y + 30:end_index, h, n], delay)) + index_max_slope_y + 30
            else:
                index_max_slope_y_2 = index_max_slope_y

            if index_max_slope_y_hat + 30 < end_index - start_index and len(y_hat[start_index + index_max_slope_y_hat + 30:end_index, h, n]) > delay:
                index_max_slope_y_hat_2 = np.argmax(slope_max(y_hat[start_index + index_max_slope_y_hat + 30:end_index, h, n], delay)) + index_max_slope_y_hat + 30
            else:
                index_max_slope_y_hat_2 = index_max_slope_y_hat

            # plot also the prediction filtered by a size N moving average
            N = 4
            y_hat_filtered = np.convolve(y_hat[start_index-N//2:end_index, h, n], np.ones(N)/N**2, mode='valid')
            factor_conv = y_hat_filtered.max() / y[start_index:end_index, h, n].max()

            factor = y_hat[start_index:end_index, h, n].max() / y[start_index:end_index, h, n].max()
            if factor < 0.:
                factor = 1.
            plt.plot(range(start_index, end_index), y[start_index:end_index, h, n], label='True')
            plt.plot(range(start_index, end_index), y_hat[start_index:end_index, h, n]/factor, label='Predicted')
            # add a dotted plot for inputs
            plt.plot(range(start_index + adjust, end_index), y[delayed_index_start:delayed_index_end, 0, n], label='Input', linestyle='dotted')
            # add filtered moved one step to the left
            plt.plot(range(start_index + adjust, end_index + 1 - N//2), y_hat_filtered/factor_conv, label='Predicted Filtered', linestyle='dashdot')
            
            plt.xlim(start_index, end_index)
            # add vertical lines at the index of the steepest slope
            plt.axvline(x= index_max_slope_y, color='red', linestyle='--', label='Max Slope True')
            plt.axvline(x= index_max_slope_y_hat, color='green', linestyle='--', linewidth=4, label='Max Slope Predicted')

            # add vertical lines at the second biggest slope
            plt.axvline(x= index_max_slope_y_2, color='orange', linestyle='--', label='Second Max Slope True')
            plt.axvline(x= index_max_slope_y_hat_2, color='purple', linestyle='--',  linewidth=4, label='Second Max Slope Predicted')

            # plt.ylim(np.min(y[start_index:end_index, h, n]) - 1 , np.max(y[start_index:end_index, h, n]) + 1)
            plt.xticks(range(start_index, end_index, 10))
            plt.grid()
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()

            # log difference between predicted and true index of steepest slope
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

            # repeat for the whole time range
            factor = y_hat[:, h, n].max() / y[:, h, n].max()

            if factor < 0.:
                factor = 1.

            # compute filtered prediction
            N = 4
            y_hat_filtered = np.convolve(y_hat[:, h, n], np.ones(N)/N**2, mode='valid')[N//2:]
            factor_conv = y_hat_filtered.max() / y[:, h, n].max()

            # compute the slope for the whole time range
            index_max_slope_y = np.argmax(slope_max(y[:, h, n], delay)) + delay//2
            index_max_slope_y_hat = np.argmax(slope_max(y_hat[:, h, n], delay)) + delay//2

            if y_hat.shape[0] - index_max_slope_y_hat - 30 > 0:
                index_max_slope_y_2 = np.argmax(slope_max(y[index_max_slope_y + 30:, h, n], delay)) + index_max_slope_y + 30
                index_max_slope_y_hat_2 = np.argmax(slope_max(y_hat[index_max_slope_y_hat + 30:, h, n], delay)) + index_max_slope_y_hat + 30

            plt.figure(figsize=(10, 5))
            plt.plot(range(y.shape[0]), y[:, h, n], label='True')
            plt.plot(range(y.shape[0]), y_hat[:, h, n]/factor, label='Predicted')
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

        if wandb_run is not None:
            wandb_run.log({
                'index_diffs': np.mean(index_diffs),
                'index_diffs_2': np.mean(index_diffs_2),
                'important_index_diffs': np.mean(important_index_diffs),
                'important_index_diffs_2': np.mean(important_index_diffs_2),
            })


    print(f'Plots saved to {plot_dir}')



def slope_max(y, delay):
    y = ((y[delay:] - y[delay//2:-delay//2]) / (y[delay//2:-delay//2] - y[:-delay]))

    # change nan to zero
    y[np.isnan(y)] = 0

    return y
