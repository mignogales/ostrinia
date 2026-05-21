import torch
import numpy as np
import os
from sklearn.metrics import classification_report

def save_predictions(predictor, data_module, seed, save_path=''):
    """
    Generate predictions and save graphs across multiple thresholds.
    
    Parameters:
    -----------
    predictor : torch.nn.Module
        Trained prediction model
    data_module : DataModule
        Data module containing test dataloader
    save_path : str
        Base path for saving plots
    """
    # Obtain test dataset
    test_data = data_module.test_dataloader()
    
    # Initialize containers for predictions and ground truth
    y_true = []
    y_pred = []
    
    # Inference mode
    predictor.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            batch = batch.to(predictor.device)
            predictions = predictor.predict_step(batch, i)
            
            y_true.append(batch.target.y[..., 0].cpu().numpy())
            y_pred.append(predictions['y_hat'][..., 0].cpu().numpy())
    
    # Concatenate batches
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # save them to a file
    np.savez_compressed(save_path + f'.npz', y_true=y_true, y_pred=y_pred)
    print(f"Saved predictions and ground truth to {save_path}.npz")