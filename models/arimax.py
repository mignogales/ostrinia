import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Optional, Tuple
import pytorch_lightning as pl
import torch

@dataclass
class ARIMAXSpec:
    order: Tuple[int, int, int]
    trend: str = "c"  # constant by default
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False

def fit_arimax(y_tr: pd.Series, X_tr: Optional[np.ndarray], spec: ARIMAXSpec, n_nodes: Optional[int] = 1):
    models = []
    for i in range(n_nodes):
        mod = sm.tsa.statespace.SARIMAX(
            y_tr[:,i],
            exog=X_tr[:,i],
            order=(spec.p, spec.d, spec.q),
            seasonal_order=(0, 0, 0, 0),
            trend=spec.trend,
            enforce_stationarity=spec.enforce_stationarity,
            enforce_invertibility=spec.enforce_invertibility,
        )
        res = mod.fit(disp=False)
        models.append(res)
    return models


class ARIMAXWrapper(pl.LightningModule):
    def __init__(self,         
                model_class,  # Now a list of ARIMAX models, one per node
                n_nodes,
                model_kwargs,
                optim_class,
                optim_kwargs,
                loss_fn=None,
                metrics=None,
                scheduler_class=None,
                scheduler_kwargs=None,
                horizon=None,
                delay=None,):
        super().__init__()

        # Now model_class is a list of fitted ARIMAX models
        self.arimax_models = model_class
        self.num_nodes = n_nodes
        self.loss_fn = loss_fn or torch.nn.MSELoss()
        self.metrics = metrics or {}
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.horizon = horizon
        self.delay = delay
    
    def forward(self, y, exog=None, steps=1):
        """
        Forward pass for prediction using multiple ARIMAX models, one per node
        
        Args:
            y: The input time series data with shape [batch, sequence_length, nodes]
            exog: Optional exogenous variables with shape [batch, features, nodes]
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values for all nodes with shape [batch, steps, nodes]
        """
        forecasts = []
        
        # Loop through each node and apply its specific ARIMAX model
        for node_idx in range(self.num_nodes):
            # Extract data for this specific node
            node_y = y[:, node_idx]
            
            if isinstance(node_y, torch.Tensor):
                node_y = node_y.cpu().numpy()
            
            # Extract exogenous variables for this node if provided
            node_exog = None
            if exog is not None:
                node_exog = exog[:, node_idx] 
                if isinstance(node_exog, torch.Tensor):
                    node_exog = node_exog.cpu().numpy()
            
            # Get the model for this node
            node_model = self.arimax_models[node_idx]
            
            # Generate forecast for this node
            # Get batch size
            batch_size = node_y.shape[0]
            node_forecasts = []

            # Process each batch element
            for b in range(batch_size):
                # Extract this batch element's data
                batch_exog = None
                if node_exog is not None:
                    batch_exog = node_exog[b]
                
                # Generate forecast for this batch element
                if batch_exog is not None:
                    forecast = node_model.forecast(steps=steps, exog=batch_exog)
                else:
                    forecast = node_model.forecast(steps=steps)
                
                node_forecasts.append(forecast)
            
            # Stack all batch forecasts
            node_forecast = np.stack(node_forecasts)
            
            forecasts.append(node_forecast)
        
        # Stack node forecasts into a single tensor [steps, nodes]
        forecasts = np.stack(forecasts, axis=-1)
        
        # Convert to PyTorch tensor
        return torch.tensor(forecasts, dtype=torch.float32, device=self.device)
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for evaluating the multi-node model
        
        Args:
            batch: (y, exog, target) tuple where target is the ground truth
            batch_idx: Index of the batch
        """
        y, exog, target = batch
        
        # Generate predictions across all nodes
        preds = self.forward(y, exog)
        
        # Calculate loss
        loss = self.loss_fn(preds, target)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        """
        Test step for evaluating the multi-node model
        
        Args:
            batch: (y, exog, target) tuple where target is the ground truth
            batch_idx: Index of the batch
        """
        y, exog, target = batch
        
        # Generate predictions across all nodes
        preds = self.forward(y, exog)
        
        # Calculate loss
        loss = self.loss_fn(preds, target)
        
        # Log metrics
        self.log('test_loss', loss)
        
        # Calculate per-node metrics
        node_metrics = {}
        for node_idx in range(self.num_nodes):
            node_loss = self.loss_fn(preds[..., node_idx], target[..., node_idx])
            node_metrics[f'node_{node_idx}_loss'] = node_loss
            self.log(f'node_{node_idx}_loss', node_loss)
        
        return {'test_loss': loss, 'preds': preds, 'targets': target, **node_metrics}
    
    def predict(self, y, exog, steps, batch_idx = None):
        """
        Prediction step for multi-node forecasting
        
        Args:
            batch: (y, exog, steps) tuple
            batch_idx: Index of the batch
        """
        
        # Generate forecast for all nodes
        forecast = self.forward(y, exog, steps)
        
        return forecast
    
    def freeze(self):
        # does nothing...
        pass