import torch
import torch.nn.functional as F
from typing import Any
from tsl.metrics.torch.metric_base import MaskedMetric

class MaskedCategoricalCrossEntropy(MaskedMetric):
    """Masked Categorical Cross-Entropy for multi-dimensional TSL tensors.
    
    Handles shape conversion between TSL convention and cross-entropy expectations.
    """
    
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    
    def __init__(self,
                mask_nans=False,
                mask_inf=False,
                at=None,
                **kwargs: Any):
        super(MaskedCategoricalCrossEntropy, self).__init__(
            metric_fn=self._cross_entropy_adapter,
            mask_nans=mask_nans,
            mask_inf=mask_inf,
            at=at,
            **kwargs
        )
    
    def _cross_entropy_adapter(self, y_pred, y_true):
        """Adapts inputs to work with F.cross_entropy."""
        # Reshape tensors to expected format
        # y_pred: [B, T, N, C] -> reshape to [B*T*N, C]
        # y_true: [B, T, N, C] -> reshape to [B*T*N]
        
        orig_shape = y_pred.shape
        batch_size = orig_shape[0]
        
        # Get number of elements per sample for reshaping back
        elements_per_sample = torch.prod(torch.tensor(orig_shape[1:-1]))
        
        # Reshape prediction: [B, T, N, C] -> [B*T*N, C]
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
        
        # Reshape target: [B, T, N, C] -> [B*T*N]
        # If target is one-hot encoded, convert to class indices
        if len(y_true.shape) > 3:
            y_true_flat = torch.argmax(y_true, dim=-1).reshape(-1)
        else:
            y_true_flat = y_true.reshape(-1)
        
        # Compute loss
        loss = F.cross_entropy(y_pred_flat, y_true_flat.to(torch.long), reduction='none')
        
        # Reshape back to original shape without last dim
        return loss.reshape(orig_shape[:-1])
    
    def _compute_masked(self, y_hat, y, mask):
        _check_same_shape = lambda a, b: True  # Override to allow different shapes
        
        # Compute loss with our adapter
        val = self.metric_fn(y_hat, y)
        
        # Reshape mask if needed
        if len(mask.shape) > len(val.shape):
            mask = mask.squeeze(-1)
        
        mask = self._check_mask(mask, val)
        val = torch.where(mask, val, torch.zeros_like(val))
        
        return val.sum(), mask.sum()




class MaskedBinaryCrossEntropy(MaskedMetric):
    """Masked Binary Cross-Entropy for multi-dimensional TSL tensors.
    
    Handles shape conversion between TSL convention and binary cross-entropy expectations.
    """
    
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    
    def __init__(self,
                mask_nans=False,
                mask_inf=False,
                at=None,
                alpha=None,
                **kwargs: Any):

        self.alpha = torch.tensor(alpha, device='cuda' if alpha is not None else None)

        loss_fn = torch.nn.BCELoss(reduction='none', weight=self.alpha)

        super(MaskedBinaryCrossEntropy, self).__init__(
            metric_fn=self._binary_cross_entropy_adapter,
            mask_nans=mask_nans,
            mask_inf=mask_inf,
            at=at,
            **kwargs
        )

        self.loss_fn = loss_fn

    def _binary_cross_entropy_adapter(self, y_pred, y_true):
        """Adapts inputs to work with F.binary_cross_entropy."""
        # Reshape tensors to expected format
        # y_pred: [B, T, N, 1] -> reshape to [B*T*N]
        # y_true: [B, T, N, 1] -> reshape to [B*T*N]
        
        orig_shape = y_pred.shape
        
        # Flatten predictions and targets
        y_pred_flat = y_pred.reshape(-1)
        y_true_flat = y_true.reshape(-1)
        
        # Compute loss
        loss = self.loss_fn(F.sigmoid(y_pred_flat), y_true_flat)
        
        # Reshape back to original shape without last dim
        return loss.reshape(orig_shape[:-1])
    
    def _compute_masked(self, y_hat, y, mask):
        _check_same_shape = lambda a, b: True  # Override to allow different shapes
        
        # Compute loss with our adapter
        val = self.metric_fn(y_hat, y)
        
        # Reshape mask if needed
        if len(mask.shape) > len(val.shape):
            mask = mask.squeeze(-1)
        
        mask = self._check_mask(mask, val)
        val = torch.where(mask, val, torch.zeros_like(val))
        
        return val.sum(), mask.sum()