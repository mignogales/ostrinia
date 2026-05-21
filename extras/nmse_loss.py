import torch
import torch.nn as nn

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

# ----------------------------------------------------------------------
# Functional form (nmse_loss) – analogous to torch.nn.functional.mse_loss
# ----------------------------------------------------------------------
def nmse_loss(pred: torch.Tensor,
              target: torch.Tensor,
              reduction: str = "none",
              eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized Mean-Squared Error (functional API), matching the signature
    style of torch.nn.functional.* loss helpers.
    """
    if pred.shape != target.shape:
        raise ValueError("`pred` and `target` must have identical shape.")
    
    input_shape = pred.shape

    pred_flat   = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    mse  = (pred_flat - target_flat) ** 2
    norm = (target_flat ** 2) + eps
    nmse = mse / norm                                  # per-sample

    # reshape to match the input shape
    nmse = nmse.view(*input_shape)

    if reduction == "mean":
        return nmse.mean()
    elif reduction == "sum":
        return nmse.sum()
    elif reduction == "none" or reduction is None:
        return nmse
    else:
        raise ValueError(f"Unsupported reduction '{reduction}'.")


# ----------------------------------------------------------------------
# Module wrapper – aligns with torch’s built-in loss classes
# ----------------------------------------------------------------------
class NMSELoss(_Loss):
    """
    Module wrapper for Normalized Mean-Squared Error that delegates the
    computation to the functional `nmse_loss`, exactly like PyTorch’s
    built-in loss classes delegate to torch.nn.functional.*.
    """
    def __init__(self,
                 reduction: str = "mean",
                 eps: float = 1e-8) -> None:
        super().__init__(reduction=reduction)
        self.eps = eps

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:  # keep names consistent
        return nmse_loss(input, target,
                         reduction=self.reduction,
                         eps=self.eps)


from typing import Any

import torch
from tsl.metrics.torch.metric_base import MaskedMetric


class MaskedNMSE(MaskedMetric):
    """Normalized Mean Squared Error Metric.

    Args:
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite
            values.
        at (int, optional): Whether to compute the metric only w.r.t. a certain
         time step.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 at=None,
                 **kwargs: Any):
        super(MaskedNMSE, self).__init__(metric_fn=nmse_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        # metric_fn_kwargs={'reduction': 'none'},
                                        at=at,
                                        **kwargs)