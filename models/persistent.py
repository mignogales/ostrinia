from tsl.nn.models.base_model import BaseModel
import torch
import torch.nn as nn
import einops

class persistent(BaseModel):
    def __init__(self, hidden_size=16, kernel_size=2, n_layers=1, activation='elu', dropout=0.2, use_final_relu=False):
        super().__init__()

        self.useless_parameter = nn.Parameter(torch.zeros(1))  # Dummy parameter to satisfy BaseModel requirements

    def forward(self, x, u=None, edge_index=None, edge_weight=None, enable_mask=None):

        x = x[:, -1, :]

        x = einops.rearrange(x, 'b n c -> b c n 1')

        return x
                
