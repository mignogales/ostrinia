import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from tsl.nn.layers.graph_convs.diff_conv import DiffConv
from tsl.nn.models.base_model import BaseModel


class LSTM(BaseModel):
    def __init__(self, input_size, hidden_size, output_size,
                 n_layers, horizon, dropout=0.1, activation='relu', 
                 exog_size=0, n_nodes=None, use_node_embeddings=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.horizon = horizon
        self.dropout = dropout
        self.activation = activation
        self.exog_size = exog_size
        self.use_node_embeddings = use_node_embeddings
        
        self.input_encode = MLP(input_size + exog_size, hidden_size, hidden_size, dropout=dropout)
        
        # LSTM layer instead of GRU
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, 
                           batch_first=True, dropout=dropout)
        
        self.fc = MLP(hidden_size, output_size * horizon, dropout=dropout)
        
        self.node_embeddings = nn.Parameter(torch.empty(n_nodes, hidden_size)) if n_nodes is not None else None
        if self.node_embeddings is not None:
            nn.init.xavier_uniform_(self.node_embeddings)
    
    def forward(self, x, u=None, edge_index=None, edge_weight=None, enable_mask=None):
        """
        Forward pass of the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_nodes, input_size).
            u (torch.Tensor, optional): Exogenous features tensor.
            edge_index (torch.Tensor): Edge index tensor for graph structure.
            edge_weight (torch.Tensor, optional): Edge weight tensor.
            enable_mask (torch.Tensor, optional): Mask tensor.
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, horizon, num_nodes, output_size).
        """
        # Check if exogenous features are provided
        if u is not None:
            # Concatenate input features with exogenous features
            x = torch.cat((x, u), dim=-1)
        
        if enable_mask is not None:
            x = torch.cat((x, enable_mask), dim=-1)
        
        # Encode input features
        x = self.input_encode(x)
        
        if self.use_node_embeddings:
            x = x + einops.repeat(self.node_embeddings, "n c -> b t n c", 
                                 b=x.shape[0], t=x.shape[1])  # [batch, T_enc, N, hidden_size]
        
        # Reshape for LSTM
        batch_size, seq_len, num_nodes, _ = x.size()
        x = einops.rearrange(x, 'b t n c -> (b n) t c')
        
        # LSTM layer - returns output and (hidden, cell) states
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Get the last timestep output
        x = lstm_out[:, -1]
        
        # Fully connected layer
        x = self.fc(x)
        
        # Rearrange output to match the expected shape
        x = einops.rearrange(x, '(b n) (t h) -> b t n h', 
                            b=batch_size, n=num_nodes, t=self.horizon)
        
        return x


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, n_layers=2, dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)