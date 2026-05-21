import math
import torch
import torch.nn as nn
import einops
from tsl.nn.models.base_model import BaseModel
from tsl.nn.layers.graph_convs.diff_conv import DiffConv


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device, x.dtype)


class TransformerSpatial(BaseModel):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 n_layers_gnn,
                 n_layers_transformer,
                 kernel_size,
                 horizon,
                 window_size=12,
                 n_heads=4,
                 dropout=0.1,
                 exog_size=0,
                 n_nodes=None,
                 use_node_embeddings=False,
                 activation='gelu',
                 ff_multiplier=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.horizon = horizon
        self.exog_size = exog_size
        self.use_node_embeddings = use_node_embeddings

        # Input encoding
        self.input_encode = subMLP(input_size + exog_size, hidden_size, hidden_size, dropout=dropout, n_layers=1)

        # Optional node embeddings
        self.node_embeddings = nn.Parameter(torch.empty(n_nodes, hidden_size)) if use_node_embeddings else None
        if self.node_embeddings is not None:
            nn.init.xavier_uniform_(self.node_embeddings)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(hidden_size)

        # Transformer temporal encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=ff_multiplier * hidden_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers_transformer)

        self.out_proj = nn.Linear(window_size, 1)

        self.diff_conv = nn.ModuleList()
        for _ in range(n_layers_gnn):
            self.diff_conv.append(
                DiffConv(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    k=kernel_size,
                    activation=activation,
                )
            )

        # Final projection (assumes T=12)
        self.fc = subMLP(hidden_size, output_size * horizon, hidden_size=hidden_size, dropout=dropout, n_layers=2)

    def forward(self, x, u=None, edge_index=None, edge_weight=None, enable_mask=None):
        # Optional concatenation of exogenous information
        if u is not None:
            x = torch.cat((x, u), dim=-1)
        if enable_mask is not None:
            x = torch.cat((x, enable_mask), dim=-1)

        # Input encoding
        x = self.input_encode(x)  # [B, T, N, H]

        # Add node embeddings if enabled
        if self.use_node_embeddings:
            x = x + einops.repeat(self.node_embeddings, "n h -> b t n h", b=x.shape[0], t=x.shape[1])

        B, T, N, H = x.shape

        # Reshape for temporal attention per node
        x = einops.rearrange(x, "b t n h -> (b n) t h")  # [B*N, T, H]

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoder
        x = self.encoder(x)  # [B*N, T, H]

        x = einops.rearrange(x, "bn t h -> bn h t")  # [B*N, H, T]

        # Project horizon dimension to 1
        x = self.out_proj(x)  # [B*N, H, 1]

        # Flatten
        x = einops.rearrange(x, "(b n) h 1 -> b n h", b=B)  # [B, N, H]

        # Diffusion convolution layers
        for diff_conv_layer in self.diff_conv:
            x = diff_conv_layer(x, edge_index, edge_weight)  # [B, N, H]

        # Final projection
        x = self.fc(x)  # [B*N, output_size*horizon]

        # Return to [B, 1, N, output_size]
        x = einops.rearrange(x, "b n h -> b 1 n h", b=B, n=N)
        return x


class subMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, n_layers=2, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
