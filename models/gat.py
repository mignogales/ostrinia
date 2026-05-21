import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from tsl.nn.models.base_model import BaseModel


class GAT(BaseModel):
    def __init__(self, input_size, hidden_size, output_size,
                 n_layers_rnn, n_layers_gnn, horizon, dropout=0.1, activation='relu', 
                 exog_size=0, n_nodes=None, use_node_embeddings=False,
                 n_heads=8, concat_heads=True, residual=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers_rnn = n_layers_rnn
        self.n_layers_gnn = n_layers_gnn
        self.horizon = horizon
        self.dropout = dropout
        self.activation = activation
        self.exog_size = exog_size
        self.use_node_embeddings = use_node_embeddings
        self.n_heads = n_heads
        self.concat_heads = concat_heads
        self.residual = residual
        
        # Input encoding layer
        self.input_encode = MLP(input_size + exog_size, hidden_size, hidden_size, dropout=dropout)
        
        # Temporal processing (GRU) - applied per node
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers_rnn, batch_first=True, dropout=dropout)
        
        # Spatial processing (GAT layers) - applied per timestep
        self.gat_layers = nn.ModuleList()
        
        gat_input_size = hidden_size
        for i in range(n_layers_gnn):
            if i > 0 and concat_heads:
                gat_input_size = hidden_size * n_heads
            
            if i == n_layers_gnn - 1:
                # Last layer: single head, no concatenation
                self.gat_layers.append(
                    GATLayer(gat_input_size, hidden_size, 1, 
                            concat=False, dropout=dropout, residual=residual)
                )
            else:
                self.gat_layers.append(
                    GATLayer(gat_input_size, hidden_size, n_heads, 
                            concat=concat_heads, dropout=dropout, residual=residual)
                )
        
        # Output projection
        self.fc = MLP(hidden_size, output_size, dropout=dropout)
        
        # Node embeddings
        self.node_embeddings = nn.Parameter(torch.empty(n_nodes, hidden_size)) if n_nodes is not None else None
        if self.node_embeddings is not None:
            nn.init.xavier_uniform_(self.node_embeddings)
    
    def forward(self, x, u=None, edge_index=None, edge_weight=None, enable_mask=None):
        """
        Forward pass of the GAT model (Time-first, then Space).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, num_nodes, input_size).
            u (torch.Tensor, optional): Exogenous features tensor.
            edge_index (torch.Tensor): Edge index tensor for graph structure [2, num_edges].
            edge_weight (torch.Tensor, optional): Edge weight tensor.
            enable_mask (torch.Tensor, optional): Node mask tensor.
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, horizon, num_nodes, output_size).
        """
        
        # Concatenate exogenous features if provided
        if u is not None:
            x = torch.cat((x, u), dim=-1)
        
        if enable_mask is not None:
            x = torch.cat((x, enable_mask), dim=-1)
        
        # Encode input features
        x = self.input_encode(x)  # [batch, T, N, hidden_size]
        
        # Add node embeddings if available
        if self.use_node_embeddings and self.node_embeddings is not None:
            x = x + einops.repeat(self.node_embeddings, "n c -> b t n c", 
                                 b=x.shape[0], t=x.shape[1])
        
        batch_size, time_steps, num_nodes, features = x.size()
        
        # STEP 1: Temporal Processing (GRU per node)
        # Reshape for GRU processing
        x = einops.rearrange(x, 'b t n f -> (b n) t f')
        
        # Apply GRU to process temporal sequences
        gru_out, _ = self.gru(x)  # [(b*n), T, hidden_size]
        
        # Take outputs at multiple horizons or just the last output
        if self.horizon == 1:
            temporal_features = gru_out[:, -1:, :]  # [(b*n), 1, hidden_size]
        else:
            # Sample at regular intervals or take last horizon steps
            temporal_features = gru_out[:, -self.horizon:, :]  # [(b*n), horizon, hidden_size]
        
        # Reshape back to separate batch, nodes, and time
        temporal_features = einops.rearrange(temporal_features, '(b n) t f -> b t n f', 
                                            b=batch_size, n=num_nodes)
        
        # STEP 2: Spatial Processing (GAT per timestep)
        outputs = []
        for t in range(temporal_features.size(1)):
            h = temporal_features[:, t]  # [batch, N, features]
            
            # Reshape for GAT processing
            h = einops.rearrange(h, 'b n f -> (b n) f')
            
            # Apply GAT layers for spatial attention
            for gat_layer in self.gat_layers:
                h = gat_layer(h, edge_index, edge_weight)
            
            # Reshape back
            h = einops.rearrange(h, '(b n) f -> b n f', b=batch_size, n=num_nodes)
            
            # Final output projection
            h = self.fc(h)  # [batch, N, output_size]
            
            outputs.append(h)
        
        # Stack temporal outputs
        x = torch.stack(outputs, dim=1)  # [batch, horizon, N, output_size]
        
        return x


class GATLayer(nn.Module):
    """Single Graph Attention Layer"""
    
    def __init__(self, in_features, out_features, n_heads, 
                 concat=True, dropout=0.1, residual=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        self.residual = residual
        
        # Linear transformations for each head
        self.W = nn.Parameter(torch.empty(n_heads, in_features, out_features))
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_features, 1))
        
        # Residual connection
        if residual:
            self.residual_transform = nn.Linear(in_features, 
                                               out_features * n_heads if concat else out_features)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_features * n_heads if concat else out_features)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [N, in_features]
            edge_index: Graph connectivity [2, E]
            edge_weight: Edge weights [E] (optional)
        
        Returns:
            Updated node features [N, out_features * n_heads] or [N, out_features]
        """
        
        N = x.size(0)
        
        # Store input for residual connection
        residual = x
        
        # Linear transformation for each head
        h = torch.matmul(x.unsqueeze(0), self.W)  # [n_heads, N, out_features]
        
        # Compute attention coefficients
        edge_src, edge_dst = edge_index[0],  edge_index[1]
        
        # Prepare source and target features for attention
        h_src = h[:, edge_src]  # [n_heads, E, out_features]
        h_dst = h[:, edge_dst]  # [n_heads, E, out_features]
        
        # Concatenate source and target features
        h_cat = torch.cat([h_src, h_dst], dim=-1)  # [n_heads, E, 2*out_features]
        
        # Compute attention scores
        e = self.leaky_relu(torch.matmul(h_cat, self.a).squeeze(-1))  # [n_heads, E]
        
        # Apply edge weights if provided
        if edge_weight is not None:
            e = e * edge_weight.unsqueeze(0)
        
        # Compute attention weights using softmax
        alpha = torch.zeros(self.n_heads, N, N, device=x.device)
        alpha[:, edge_dst, edge_src] = e
        
        # Masked softmax for attention normalization
        alpha = F.softmax(alpha, dim=-1)
        alpha = self.dropout_layer(alpha)
        
        # Apply attention to node features
        out = torch.matmul(alpha, h)  # [n_heads, N, out_features]
        
        # Concatenate or average heads
        if self.concat:
            out = einops.rearrange(out, 'h n f -> n (h f)')
        else:
            out = out.mean(dim=0)
        
        # Apply residual connection
        if self.residual:
            out = out + self.residual_transform(residual)
        
        # Apply activation and normalization
        out = F.elu(out)
        out = self.layer_norm(out)
        out = self.dropout_layer(out)
        
        return out


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