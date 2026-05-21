import torch
import torch.nn as nn
import torch.nn.functional as F
from tsl.engines import Predictor
from torchmetrics import Metric
from tsl.data import Data
from typing import Callable, Mapping, Optional, Type
import einops

from tsl.nn.layers.graph_convs.diff_conv import DiffConv
from tsl.nn.layers.recurrent.dcrnn import DCRNNCell
from tsl.nn.blocks.encoders.recurrent.dcrnn import DCRNN as DCRNN_tsl

from tsl.nn.models.base_model import BaseModel

class DiffusionConv(nn.Module):
    """
    Performs diffusion convolution:
      X_out = sum_{k=0}^{K-1} [ theta_k^fwd (D_o^{-1} W)^k  +  theta_k^bwd (D_i^{-1} W^T)^k ] X_in
    Inputs:
      - A_fwd, A_bwd: [N×N] precomputed normalized adjacency (fwd/backward)
      - K: number of diffusion steps
      - in_channels, out_channels
    """
    def __init__(self, A_fwd, A_bwd, kernel_size, in_ch, out_ch):
        super().__init__()
        self.kernel_size = kernel_size
        self.A_fwd = A_fwd     # tensor [N×N]
        self.A_bwd = A_bwd     # tensor [N×N]
        # learnable weights theta for each hop and direction:
        self.theta_fwd = nn.Parameter(torch.Tensor(kernel_size, in_ch, out_ch))
        self.theta_bwd = nn.Parameter(torch.Tensor(kernel_size, in_ch, out_ch))

        self.theta_self = nn.Parameter(torch.Tensor(in_ch, out_ch))

        self.bias = nn.Parameter(torch.Tensor(out_ch))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta_fwd)
        nn.init.xavier_uniform_(self.theta_bwd)

    def forward(self, X):
        """
        X: [batch, N, in_ch]
        returns: [batch, N, out_ch]
        """
        out = 0
        X_k_fwd = X   # (D_o^{-1}W)^0 X
        X_k_bwd = X   # (D_i^{-1}W^T)^0 X
        for k in range(self.kernel_size):
            # apply hop-k filters
            out = out \
                + torch.einsum('bnc,cd->bnd', X_k_fwd, self.theta_fwd[k]) \
                + torch.einsum('bnc,cd->bnd', X_k_bwd, self.theta_bwd[k]) \
                + self.bias
            
            # propagate for next hop
            X_k_fwd = torch.einsum('nm, bmc -> bnc', self.A_fwd, X_k_fwd)
            X_k_bwd = torch.einsum('nm, bmc -> bnc', self.A_bwd, X_k_bwd)


        return out + torch.einsum('bnc,cd->bnd', X, self.theta_self)


class DCGRUCell(nn.Module):
    """
    A single DCGRU cell.  Replaces linear maps with diffusion convs.
    """
    def __init__(self, kernel_size, in_channels, hidden_size, activation='relu'):
        super().__init__()
        self.hidden_size = hidden_size
        # diffusion conv for gates:
        diffusion_operator = DiffConv
        
        self.dc_xz = diffusion_operator(in_channels, hidden_size, kernel_size, activation)
        self.dc_hz = diffusion_operator(in_channels, hidden_size, kernel_size, activation)
        self.dc_xr = diffusion_operator(in_channels, hidden_size, kernel_size, activation)
        self.dc_hr = diffusion_operator(in_channels, hidden_size, kernel_size, activation)
        self.dc_xh = diffusion_operator(in_channels, hidden_size, kernel_size, activation)
        self.dc_hh = diffusion_operator(in_channels, hidden_size, kernel_size, activation)


    def forward(self, x, h_prev, edge_index=None, edge_weight=None):
        # x: [batch, N, input_size], h_prev: [batch, N, hidden_size]
        z = torch.sigmoid(self.dc_xz(x, edge_index, edge_weight) + self.dc_hz(h_prev, edge_index, edge_weight))
        r = torch.sigmoid(self.dc_xr(x, edge_index, edge_weight) + self.dc_hr(h_prev, edge_index, edge_weight))
        h_tilde = torch.tanh(self.dc_xh(x, edge_index, edge_weight) + self.dc_hh(r * h_prev, edge_index, edge_weight))
        h = (1 - z) * h_prev + z * h_tilde
        return h


class DCRNNModel(BaseModel):
    """
    A simple seq2seq DCRNN with scheduled sampling.
    """
    def __init__(self, kernel_size, input_size, hidden_size, output_size,
                 n_layers, horizon, dropout = 0.1, use_final_relu=False, autoregressive=False, activation='relu', exog_size=0, n_nodes=None, use_node_embeddings=False):
        super().__init__()
        self.horizon = horizon
        self.n_layers = n_layers
        self.autoregressive = autoregressive
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_final_relu = use_final_relu
        self.use_node_embeddings = use_node_embeddings

        # final projection from hidden_size -> output_size
        self.in_proj = nn.Linear(input_size + exog_size, hidden_size)

        # Encoder: stack of DCGRU cells
        self.encoder_cells = nn.ModuleList([
            DCGRUCell(kernel_size,
                      hidden_size,
                      hidden_size)
            for _ in range(n_layers)
        ])
        
        if autoregressive:
            self.decoder_cells = nn.ModuleList([
                DCGRUCell(kernel_size, output_size if i == 0 else hidden_size, hidden_size)
                for i in range(n_layers)
            ]) 


        # out projection
        self.readout = MLP(input_size=hidden_size,
                           output_size=output_size * horizon if not autoregressive else output_size,
                           n_layers=1,
                           dropout=dropout,)
        
        # self.readout = StrangeMLP(input_size=hidden_size,
        #                     horizon=horizon if not autoregressive else output_size,
        #                     output_size=output_size,
        #                     hidden_size=hidden_size,
        #                     n_layers=1,
        #                     dropout=dropout,)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

        self.node_embeddings = nn.Parameter(torch.empty(n_nodes, hidden_size)) if n_nodes is not None else None
        if self.node_embeddings is not None:
            nn.init.xavier_uniform_(self.node_embeddings)

    def encode(self, inputs, edge_index=None, edge_weight=None):
        # inputs: [batch, T, N, input_size]
        batch_size, T, N, _ = inputs.shape
        h = [
            torch.zeros(batch_size, N, cell.hidden_size, device=inputs.device)
            for cell in self.encoder_cells
        ]

        for t in range(T):
            x_t = inputs[:, t]
            for i, cell in enumerate(self.encoder_cells):
                h[i] = cell(x_t, h[i], edge_index=edge_index, edge_weight=edge_weight)
                x_t = self.dropout(h[i])                
        return h

    def decode_autoregressive(self, h, teacher_forcing=None, training=False, epsilon=0.0):
        # h: list of final encoder hidden states
        # teacher_forcing: [batch, T_dec, N, output_size] or None
        batch_size, N, _ = h[0].shape
        outputs = []
        # start with zeros or last input

        # start with zeros
        # y_prev = torch.zeros(batch_size, N, self.hidden_size, device=h[0].device)
        y_prev = h[-1]  # [batch, N, hidden_size] 

        for t in range(self.horizon):
            # x_t = 
            x_t = y_prev  # [batch, N, hidden_size]
            new_h = []
            for i, cell in enumerate(self.decoder_cells):
                h_i = cell(x_t, h[i])
                new_h.append(h_i)
                x_t = h_i
            h = new_h
            y_prev = self.readout(x_t)

            outputs.append(y_prev.unsqueeze(1))

            # scheduled sampling
            if (teacher_forcing is not None and training and (torch.rand(1).item() < epsilon)):
                y_prev = teacher_forcing[:, t]

        return torch.cat(outputs, dim=1)

        

    def forward(self, x, y_true=None, training=False, epsilon=0.0, edge_index=None, edge_weight=None, u=None, past_values=None, enable_mask=None):
        # x: [batch, T_enc, N, input_size]
        # y_true: optional [batch, T_dec, N, output_size] for teacher forcing

        x_skip = x[:, -1:, :, :]  # [batch, 1, N, input_size]

        if u is not None:
            x = torch.cat((x, u), dim=-1)  # [batch, T_enc, N, input_size + exog_size]

        if past_values is not None:
            x = torch.cat((x, past_values), dim=-1)

        if enable_mask is not None:
            x = torch.cat((x, enable_mask), dim=-1)

        # if self.node_embeddings is not None:
        #     # Add node embeddings to the input
        #     x = torch.cat((x, einops.repeat(self.node_embeddings, "n c -> b t n c", b=x.shape[0], t=x.shape[1])), dim=-1)

        x = self.in_proj(x)  # [batch, T_enc, N, hidden_size]

        if self.use_node_embeddings:
            x = x + einops.repeat(self.node_embeddings, "n c -> b t n c", b=x.shape[0], t=x.shape[1])  # [batch, T_enc, N, hidden_size]

        x = self.dropout(x)  # [batch, T_enc, N, hidden_size]
        x = self.activation(x)


        h = self.encode(x, edge_index=edge_index, edge_weight=edge_weight)  # [batch, T_enc, N, hidden_size]

        if self.autoregressive:
            # h = self.encoder(x, edge_index = edge_index, edge_weight = edge_weight)  # [batch, T_enc, N, hidden_size]
            y_pred = self.decode_autoregressive(h, teacher_forcing=y_true, training=training, epsilon=epsilon)
        else:
            h = h[-1]  # [batch, N, hidden_size]
            y_pred = self.readout(h)
            y_pred = einops.rearrange(y_pred, "b n t -> b t n 1")
            # last two slices of t are integers to check the slope, if add_second_target is True. get them and concatenate along last dim
            if y_pred.shape[1] != self.horizon:
                y_pred_1 = y_pred[:, :self.horizon, :, :]
                y_pred_2 = y_pred[:, self.horizon:, :, :]
                y_pred = torch.cat((y_pred_1, y_pred_2.transpose(1,3)), dim=-1)


        if self.use_final_relu:
            # if self.training:
            #     y_pred = y_pred + einops.repeat(x_skip, "b 1 n c -> b t n c", t = self.horizon)
            # else:
                y_pred = F.relu(y_pred) + einops.repeat(x_skip, "b 1 n c -> b t n c", t = self.horizon)

        return y_pred   # [batch, N, T_dec, output_size]
    
class StrangeMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, horizon: int, n_layers: int = 1, hidden_size: int = 256, dropout: float = 0.):
        super(StrangeMLP, self).__init__()
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList()
        # for _ in range(n_layers - 1):
        #     self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.next_time_step = nn.Linear(hidden_size, hidden_size)
        self.readout = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.horizon = horizon
            
    def forward(self, x):
        
        x = self.encoder(x)  # [batch, N, hidden_size]
        x = self.relu(x)
        x = self.dropout(x)

        outputs = []

        for _ in range(self.horizon):
            x = self.next_time_step(x)
            x = self.relu(x)
            x = self.dropout(x)
            x_out = self.readout(x)  # [batch, N, output_size]
            x = x_out + x  # residual connection
            outputs.append(x_out)        

        x = torch.cat(outputs, dim=-1)  # [batch, horizon, N, output_size]
        
        return x
    
class MLP(nn.Module):
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 hidden_size: int = 256,
                 horizon: int = 1,
                 n_layers: int = 1,
                 activation: str = 'ELU',
                 dropout: float = 0.):
        
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size * horizon))
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(torch.nn, activation)()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

class Predictor_DCRNN(Predictor):
    def __init__(self, model: Optional[torch.nn.Module] = None,
                 loss_fn: Optional[Callable] = None,
                 scale_target: bool = False,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 *,
                 model_class: Optional[Type] = None,
                 model_kwargs: Optional[Mapping] = None,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional[Type] = None,
                 scheduler_kwargs: Optional[Mapping] = None,
                 num_epochs: Optional[int] = None,):
        
        super().__init__(model=model, loss_fn=loss_fn, scale_target=scale_target,
                         metrics=metrics, model_class=model_class,
                         model_kwargs=model_kwargs, optim_class=optim_class,
                         optim_kwargs=optim_kwargs, scheduler_class=scheduler_class,
                            scheduler_kwargs=scheduler_kwargs)
        
        self.num_epochs = num_epochs
        self.scheduler_kwargs["epsilon"] = None
    

    def training_step(self, batch, batch_idx):

        if batch_idx == 0:
            if self.scheduler_kwargs["epsilon"] is None:
                epsilon = 1.0
                self.scheduler_kwargs["epsilon"] = epsilon
            else:
                self.scheduler_kwargs["epsilon"] = self.scheduler_kwargs["epsilon"] - (1 / self.num_epochs)
        
        epsilon = self.scheduler_kwargs["epsilon"]
        

        y = y_loss = batch.y
        mask = batch.get('mask')

        # Compute predictions and compute loss
        y_hat_loss = self.predict_batch(batch,
                                        preprocess=False,
                                        postprocess=False, 
                                        epsilon=epsilon,
                                        training=True)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        y_loss = batch.transform['y'].transform(y)
        y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.train_metrics.update(y_hat, y, mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        
        y = y_loss = batch.y
        mask = batch.get('mask')

        # Compute predictions
        y_hat_loss = self.predict_batch(batch,
                                        preprocess=False,
                                        postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        val_loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.val_metrics.update(y_hat, y, mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)

        return val_loss
    
    def test_step(self, batch, batch_idx):
        y = y_loss = batch.y
        mask = batch.get('mask')

        # Compute predictions
        y_hat_loss = self.predict_batch(batch,
                                        preprocess=False,
                                        postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        test_loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.test_metrics.update(y_hat, y, mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)

        return test_loss
    
    def predict_batch(self,
                      batch: Data,
                      preprocess: bool = False,
                      postprocess: bool = True,
                      return_target: bool = False,
                      training: bool = False,
                      epsilon: float = 0.0,
                      **forward_kwargs):
        
        inputs, targets, mask, transform = self._unpack_batch(batch)
        if preprocess:
            for key, trans in transform.items():
                if key in inputs:
                    inputs[key] = trans.transform(inputs[key])

        # rescale targets
        y = batch.transform['y'].transform(targets.y)

        if forward_kwargs is None:
            forward_kwargs = dict()

        inputs["y_true"] = y
        forward_kwargs["training"] = training
        forward_kwargs["epsilon"] = epsilon
        y_hat = self.forward(**inputs, **forward_kwargs)

        # Rescale outputs
        if postprocess:
            trans = transform.get('y')
            if trans is not None:
                y_hat = trans.inverse_transform(y_hat)
        if return_target:
            y = targets.get('y')
            return y, y_hat, mask
        return y_hat

    def predict_batch(self,
                      batch: Data,
                      preprocess: bool = False,
                      postprocess: bool = True,
                      return_target: bool = False,
                      **forward_kwargs):
        """This method takes as input a :class:`~tsl.data.Data` object and
        outputs the predictions.

        Note that this method works seamlessly for all :class:`~tsl.data.Data`
        subclasses like :class:`~tsl.data.StaticBatch` and
        :class:`~tsl.data.DisjointBatch`.

        Args:
            batch (Data): The batch to be forwarded to the model.
            preprocess (bool, optional): If :obj:`True`, then preprocess tensors
                in :attr:`batch.input` using transformation modules in
                :attr:`batch.transform`. Note that inputs are preprocessed
                before creating the batch by default.
                (default: :obj:`False`)
            postprocess (bool, optional): If :obj:`True`, then postprocess the
                model output using transformation modules for
                :attr:`batch.target` in :attr:`batch.transform`.
                (default: :obj:`True`)
            return_target (bool, optional): If :obj:`True`, then returns also
                the prediction target :attr:`batch.target` and the prediction
                mask :attr:`batch.mask`, besides the model output. In this case,
                the order of the arguments in the return is
                :attr:`batch.target`, :obj:`y_hat`, :attr:`batch.mask`.
                (default: :obj:`False`)
            **forward_kwargs: additional keyword arguments passed to the forward
                method.
        """
        inputs, targets, mask, transform = self._unpack_batch(batch)
        if preprocess:
            for key, trans in transform.items():
                if key in inputs:
                    inputs[key] = trans.transform(inputs[key])

        if forward_kwargs is None:
            forward_kwargs = dict()
        y_hat = self.forward(**inputs, **forward_kwargs)
        # Rescale outputs
        if postprocess:
            trans = transform.get('y')
            if trans is not None:
                y_hat = trans.inverse_transform(y_hat)
        if return_target:
            y = targets.get('y')
            return y, y_hat, mask
        return y_hat
