"""Paper-results runner extended with the Degree-Day baseline.

The DD baseline is non-trainable: ``trainer.fit`` is skipped, the fitted
per-node (or pooled) mappings are computed before ``trainer.test`` and
attached to the DegreeDayWrapper. Save paths follow the convention
``paper_results/degree_day_{mapping}_pooled{pooled}_{dataset}/{seed}/...``
so they line up with the aggregation script (aggregate_results.py).

Run examples:
    python paper_results.py model=degree_day                                   # uses YAML defaults
    python paper_results.py model=degree_day model.hparams.mapping=logistic    # override mapping
    python paper_results.py model=degree_day model.hparams.pooled=false        # per-node fit
"""
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.metrics import torch_metrics

import json
import numpy as np
import pandas as pd
import random
import os

from datasets.ostrinia import Ostrinia
from datasets.peakweather import PeakWeatherTSL

from models.dcrnn import DCRNNModel
from models.gru import GRU
from models.grugcn import GRUGCN
from models.gat import GAT
from models.tsl_dcrnn import tsl_dcrnn
from models.lstm import LSTM
from models.mlp import MLP
from models.transformer_encoder import TransformerBlock
from models.transformer_spatial import TransformerSpatial
# >>> Degree-Day baseline ---------------------------------------------------
from models.degree_day import (
    DegreeDayWrapper,
    compute_gdd,
    fit_degree_day,
    gdd_to_target_frame,
)
# <<< -----------------------------------------------------------------------

from extras.predictor import WrapPredictor, WrapPredictorDoubleTarget
from extras.metrics_logging import MetricsLogger
from extras.callbacks import Wandb_callback, MetricsHistory
from extras.plots import plot_predictions_test
from extras.plot_single_node import plot_predictions_test_single
from extras.masked_categorical_CE import MaskedCategoricalCrossEntropy, MaskedBinaryCrossEntropy
from extras.plot_flag_increment_results import plot_flag_increment
from extras.plot_last_results import plot_results_over_runs
from extras.plot_best import plot_results_over_runs as plot_best_results_over_runs
from extras.save_predictions import save_predictions

from numpy import concatenate, isnan, nan_to_num
from colorama import Fore
from extras.nmse_loss import MaskedNMSE
from tsl.data.batch_map import BatchMap, BatchMapItem
from torch.random import manual_seed as set_seed

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def set_reproducible_seeds(seed):
    """Set all random seeds for reproducibility."""
    seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        torch.set_deterministic(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_model(name):
    if name == 'dcrnn':
        return DCRNNModel
    elif name == 'tsl_dcrnn':
        return tsl_dcrnn
    elif name == 'gru':
        return GRU
    elif name == 'persistent':
        from models.persistent import persistent
        return persistent
    elif name == 'grugcn':
        return GRUGCN
    elif name == 'gat':
        return GAT
    elif name == 'lstm':
        return LSTM
    elif name == 'mlp':
        return MLP
    elif name == 'transformer':
        return TransformerBlock
    elif name == 'transformer_spatial':
        return TransformerSpatial
    elif name == 'degree_day':
        # Handled out-of-band; the wrapper holds the fitted per-node mappings.
        return None
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")


def build_cfg(cfg: DictConfig):
    model_name = cfg.model.name

    if cfg.dataset.name == "peakweather":
        suffix = "synth"
    else:
        suffix = "real"

    model_config_path = f"./model/paper_results/{model_name}_{suffix}.yaml"
    model_cfg = hydra.compose(config_name=model_config_path)
    print(f"Loaded configuration for {model_name}: {model_cfg}")

    # Merge the model updates into cfg.model (same nested access as the rest
    # of the codebase). Empty-string keys come from the leading "./" in the
    # config path; see your existing config layout for the rationale.
    model_updates = model_cfg['']['']['model']["paper_results"]
    cfg.model = OmegaConf.merge(cfg.model, model_updates)

    # >>> Degree-Day: no optimizer YAML; provide the minimum stubs required by
    # the downstream pipeline (num_workers for the data_module, loss_fn for
    # metric instantiation, epochs/grad_clip_val for the Trainer constructor
    # even though trainer.fit is not called).
    if model_name == "degree_day":
        if not hasattr(cfg, "optimizer") or cfg.optimizer is None:
            cfg.optimizer = OmegaConf.create({})
        cfg.optimizer = OmegaConf.merge(
            cfg.optimizer,
            OmegaConf.create({
                "name": cfg.optimizer.get("name", "Adam"),
                "loss_fn": cfg.optimizer.get("loss_fn", "mae"),
                "num_workers": cfg.optimizer.get("num_workers", 0),
                "epochs": 0,
                "grad_clip_val": 0.0,
                "hparams": cfg.optimizer.get("hparams", {"lr": 0.0}),
            }),
        )
        return cfg
    # <<< --------------------------------------------------------------------

    # Optimizer config (DL models only)
    optimizer_config_path = f"./optimizer/{model_name}_{suffix}.yaml"
    optimizer_cfg = hydra.compose(config_name=optimizer_config_path)
    optimizer_updates = optimizer_cfg['']['']['optimizer']
    cfg.optimizer = OmegaConf.merge(cfg.optimizer, optimizer_updates)
    print(f"Loaded configuration for optimizer {model_name}: {optimizer_cfg}")
    return cfg


@hydra.main(version_base=None, config_path="../config", config_name="paper_results")
def main(cfg: DictConfig):
    cfg = build_cfg(cfg)
    cfg.wandb.enable = False  # disable wandb init at the beginning

    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg.model, resolve=True),
            name=cfg.wandb.name,
        )

    is_dd = cfg.model.name == "degree_day"

    #######################################
    # dataset Initialization
    #######################################
    if cfg.dataset.name == "ostrinia":
        dataset = Ostrinia(
            root="datasets",
            target=cfg.dataset.target,
            add_second_target=cfg.dataset.add_second_target,
            spatial_information=cfg.dataset.get('spatial_information', True),
        )
    elif cfg.dataset.name == "peakweather":
        dataset = PeakWeatherTSL(
            root="datasets",
            input_zeros=True,
            station_type="meteo_station",
            freq='h',
            target='temperature',
            synth_data=cfg.dataset.get('synth_data', False),
            add_second_target=cfg.dataset.add_second_target,
        )

    # ---------------- Covariates ------------------------------------------
    if cfg.dataset.add_covariates:
        u = []
        covariates = dict()
        # Snapshot raw (non-standardised) extra_data BEFORE normalisation, so
        # the Degree-Day branch can compute GDD on physical units.
        raw_extra = {k: v.copy() if hasattr(v, "copy") else np.array(v)
                     for k, v in dataset.extra_data.items()}

        for key in dataset.extra_data.keys():
            covariate = dataset.extra_data[key].to_numpy().astype(float) \
                if hasattr(dataset.extra_data[key], "to_numpy") \
                else np.asarray(dataset.extra_data[key], dtype=float)
            if isnan(covariate).any():
                covariate_mean = covariate[~isnan(covariate)].mean()
                covariate_std = covariate[~isnan(covariate)].std()
            else:
                covariate_mean = covariate.mean()
                covariate_std = covariate.std()
            covariate = (covariate - covariate_mean) / covariate_std
            dataset.extra_data[key] = nan_to_num(covariate)
            u.append(dataset.extra_data[key].astype(float)[:, :, None])

        if cfg.dataset.add_second_target:
            u.append(dataset.flags["increment_flag"].astype(float).to_numpy()[..., None])

        covariates.update(u=concatenate(u, axis=-1))
    else:
        covariates = None
        raw_extra = {}

    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        mask=dataset.mask,
        covariates=covariates,
        horizon=cfg.dataset.horizon,
        window=cfg.dataset.window,
        stride=cfg.dataset.stride,
        delay=cfg.dataset.delay,
    )
    input_size = torch_dataset.n_channels

    torch_dataset.add_exogenous("enable_mask", dataset.mask.astype(float))

    if cfg.dataset.add_second_target:
        torch_dataset.add_covariate(
            name="second_target",
            value=dataset.flags["increment_flag"].astype(float),
            pattern="t n",
            add_to_input_map=True,
            synch_mode='horizon',
            preprocess=False,
            convert_precision=True,
        )

    # >>> Degree-Day: attach horizon-aligned GDD covariate ------------------
    gdd_full = None
    if is_dd:
        temp_key = cfg.model.hparams.get("temperature_key", "temperature")
        raw_temp = raw_extra.get(temp_key, None)
        if raw_temp is None:
            raise KeyError(
                f"[degree_day] Temperature key '{temp_key}' not found in "
                f"dataset.extra_data. Available keys: {list(raw_extra.keys())}"
            )
        if isinstance(raw_temp, pd.DataFrame):
            temp_arr = raw_temp.values.astype(float)
            timestamps = raw_temp.index
            if not isinstance(timestamps, pd.DatetimeIndex):
                timestamps = pd.DatetimeIndex(dataset.dataframe().index)

        else:
            temp_arr = np.asarray(raw_temp, dtype=float)
            timestamps = pd.DatetimeIndex(dataset.dataframe().index)

        target_df = dataset.dataframe()
        gdd_full = compute_gdd(
            temperature=temp_arr,
            timestamps=timestamps,
            t_base=cfg.model.hparams.t_base,
            t_upper=cfg.model.hparams.get("t_upper", None),
            cutoff=cfg.model.hparams.get("cutoff", "horizontal"),
            biofix_doy=int(cfg.model.hparams.get("biofix_doy", 1)),
        )
        gdd_df = gdd_to_target_frame(
            gdd=gdd_full,
            timestamps=timestamps,
            target_index=target_df.index,
            columns=target_df.columns,
        )
        gdd_full = gdd_df.values
        torch_dataset.add_covariate(
            name="gdd",
            value=gdd_full,
            pattern="t n",
            add_to_input_map=True,
            synch_mode='horizon',
            preprocess=False,
            convert_precision=True,
        )
        print(Fore.GREEN + f"[degree_day] GDD covariate attached "
              f"(shape={gdd_full.shape}, max={gdd_full.max():.1f} DD)" + Fore.RESET)
    # <<< --------------------------------------------------------------------

    scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
    # DD model predicts in original units, so skip target scaling to keep
    # batch.y comparable with y_hat; DL models are trained on scaled targets.
    transform = None if is_dd else {'target': StandardScaler(axis=scale_axis)}

    if cfg.wandb.enable:
        if 'batch_size' in wandb.config.keys():
            cfg.batch_size = wandb.config['batch_size']
            print(Fore.GREEN + f"Updated batch size: {cfg.batch_size}")
        if "epochs" in wandb.config.keys():
            cfg.optimizer.epochs = wandb.config['epochs']
            print(Fore.GREEN + f"Updated epochs: {cfg.optimizer.epochs}")

    data_module = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        batch_size=cfg.batch_size,
        workers=cfg.optimizer.num_workers,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
    )
    data_module.setup()

    if cfg.dataset.spatial_information:
        adj = dataset.get_connectivity(**cfg.dataset.connectivity,
                                       train_slice=data_module.train_slice)
        data_module.torch_dataset.set_connectivity(adj)

    ######################################
    # Model Initialization
    ######################################
    model = get_model(cfg.model.name)

    length_u = len(u) + 1 if covariates is not None else 1   # +1 for the mask

    model_kwargs = dict(
        n_nodes=torch_dataset.n_nodes,
        input_size=input_size + length_u if covariates is not None else 0,
        exog_size=0,
        output_size=torch_dataset.n_channels + 1 if cfg.dataset.add_second_target else torch_dataset.n_channels,
        weighted_graph=torch_dataset.edge_weight is not None,
        embedding_cfg=cfg.get('embedding'),
        horizon=torch_dataset.horizon,
        add_second_target=cfg.dataset.add_second_target,
        use_node_embeddings=cfg.model.hparams.get('use_node_embeddings', False),
    )

    if model is not None:
        model.filter_model_args_(model_kwargs)

    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################
    if cfg.wandb.enable and "loss_fn" in wandb.config:
        cfg.optimizer.loss_fn = wandb.config["loss_fn"]
        print(Fore.GREEN + f"Updated loss function: {cfg.optimizer.loss_fn}")

    if cfg.optimizer.loss_fn == "mae":
        base_loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)
    elif cfg.optimizer.loss_fn == "mse":
        base_loss_fn = torch_metrics.MaskedMSE(compute_on_step=True)
    elif cfg.optimizer.loss_fn == "nmse":
        base_loss_fn = MaskedNMSE()
    else:
        raise ValueError(f"Unknown loss type: {cfg.optimizer.loss_fn}")

    loss_fn = base_loss_fn

    if cfg.dataset.add_second_target:
        alpha = cfg.optimizer.alpha
        loss_fn_classification = MaskedBinaryCrossEntropy(mask_nans=True, mask_inf=True, alpha=5.0)
        loss_fn = {"loss_regression": base_loss_fn,
                   "loss_classification": loss_fn_classification,
                   "alpha": alpha}

    log_list = cfg.dataset.log_metrics
    log_metrics = MetricsLogger()
    metrics = log_metrics.filter_metrics(log_list)

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # ---------------- predictor selection ---------------------------------
    if is_dd:
        predictor_fn = DegreeDayWrapper
    elif cfg.dataset.add_second_target:
        predictor_fn = WrapPredictorDoubleTarget
    else:
        predictor_fn = WrapPredictor

    predictor = predictor_fn(
        model_class=model,
        n_nodes=torch_dataset.n_nodes,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
    )

    exp_logger = TensorBoardLogger(save_dir=cfg.run_dir, name=cfg.run_name)

    ######################################
    # Training and Setting up
    ######################################
    early_stop_callback = EarlyStopping(monitor='val_mae', patience=cfg.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.run_dir, save_top_k=1, monitor='val_mae', mode='min')

    if cfg.wandb.enable:
        # ---------------- wandb run init (DD has a different payload) -----
        if is_dd:
            run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                config={
                    "model": cfg.model.name,
                    "t_base": cfg.model.hparams.t_base,
                    "t_upper": cfg.model.hparams.get("t_upper", None),
                    "cutoff": cfg.model.hparams.get("cutoff", "horizontal"),
                    "biofix_doy": cfg.model.hparams.get("biofix_doy", 1),
                    "mapping": cfg.model.hparams.mapping,
                    "pooled": bool(cfg.model.hparams.pooled),
                    "dataset": cfg.dataset.name,
                    "window": cfg.dataset.window,
                    "horizon": cfg.dataset.horizon,
                },
            )
        else:
            run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                config={
                    "learning_rate": cfg.optimizer.hparams.lr,
                    "batch_size": cfg.batch_size,
                    "model": cfg.model.name,
                    "optimizer": cfg.optimizer.name,
                    "hidden_size": cfg.model.hparams.hidden_size,
                    "dropout": cfg.model.hparams.dropout,
                    "regularization_weight": cfg.get('regularization_weight', 0.0),
                    "dataset": cfg.dataset.name,
                    "epochs": cfg.optimizer.epochs,
                    "window": cfg.dataset.window,
                    "horizon": cfg.dataset.horizon,
                },
            )
        wandb_logger_callback = Wandb_callback(
            log_dir=cfg.run_dir, run=run, log_metrics=log_list,
        )
        callbacks = [checkpoint_callback, early_stop_callback, wandb_logger_callback]
    else:
        callbacks = [checkpoint_callback, early_stop_callback]

    trainer = Trainer(
        max_epochs=cfg.optimizer.epochs,
        limit_train_batches=cfg.train_batches,
        default_root_dir=cfg.run_dir,
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        gradient_clip_val=cfg.optimizer.grad_clip_val,
        callbacks=callbacks,
    )

    # ---------------- Fit phase --------------------------------------------
    if is_dd:
        # Out-of-band fit on the training slice (no gradient descent)
        y_full = dataset.dataframe().values.astype(np.float32)
        mask_full = dataset.mask.astype(bool) if dataset.mask is not None \
            else np.ones_like(y_full, dtype=bool)
        # TSL stores the mask as (T, N, 1) after _parse_target pads it to 3D.
        # fit_degree_day expects (T, N), so squeeze the trailing channel dim.
        if mask_full.ndim == 3 and mask_full.shape[-1] == 1:
            mask_full = mask_full.squeeze(-1)
        train_slice = data_module.train_slice

        fitted = fit_degree_day(
            gdd_train=gdd_full[train_slice],
            y_train=y_full[train_slice],
            mapping=cfg.model.hparams.mapping,
            pooled=bool(cfg.model.hparams.pooled),
            mask=mask_full[train_slice],
        )
        predictor.set_fitted(fitted)
        print(Fore.GREEN + f"[degree_day] Fit complete "
              f"(mapping={cfg.model.hparams.mapping}, "
              f"pooled={cfg.model.hparams.pooled}, "
              f"n_nodes={len(fitted)})" + Fore.RESET)
    else:
        trainer.fit(
            predictor,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )
        predictor.load_model(checkpoint_callback.best_model_path)

    ########################################
    # testing                              #
    ########################################
    predictor.freeze()
    test_data = trainer.test(predictor, dataloaders=data_module.test_dataloader())
    exp_logger.finalize('success')

    seed = torch.random.initial_seed()

    if cfg.dataset.add_second_target and not is_dd:
        plot_flag_increment(
            predictor=predictor,
            data_module=data_module,
            save_path=f"./paper_results/{cfg.model.name}_{cfg.dataset.name}/{seed}/flag_increment_plot.png",
        )

    # ---------------- Save path -------------------------------------------
    # DD path matches the convention expected by aggregate_results.py:
    #   degree_day_{mapping}_pooled{True|False}_{dataset}/{seed}/test_results.json
    if is_dd:
        run_tag = (f"degree_day_{cfg.model.hparams.mapping}"
                   f"_pooled{bool(cfg.model.hparams.pooled)}_{cfg.dataset.name}")
    else:
        run_tag = (f"{cfg.model.name}_{cfg.dataset.name}"
                   f"_nodes_embd_{cfg.model.hparams.use_node_embeddings}")

    suffix = "_double_target" if (cfg.dataset.add_second_target and not is_dd) else ""
    test_results_path = f"./paper_results/{run_tag}/{seed}/test_results{suffix}.json"
    os.makedirs(os.path.dirname(test_results_path), exist_ok=True)
    with open(test_results_path, 'w') as f:
        json.dump(test_data[0], f, indent=4)

    save_predictions(
        predictor=predictor,
        data_module=data_module,
        seed=seed,
        save_path=f"./paper_results/{run_tag}/{seed}/predictions",
    )


if __name__ == "__main__":
    for seed in range(42, 52):
        print(Fore.YELLOW + f"Running experiment with seed {seed}" + Fore.RESET)
        set_reproducible_seeds(seed)
        results = main()
        # print(results)
