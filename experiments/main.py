import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import pandas as pd
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.metrics import torch_metrics

import numpy as np

from datasets.ostrinia import Ostrinia
from datasets.peakweather import PeakWeatherTSL
from datasets.ostrinia_simple_synth import OstriniaSimpleSynth

from models.dcrnn import DCRNNModel
from models.gru import GRU
from models.gat import GAT
from models.grugcn import GRUGCN
from models.lstm import LSTM
from models.mlp import MLP
from models.transformer_encoder import TransformerBlock
from models.transformer_spatial import TransformerSpatial

from models.arimax import fit_arimax, ARIMAXWrapper
from models.tsl_dcrnn import tsl_dcrnn
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
from extras.masked_categorical_CE import MaskedCategoricalCrossEntropy

from numpy import concatenate, isnan, nan_to_num
from colorama import Fore
from extras.nmse_loss import MaskedNMSE

from tsl.data.batch_map import BatchMap, BatchMapItem


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


# ---------------------------------------------------------------------------
# Models that are NOT trained through pytorch_lightning's trainer.fit()
# (handled out-of-band, like ARIMAX).
# ---------------------------------------------------------------------------
NON_TRAINABLE_MODELS = ("arimax", "degree_day", "persistent")


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
    elif name == 'arimax':
        print("Using ARIMAX model")
        return None
    elif name == 'degree_day':
        print("Using Degree-Day baseline model")
        return None
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
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")

def build_cfg(cfg: DictConfig):

    model_name = cfg.model.name  # Get the model name from the config
    model_config_path = f"./models/{model_name}.yaml"  # Construct the model-specific config path

    # Load the model-specific config
    model_cfg = hydra.compose(config_name=model_config_path)
    print(f"Loaded configuration for {model_name}: {model_cfg}")

    # Optimizer configuration
    optimizer_name = cfg.optimizer.name  # Get the optimizer name from the config
    optimizer_config_path = f"./optimizers/{optimizer_name}.yaml"  # Construct the optimizer-specific config path
    optimizer_cfg = hydra.compose(config_name=optimizer_config_path)
    print(f"Loaded configuration for optimizer {optimizer_name}: {optimizer_cfg}")

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):

    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg.model, resolve=True),
            name=cfg.wandb.name,
            # mode=cfg.wandb.mode
        )

    #######################################
    # dataset Initialization
    #######################################
    if cfg.dataset.name == "ostrinia":
        dataset = Ostrinia(root = "datasets", target = cfg.dataset.target, add_second_target = cfg.dataset.add_second_target, spatial_information = cfg.dataset.get('spatial_information', True))
    elif cfg.dataset.name == "peakweather":
        dataset = PeakWeatherTSL(
                 root="datasets",
                   input_zeros=True,
                    station_type="meteo_station",
                     freq='h',
                      target='temperature',
                      synth_data=cfg.dataset.get('synth_data', False),
                      add_second_target=cfg.dataset.add_second_target)
    elif cfg.dataset.name == "ostrinia_synth":
        dataset = OstriniaSimpleSynth(
            years=list(range(cfg.dataset.years.start, cfg.dataset.years.end + 1)),
            n_nodes=cfg.dataset.n_nodes,
        )

    # Compute covariates
    if cfg.dataset.add_covariates and list(dataset.extra_data.keys()):
        u = []
        covariates = dict() 
        for key in dataset.extra_data.keys():
            # normalize the covariate
            covariate = dataset.extra_data[key].to_numpy().astype(float)
            # if there are nans mask the data before computing the mean and std
            if isnan(covariate).any():
                covariate_mean = covariate[~isnan(covariate)].mean()
                covariate_std = covariate[~isnan(covariate)].std()
            else:
                covariate_mean = covariate.mean()
                covariate_std = covariate.std()
            covariate = (covariate - covariate_mean) / covariate_std
            # we normalize by mean correctly but not by std, it ends up being smaller
            dataset.extra_data[key] = nan_to_num(covariate)

            u.append(dataset.extra_data[key].astype(float)[:, :, None])  # add a new axis to the covariates

        if cfg.dataset.add_second_target:
            # add the increment flag as a covariate
            u.append(dataset.flags["increment_flag"].astype(float).to_numpy()[..., None])

        covariates.update(u=concatenate(u, axis=-1))
    else:
        covariates = None

    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          horizon=cfg.dataset.horizon,
                                          window=cfg.dataset.window,
                                          stride=cfg.dataset.stride,
                                          delay=cfg.dataset.delay)
    
    input_size = torch_dataset.n_channels

    torch_dataset.add_exogenous("enable_mask", dataset.mask.astype(float))

    if cfg.dataset.add_second_target:
        torch_dataset.add_covariate(name="second_target",
                                      value=dataset.flags["increment_flag"].astype(float),
                                      pattern="t n",
                                      add_to_input_map=True,
                                      synch_mode='horizon',
                                      preprocess=False,
                                      convert_precision=True)
        
        # torch_dataset.update_target_map(BatchMap(
        #          keys='second_target')
        # )

    # >>> Degree-Day: attach horizon-aligned GDD covariate -------------------
    # Must happen BEFORE data_module.setup() / construction so the batch
    # input map contains it. We compute GDD from raw (non-standardised)
    # temperature, retrieved from dataset.extra_data with the configurable key.
    gdd_full = None  # filled when model is degree_day; used later for the fit
    if cfg.model.name == "degree_day":
        temp_key = cfg.model.hparams.temperature_key
        if temp_key not in dataset.extra_data:
            raise KeyError(
                f"[degree_day] Temperature key '{temp_key}' not found in "
                f"dataset.extra_data. Available keys: "
                f"{list(dataset.extra_data.keys())}"
            )
        # NB: dataset.extra_data[temp_key] may already have been normalised in
        # the covariate-construction block above. We re-read the raw values
        # from the dataset's underlying source if available; otherwise we warn.
        raw_temp = getattr(dataset, "raw_extra_data", {}).get(temp_key, None)
        if raw_temp is None:
            # Fall back to whatever is currently stored. If it has been
            # standardised, the resulting GDD will be on a relative scale, but
            # the downstream mapping is fitted on the same scale and remains
            # consistent for forecasting.
            print(Fore.YELLOW + "[degree_day] WARNING: using possibly "
                  "standardised temperature for GDD computation. To compute "
                  "GDD on physical units, expose 'raw_extra_data' on the "
                  "dataset before covariate normalisation." + Fore.RESET)
            raw_temp = dataset.extra_data[temp_key]
        if isinstance(raw_temp, pd.DataFrame):
            temp_arr = raw_temp.values.astype(float)
            timestamps = raw_temp.index
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
            value=gdd_df,
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
    transform = {
        'target': StandardScaler(axis=scale_axis),
    }

    # update config with wandb config: batch_size
    if 'batch_size' in wandb.config.keys():
        cfg.batch_size = wandb.config['batch_size']
        print(Fore.GREEN + f"Updated batch size: {cfg.batch_size}")

    if "epochs" in wandb.config.keys():
        cfg.epochs = wandb.config['epochs']
        print(Fore.GREEN + f"Updated epochs: {cfg.epochs}")

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

    if covariates is not None:
        length_u = len(u) + 1 # +1 for the mask
    else:
        length_u = 1

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=input_size + length_u,
                        exog_size= length_u if cfg.model.name == "arimax" else 0,  
                        output_size=torch_dataset.n_channels + 2 if cfg.dataset.add_second_target else torch_dataset.n_channels,
                        weighted_graph=torch_dataset.edge_weight is not None,
                        embedding_cfg=cfg.get('embedding'), #### changed from None to embedding_cfg
                        horizon=torch_dataset.horizon,
                        add_second_target=cfg.dataset.add_second_target,
                        use_node_embeddings=cfg.model.hparams.get('use_node_embeddings', False)
                        )
    
    if model is not None:
        model.filter_model_args_(model_kwargs)

    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    if "loss_fn" in wandb.config:
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

    loss_fn = base_loss_fn           # default when there is just one target

    # --- add the second-target head (regression + classification) -----------------
    if cfg.dataset.add_second_target:

        alpha = cfg.optimizer.alpha               # cache for speed/readability

        # wrap in the masked-metric adaptor (if you need NaN/∞ masking)
        loss_fn_classification = MaskedCategoricalCrossEntropy(mask_nans=True, mask_inf=True)

        loss_fn = {"loss_regression": base_loss_fn,
                   "loss_classification": loss_fn_classification,
                   "alpha": alpha}

    log_list = cfg.dataset.log_metrics

    log_metrics = MetricsLogger()

    metrics = log_metrics.filter_metrics(log_list)

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup predictor
    if cfg.model.name == "arimax":
        predictor_fn = ARIMAXWrapper
    elif cfg.model.name == "degree_day":
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

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run_dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )
    
    # ---------------- wandb run init (per-model config payload) ------------
    if cfg.model.name == "arimax":
        run = wandb.init(
                    entity=cfg.wandb.entity,
                    project=cfg.wandb.project,
                    config={
                        "p": cfg.model.hparams.p,
                        "d": cfg.model.hparams.d,
                        "q": cfg.model.hparams.q,
                        "n_exog": len(u) + 1 if covariates is not None else 0,  # +1 for the mask
                        "trend": cfg.model.hparams.trend,
                        "enforce_stationarity": cfg.model.hparams.enforce_stationarity,
                        "enforce_invertibility": cfg.model.hparams.enforce_invertibility,
                        "dataset": cfg.dataset.name,
                        "epochs": cfg.epochs,
                        "window": cfg.dataset.window,
                        "horizon": cfg.dataset.horizon
                    },
                )
    elif cfg.model.name == "degree_day":
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
                        "epochs": cfg.epochs,
                        "window": cfg.dataset.window,
                        "horizon": cfg.dataset.horizon
                    },
                )

    wandb_logger_callback = Wandb_callback(
        log_dir=cfg.run_dir,
        run=run,
        log_metrics=log_list,
    )

    # ---------------- Trainer construction ---------------------------------
    # ARIMAX evaluates out-of-band; every other model (including persistent and
    # degree_day) uses Trainer.test() with its own test_step.
    if cfg.model.name != "arimax":
        trainer = Trainer(max_epochs=cfg.epochs,
                          limit_train_batches=cfg.train_batches,
                          default_root_dir=cfg.run_dir,
                          logger=exp_logger,
                          accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                          gradient_clip_val=cfg.grad_clip_val,
                          callbacks=[early_stop_callback, checkpoint_callback, wandb_logger_callback])

    # ---------------- ARIMAX out-of-band fit -------------------------------
    if cfg.model.name == "arimax":
        # import numpy as np

        all_features, all_targets = [], []
        train_dataloader = data_module.train_dataloader()

        for batch in train_dataloader:
            X_batch, y_batch = batch.input, batch.target
            if cfg.dataset.add_covariates:
                x = torch.concatenate([X_batch.x, X_batch.u], axis=-1)
            else:
                x = X_batch.x
            all_features.append(x.numpy())
            all_targets.append(y_batch.y.numpy())

        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_targets, axis=0)

        model = fit_arimax(
            y_tr = y.reshape(-1, torch_dataset.n_nodes),
            X_tr = X[:, -1].reshape(-1, torch_dataset.n_nodes, X.shape[-1]),
            spec = cfg.model.hparams,
            n_nodes = torch_dataset.n_nodes
        )
        print("ARIMAX model fitted.")

        predictor.arimax_models = model
        predictor.horizon = torch_dataset.horizon
        predictor.delay = torch_dataset.delay

    # ---------------- Degree-Day out-of-band fit ---------------------------
    elif cfg.model.name == "degree_day":
        # import numpy as np

        # Pull the full target + mask arrays and slice the training portion.
        y_full = dataset.dataframe().values.astype(np.float32)
        mask_full = dataset.mask.astype(bool) if dataset.mask is not None \
            else np.ones_like(y_full, dtype=bool)

        # data_module.train_slice is a boolean mask or slice along the time axis.
        train_slice = data_module.train_slice
        gdd_train = gdd_full[train_slice]
        y_train = y_full[train_slice]
        mask_train = mask_full[train_slice]

        fitted = fit_degree_day(
            gdd_train=gdd_train,
            y_train=y_train,
            mapping=cfg.model.hparams.mapping,
            pooled=bool(cfg.model.hparams.pooled),
            mask=mask_train,
        )
        predictor.set_fitted(fitted)

        # Log a compact summary of the fit
        summary = [m.summary() for m in fitted[: min(5, len(fitted))]]
        print(Fore.GREEN + f"[degree_day] Fit complete "
              f"(mapping={cfg.model.hparams.mapping}, "
              f"pooled={cfg.model.hparams.pooled}). "
              f"First-node summaries: {summary}" + Fore.RESET)
        if run is not None:
            run.log({"degree_day/n_nodes_fitted": len(fitted)})

    # ---------------- Deep-learning training -------------------------------
    if cfg.model.name not in NON_TRAINABLE_MODELS:
        load_model_path = cfg.get('load_model_path')
        if load_model_path is not None:
            predictor.load_model(load_model_path)
        else:
            trainer.fit(predictor, train_dataloaders=data_module.train_dataloader(),
                        val_dataloaders=data_module.val_dataloader())
            predictor.load_model(checkpoint_callback.best_model_path)
    elif cfg.model.name == "persistent":
        pass  # nothing to fit

    ########################################
    # testing                              #
    ########################################

    predictor.freeze()

    # ARIMAX is still evaluated out-of-band (e.g. in plot_predictions_test).
    # Degree-Day uses Lightning's test loop via DegreeDayWrapper.test_step.
    if cfg.model.name not in ("arimax",):
        trainer.test(predictor, dataloaders=data_module.test_dataloader())
    
    exp_logger.finalize('success')

    # if cfg.dataset.spatial_information:
    #     plot_predictions_test(
    #         predictor=predictor,
    #         data_module=data_module,
    #         run_dir=cfg.run_dir,
    #         model_type= "arimax" if cfg.model.name == "arimax" else "dl",
    #         log_metrics=log_list,
    #         delay=cfg.dataset.delay,
    #         wandb_run=run,
    #     )
    # else:
    #     plot_predictions_test_single(
    #         predictor=predictor,
    #         data_module=data_module,
    #         run_dir=cfg.run_dir,
    #         model_type= "arimax" if cfg.model.name == "arimax" else "dl",
    #         log_metrics=log_list,
    #         delay=cfg.dataset.delay,
    #         wandb_run=run,
    #     )

    

if __name__ == "__main__":
    main()
