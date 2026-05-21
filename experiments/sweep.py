import torch
import hydra
from zoneinfo import ZoneInfo
from omegaconf import DictConfig, OmegaConf
import wandb
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from extras.sampling_st_datamodule import SamplingSTDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.metrics import torch_metrics

from datasets.ostrinia import Ostrinia
from datasets.peakweather import PeakWeatherTSL
from datasets.ostrinia_simple_synth import OstriniaSimpleSynth

from models.gru import GRU
from models.dcrnn import DCRNNModel
from models.grugcn import GRUGCN
from models.gat import GAT
from models.lstm import LSTM
from models.mlp import MLP
from models.transformer_encoder import TransformerBlock
from models.transformer_spatial import TransformerSpatial

from extras.predictor import WrapPredictor, WrapPredictorDoubleTarget
from extras.metrics_logging import MetricsLogger
from extras.callbacks import Wandb_callback, MetricsHistory
from extras.plots import plot_predictions_test
from extras.nmse_loss import MaskedNMSE
from extras.masked_categorical_CE import MaskedCategoricalCrossEntropy


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from numpy import concatenate, isnan, nan_to_num

from datetime import datetime

import yaml
import sys

from colorama import Fore, Style, init
init()

# Register the custom resolver
OmegaConf.register_resolver(
    "now", 
    lambda fmt: datetime.datetime.now(tz=ZoneInfo('Europe/Berlin')).strftime(fmt)
)

def get_model(name):
    if name == 'dcrnn':
        return DCRNNModel
    elif name == 'grugcn':
        return GRUGCN
    elif name == 'gru':
        return GRU
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

    wandb.init()

    #######################################
    # dataset Initialization
    #######################################
    if cfg.dataset.name == 'ostrinia':
        dataset = Ostrinia(root = "datasets",
                       target = cfg.dataset.target,
                       smooth = cfg.dataset.smooth,
                       full_normalization = cfg.dataset.full_normalization,
                       add_second_target = cfg.dataset.add_second_target)
    elif cfg.dataset.name == 'peakweather':
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

    adj = dataset.get_connectivity(**cfg.dataset.connectivity,
                                   train_slice=data_module.train_slice)
    
    data_module.torch_dataset.set_connectivity(adj)

    ######################################
    # Model Initialization
    ######################################
    model = get_model(cfg.model.name)

    # if cfg.model.name == 'gat':
    #     cfg.optimizer.batch_size = cfg.optimizer.batch_size // 4

    if cfg.dataset.add_covariates and list(dataset.extra_data.keys()):
        length_u = len(u) + 1 # +1 for the mask
    else:
        length_u = 1

    if cfg.model.name == 'gat':
        print(Fore.GREEN + f"Using GAT model with {cfg.model.hparams.n_heads} heads")

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=input_size + length_u,
                        exog_size=0,
                        output_size=torch_dataset.n_channels + 2 if cfg.dataset.add_second_target else torch_dataset.n_channels,
                        weighted_graph=torch_dataset.edge_weight is not None,
                        embedding_cfg=cfg.get('embedding'), #### changed from None to embedding_cfg
                        horizon=torch_dataset.horizon,
                        add_second_target=cfg.dataset.add_second_target,
                        use_node_embeddings=cfg.model.hparams.use_node_embeddings
                        )
    
    model.filter_model_args_(model_kwargs)

    model_kwargs.update(cfg.model.hparams)

    # update with wandb config
    for key in wandb.config.keys():
        if key in model_kwargs.keys():
            model_kwargs[key] = wandb.config[key]
            cfg.model.hparams[key] = wandb.config[key]
            print(Fore.GREEN + f"Updated model kwargs: {key} = {wandb.config[key]}")
        else:
            print(Fore.RED + f"Key {key} not found in model kwargs, skipping.")

    if cfg.model.name == 'grugcn':
        print(Fore.GREEN + f"Using GRUGCN model with {model_kwargs['n_layers_rnn']} RNN layers and {model_kwargs['n_layers_gnn']} GNN layers.")

    ########################################
    # predictor                            #
    ########################################

    if "loss_fn" in wandb.config.keys():
        cfg.optimizer.loss_fn = wandb.config['loss_fn']
        print(Fore.GREEN + f"Updated loss function: {cfg.optimizer.loss_fn}")
        
    if cfg.optimizer.loss_fn == 'mae':
        base_loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)
    elif cfg.optimizer.loss_fn == 'mse':
        base_loss_fn = torch_metrics.MaskedMSE(compute_on_step=True)
    elif cfg.optimizer.loss_fn == 'nmse':
        base_loss_fn = MaskedNMSE()
    elif cfg.optimizer.loss_fn == 'mape':
        base_loss_fn = torch_metrics.MaskedMAPE(compute_on_step=True)

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

    optimizer_kwargs = dict(cfg.optimizer.hparams)

    # update optimizer kwargs with wandb config
    if 'optimizer' in wandb.config.keys():
        cfg.optimizer.name = wandb.config['optimizer']
        print(Fore.GREEN + f"Updated optimizer: {cfg.optimizer}")

    for key in wandb.config.keys():
        if key in optimizer_kwargs.keys():
            optimizer_kwargs[key] = wandb.config[key]
            cfg.optimizer.hparams[key] = wandb.config[key]
            print(Fore.GREEN + f"Updated optimizer kwargs: {key} = {wandb.config[key]}")
        else:
            print(Fore.RED + f"Key \'{key}\' not found in optimizer kwargs, skipping.")

    # setup predictor
    if cfg.dataset.add_second_target:
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
    
    config = {
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
            }
    
    if cfg.model.name == 'grugcn':
        config.update({
            "n_layers_rnn": cfg.model.hparams.n_layers_rnn,
            "n_layers_gnn": cfg.model.hparams.n_layers_gnn,
        })

    run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity=cfg.wandb.entity,
                # Set the wandb project where this run will be logged.
                project=cfg.wandb.project,
                # Track hyperparameters and run metadata.
                config=config
            )
    
    wandb_logger_callback = Wandb_callback(
        log_dir=cfg.run_dir,
        run=run,
        log_metrics=log_list,
    )

    if cfg.get('grad_clip_val') is not None:
        print(Fore.GREEN + f"Using gradient clipping with value {cfg.grad_clip_val}")
    else:
        print(Fore.YELLOW + "Not using gradient clipping")

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      default_root_dir=cfg.run_dir,
                      logger=exp_logger,  # Disable default logger
                      accelerator= 'gpu' if torch.cuda.is_available() else 'cpu',
                      devices= 1 if torch.cuda.is_available() else None,
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback, wandb_logger_callback])

    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        trainer.fit(predictor, train_dataloaders=data_module.train_dataloader(),
                    val_dataloaders=data_module.val_dataloader())
        predictor.load_model(checkpoint_callback.best_model_path)

    ########################################
    # testing                              #
    ########################################

    predictor.freeze()
    trainer.test(predictor, dataloaders=data_module.test_dataloader())
    
    exp_logger.finalize('success')
    
    # plot_predictions_test(
    #     predictor=predictor,
    #     data_module=data_module,
    #     run_dir=cfg.run_dir,
    #     log_metrics=log_list, 
    #     wandb_run=run,
    # )

def wandb_sweep():
    """
    Run a sweep with wandb.
    """

    # get model name from command line arguments
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'gru'

    # get model from argv
    if "grugcn" in model_name:
        wandb_sweep_path = "./config/wandb/sweep_grugcn.yaml"
    elif 'arimax' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_arimax.yaml"
    elif 'gat' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_gat.yaml"
    elif "transformer_spatial" in model_name:
        wandb_sweep_path = "./config/wandb/sweep_transformer_spatial.yaml"
    elif 'transformer' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_transformer.yaml"
    else:
        wandb_sweep_path = "./config/wandb/sweep.yaml"

    wandb_keys_path = "./config/wandb/keys.yaml"
    with open(wandb_sweep_path, 'r') as f:
        dict_sweep = yaml.safe_load(f)
    with open(wandb_keys_path, 'r') as f:
        dict_keys = yaml.safe_load(f)

    dict_sweep = dict_sweep['sweep']

    wandb.login(key=dict_keys['key'])

    for key in dict_sweep.keys():
        print(f"Setting sweep parameter {key} to {dict_sweep[key]}")

    # make name of sween name of model + date
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    dict_sweep['name'] = f"sweep_{model_name}_{date_str}"

    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=dict_sweep,
        project=dict_keys['project'],
        entity=dict_keys['entity'],
    )

    # Start the sweep
    wandb.agent(sweep_id, function=main, count=dict_sweep['count'])

if __name__ == "__main__":
    wandb_sweep()