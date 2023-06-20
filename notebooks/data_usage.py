# %% [markdown]
# # Example of data usage

# %% [markdown]
# ### Neural network model training

# %%
# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from NeiA-PyTorch (https://github.com/amorehead/NeiA-PyTorch):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.plugins import DDPPlugin

from project.datasets.DB5.db5_dgl_data_module import DB5DGLDataModule
from project.utils.modules import LitNeiA
from project.utils.training_utils import collect_args, process_args, construct_pl_logger

# %%
def main(args):
    # -----------
    # Data
    # -----------
    # Load Docking Benchmark 5 (DB5) data module
    db5_data_module = DB5DGLDataModule(data_dir=args.db5_data_dir,
                                       batch_size=args.batch_size,
                                       num_dataloader_workers=args.num_workers,
                                       knn=args.knn,
                                       self_loops=args.self_loops,
                                       percent_to_use=args.db5_percent_to_use,
                                       process_complexes=args.process_complexes,
                                       input_indep=args.input_indep)
    db5_data_module.setup()

    # ------------
    # Model
    # ------------
    # Assemble a dictionary of model arguments
    dict_args = vars(args)
    use_wandb_logger = args.logger_name.lower() == 'wandb'  # Determine whether the user requested to use WandB

    # Pick model and supply it with a dictionary of arguments
    if args.model_name.lower() == 'neiwa':  # Neighborhood Weighted Average (NeiWA)
        model = LitNeiA(num_node_input_feats=db5_data_module.db5_test.num_node_features,
                        num_edge_input_feats=db5_data_module.db5_test.num_edge_features,
                        gnn_activ_fn=nn.Tanh(),
                        interact_activ_fn=nn.ReLU(),
                        num_classes=db5_data_module.db5_test.num_classes,
                        weighted_avg=True,  # Use the neighborhood weighted average variant of NeiA
                        num_gnn_layers=dict_args['num_gnn_layers'],
                        num_interact_layers=dict_args['num_interact_layers'],
                        num_interact_hidden_channels=dict_args['num_interact_hidden_channels'],
                        num_epochs=dict_args['num_epochs'],
                        pn_ratio=dict_args['pn_ratio'],
                        knn=dict_args['knn'],
                        dropout_rate=dict_args['dropout_rate'],
                        metric_to_track=dict_args['metric_to_track'],
                        weight_decay=dict_args['weight_decay'],
                        batch_size=dict_args['batch_size'],
                        lr=dict_args['lr'],
                        multi_gpu_backend=dict_args["accelerator"])
        args.experiment_name = f'LitNeiWA-b{args.batch_size}-gl{args.num_gnn_layers}' \
                               f'-n{db5_data_module.db5_test.num_node_features}' \
                               f'-e{db5_data_module.db5_test.num_edge_features}' \
                               f'-il{args.num_interact_layers}-i{args.num_interact_hidden_channels}' \
            if not args.experiment_name \
            else args.experiment_name
        template_ckpt_filename = 'LitNeiWA-{epoch:02d}-{val_ce:.2f}'

    else:  # Default Model - Neighborhood Average (NeiA)
        model = LitNeiA(num_node_input_feats=db5_data_module.db5_test.num_node_features,
                        num_edge_input_feats=db5_data_module.db5_test.num_edge_features,
                        gnn_activ_fn=nn.Tanh(),
                        interact_activ_fn=nn.ReLU(),
                        num_classes=db5_data_module.db5_test.num_classes,
                        weighted_avg=False,
                        num_gnn_layers=dict_args['num_gnn_layers'],
                        num_interact_layers=dict_args['num_interact_layers'],
                        num_interact_hidden_channels=dict_args['num_interact_hidden_channels'],
                        num_epochs=dict_args['num_epochs'],
                        pn_ratio=dict_args['pn_ratio'],
                        knn=dict_args['knn'],
                        dropout_rate=dict_args['dropout_rate'],
                        metric_to_track=dict_args['metric_to_track'],
                        weight_decay=dict_args['weight_decay'],
                        batch_size=dict_args['batch_size'],
                        lr=dict_args['lr'],
                        multi_gpu_backend=dict_args["accelerator"])
        args.experiment_name = f'LitNeiA-b{args.batch_size}-gl{args.num_gnn_layers}' \
                               f'-n{db5_data_module.db5_test.num_node_features}' \
                               f'-e{db5_data_module.db5_test.num_edge_features}' \
                               f'-il{args.num_interact_layers}-i{args.num_interact_hidden_channels}' \
            if not args.experiment_name \
            else args.experiment_name
        template_ckpt_filename = 'LitNeiA-{epoch:02d}-{val_ce:.2f}'

    # ------------
    # Checkpoint
    # ------------
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    ckpt_path_exists = os.path.exists(ckpt_path)
    ckpt_provided = args.ckpt_name != '' and ckpt_path_exists
    model = model.load_from_checkpoint(ckpt_path,
                                       use_wandb_logger=use_wandb_logger,
                                       batch_size=args.batch_size,
                                       lr=args.lr,
                                       weight_decay=args.weight_decay,
                                       dropout_rate=args.dropout_rate) if ckpt_provided else model

    # ------------
    # Trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # -------------
    # Learning Rate
    # -------------
    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(model, datamodule=db5_data_module)  # Run learning rate finder
        fig = lr_finder.plot(suggest=True)  # Plot learning rates
        fig.savefig('optimal_lr.pdf')
        fig.show()
        model.hparams.lr = lr_finder.suggestion()  # Save optimal learning rate
        print(f'Optimal learning rate found: {model.hparams.lr}')

    # ------------
    # Logger
    # ------------
    pl_logger = construct_pl_logger(args)  # Log everything to an external logger
    trainer.logger = pl_logger  # Assign specified logger (e.g. TensorBoardLogger) to Trainer instance

    # -----------
    # Callbacks
    # -----------
    # Create and use callbacks
    mode = 'min' if 'ce' in args.metric_to_track else 'max'
    early_stop_callback = pl.callbacks.EarlyStopping(monitor=args.metric_to_track,
                                                     mode=mode,
                                                     min_delta=args.min_delta,
                                                     patience=args.patience)
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor=args.metric_to_track,
        mode=mode,
        verbose=True,
        save_last=True,
        save_top_k=3,
        filename=template_ckpt_filename  # Warning: May cause a race condition if calling trainer.test() with many GPUs
    )
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True)
    trainer.callbacks = [early_stop_callback, ckpt_callback, lr_monitor_callback]

    # ------------
    # Restore
    # ------------
    # If using WandB, download checkpoint artifact from their servers if the checkpoint is not already stored locally
    if use_wandb_logger and args.ckpt_name != '' and not os.path.exists(ckpt_path):
        checkpoint_reference = f'{args.entity}/{args.project_name}/model-{args.run_id}:best'
        artifact = trainer.logger.experiment.use_artifact(checkpoint_reference, type='model')
        artifact_dir = artifact.download()
        model = model.load_from_checkpoint(Path(artifact_dir) / 'model.ckpt',
                                           use_wandb_logger=use_wandb_logger,
                                           batch_size=args.batch_size,
                                           lr=args.lr,
                                           weight_decay=args.weight_decay)

    # -------------
    # Training
    # -------------
    # Train with the provided model and DataModule
    trainer.fit(model=model, datamodule=db5_data_module)

    # -------------
    # Testing
    # -------------
    trainer.test()


# %%
# -----------
# Jupyter
# -----------
# sys.argv = ['']

# %%
# -----------
# Arguments
# -----------
# Collect all arguments
parser = collect_args()

# Parse all known and unknown arguments
args, unparsed_argv = parser.parse_known_args()

# Let the model add what it wants
parser = LitNeiA.add_model_specific_args(parser)

# Re-parse all known and unknown arguments after adding those that are model specific
args, unparsed_argv = parser.parse_known_args()

# TODO: Manually set arguments within a Jupyter notebook from here
args.model_name = "neia"
args.multi_gpu_backend = "dp"
args.db5_data_dir = "project/datasets/DB5/final/raw"
args.process_complexes = True
args.batch_size = 1  # Note: `batch_size` must be `1` for compatibility with the current model implementation

# Set Lightning-specific parameter values before constructing Trainer instance
args.max_time = {'hours': args.max_hours, 'minutes': args.max_minutes}
args.max_epochs = args.num_epochs
args.profiler = args.profiler_method
args.accelerator = args.multi_gpu_backend
args.auto_select_gpus = args.auto_choose_gpus
args.gpus = args.num_gpus
args.num_nodes = args.num_compute_nodes
args.precision = args.gpu_precision
args.accumulate_grad_batches = args.accum_grad_batches
args.gradient_clip_val = args.grad_clip_val
args.gradient_clip_algo = args.grad_clip_algo
args.stochastic_weight_avg = args.stc_weight_avg
args.deterministic = True  # Make LightningModule's training reproducible

# Set plugins for Lightning
args.plugins = [
    # 'ddp_sharded',  # For sharded model training (to reduce GPU requirements)
    # DDPPlugin(find_unused_parameters=False),
]

# Finalize all arguments as necessary
args = process_args(args)

# Begin execution of model training with given args above
main(args)


