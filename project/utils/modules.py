# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from NeiA-PyTorch (https://github.com/amorehead/NeiA-PyTorch):
# -------------------------------------------------------------------------------------------------------------------------------------

from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from torch.optim import Adam

from project.utils.training_utils import construct_interact_tensor


# ------------------
# PyTorch Modules
# ------------------
class NeiAGraphConv(nn.Module):
    """A neighborhood-averaging graph neural network layer as a PyTorch module.

    NeiAGraphConv stands for a Graph Convolution neighborhood-averaging layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph conv layer in a GCN.
    """

    def __init__(
            self,
            num_node_feats: int,
            num_edge_feats: int,
            nbrhd_size: int,
            activ_fn=nn.Tanh(),
            dropout=0.1,
            **kwargs
    ):
        """Neighborhood-Averaging Graph Conv Layer

        Parameters
        ----------
        num_node_feats : int
            Input node feature size.
        num_edge_feats : int
            Input edge feature size.
        nbrhd_size : int
            The size of each residue's receptive field for feature updates.
        activ_fn : Module
            Activation function to apply in MLPs.
        dropout : float
            How much dropout (forget rate) to apply before activation functions.
        """
        super().__init__()

        # Initialize shared layer variables
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.nbrhd_size = nbrhd_size
        self.activ_fn = activ_fn
        self.dropout = dropout

        # Define weight matrix for neighboring node feature matrix (i.e. W^N for H_i)
        self.W_N = nn.Linear(self.num_node_feats, self.num_node_feats, bias=False)

        # Define weight matrix for neighboring edge feature matrix (i.e. W^E for E_i)
        self.W_E = nn.Linear(self.num_edge_feats, self.num_node_feats, bias=False)

    def forward(self, X: torch.Tensor, H: torch.Tensor, E: torch.Tensor, device: torch.device):
        """Forward pass of the network

        Parameters
        ----------
        X : Tensor
            Tensor of node features to update with graph convolutions.
        H : Tensor
            Tensor of neighboring node features with which to convolve.
        E : Tensor
            Tensor of neighboring edge features with which to convolve.
        device : torch.device
            Computation device (e.g. CPU or GPU) on which to collect tensors.
        """

        # Create neighbor node signals
        H_sig = self.W_N.weight.matmul(H.transpose(1, 2))  # (n_node_feats, n_nodes)
        assert not H_sig.isnan().any()

        # Create neighbor in-edge signals
        E_sig = self.W_E.weight.matmul(E.transpose(1, 2))  # (n_nodes, n_node_feats, n_nbrs)
        assert not E_sig.isnan().any()

        # Create combined neighbor node + in-edge signals
        Z = self.activ_fn(H_sig + E_sig)
        assert not Z.isnan().any()

        # Average each learned feature vector in each sub-tensor Z_i corresponding to node i
        Z_avg = Z / Z.shape[-1]
        Z_sig = Z_avg.matmul(torch.ones(Z.shape[-1], device=device))
        Z_sig = F.dropout(Z_sig, p=self.dropout, training=self.training).squeeze()  # Remove "1" leftover from 'q'
        assert not Z_sig.isnan().any()

        # Apply a residual node feature update with the learned matrix Z after performing neighborhood averaging
        X += Z_sig
        assert not X.isnan().any()
        
        # Update node features of original graph via an updated subgraph
        return X

    def __repr__(self):
        return f'NeiAGraphConv(structure=h_in{self.num_node_feats}-h_hid{self.num_node_feats}' \
               f'-h_out{self.num_node_feats}-h_e{self.num_edge_feats})'


class NeiWAGraphConv(nn.Module):
    """A neighborhood weighted-averaging graph neural network layer as a PyTorch module.

    NeiWAGraphConv stands for a Graph Convolution neighborhood weighted-averaging layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph conv layer in a GCN.
    """

    def __init__(
            self,
            num_node_feats: int,
            num_edge_feats: int,
            nbrhd_size: int,
            activ_fn=nn.Tanh(),
            dropout=0.3,
            **kwargs
    ):
        """Neighborhood Weighted-Averaging Graph Conv Layer

        Parameters
        ----------
        num_node_feats : int
            Input node feature size.
        num_edge_feats : int
            Input edge feature size.
        nbrhd_size : int
            The size of each residue's receptive field for feature updates.
        activ_fn : Module
            Activation function to apply in MLPs.
        dropout : float
            How much dropout (forget rate) to apply before activation functions.
        """
        super().__init__()

        # Initialize shared layer variables
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.nbrhd_size = nbrhd_size
        self.activ_fn = activ_fn
        self.dropout = dropout

        # Define weight matrix for neighboring node feature matrix (i.e. W^N for H_i)
        self.W_N = nn.Linear(self.num_node_feats, self.num_node_feats, bias=False)

        # Define weight matrix for neighboring edge feature matrix (i.e. W^E for E_i)
        self.W_E = nn.Linear(self.num_edge_feats, self.num_node_feats, bias=False)

        # Define weight vector for neighboring node-edge matrix (i.e. the q in 'a = softmax(Z^T matmul q)')
        self.q = nn.Linear(1, self.num_node_feats, bias=False)

    def forward(self, X: torch.Tensor, H: torch.Tensor, E: torch.Tensor, device: torch.device):
        """Forward pass of the network

        Parameters
        ----------
        X : Tensor
            Tensor of node features to update with graph convolutions.
        H : Tensor
            Tensor of neighboring node features with which to convolve.
        E : Tensor
            Tensor of neighboring edge features with which to convolve.
        device : torch.device
            Computation device (e.g. CPU or GPU) on which to collect tensors.
        """

        # Create neighbor node signals
        H_sig = self.W_N.weight.matmul(H.transpose(1, 2))  # (n_node_feats, n_nodes)

        # Create neighbor in-edge signals
        E_sig = self.W_E.weight.matmul(E.transpose(1, 2))  # (n_nodes, n_node_feats, n_nbrs)

        # Create combined neighbor node + in-edge signals
        Z = self.activ_fn(H_sig + E_sig)

        # Calculate weight vector for neighboring node-edge features (i.e. the a in 'a = softmax(Z^T matmul q)')
        a = torch.softmax(Z.transpose(1, 2).matmul(self.q.weight), dim=0)  # Element-wise softmax each row

        # Average each learned feature vector in each sub-tensor Z_i corresponding to node i
        Z_avg = Z / Z.shape[-1]
        Z_sig = Z_avg.matmul(a)
        Z_sig = F.dropout(Z_sig, p=self.dropout, training=self.training).squeeze()  # Remove "1" leftover from 'q'

        # Apply a residual node feature update with the learned matrix Z after performing neighborhood averaging
        X += Z_sig.squeeze()  # Remove trivial "1" dimension leftover from the vector q's definition

        # Update node features of original graph via an updated subgraph
        return X

    def __repr__(self):
        return f'NeiWAGraphConv(structure=h_in{self.num_node_feats}-h_hid{self.num_node_feats}' \
               f'-h_out{self.num_node_feats}-h_e{self.num_edge_feats})'


# ------------------
# Lightning Modules
# ------------------
class LitNeiA(pl.LightningModule):
    """A siamese neighborhood-averaging (NeiA) module."""

    def __init__(self, num_node_input_feats: int, num_edge_input_feats: int, gnn_activ_fn=nn.Tanh(),
                 interact_activ_fn=nn.ReLU(), num_classes=2, weighted_avg=False, num_gnn_layers=1,
                 num_interact_layers=3, num_interact_hidden_channels=214, num_epochs=50, pn_ratio=0.1, knn=20,
                 dropout_rate=0.3, metric_to_track='val_bce', weight_decay=1e-7, batch_size=32, lr=1e-5,
                 multi_gpu_backend="ddp"):
        """Initialize all the parameters for a LitNeiA module."""
        super().__init__()

        # Build the network
        self.num_node_input_feats = num_node_input_feats
        self.num_edge_input_feats = num_edge_input_feats
        self.gnn_activ_fn = gnn_activ_fn
        self.interact_activ_fn = interact_activ_fn
        self.num_classes = num_classes
        self.weighted_avg = weighted_avg

        # GNN module's keyword arguments provided via the command line
        self.num_gnn_layers = num_gnn_layers

        # Interaction module's keyword arguments provided via the command line
        self.num_interact_layers = num_interact_layers
        self.num_interact_hidden_channels = num_interact_hidden_channels

        # Model hyperparameter keyword arguments provided via the command line
        self.num_epochs = num_epochs
        self.pn_ratio = pn_ratio
        self.nbrhd_size = knn
        self.dropout_rate = dropout_rate
        self.metric_to_track = metric_to_track
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr = lr
        self.multi_gpu_backend = multi_gpu_backend

        # Assemble the layers of the network
        self.gnn_block = self.build_gnn_block()
        self.init_res_block, self.interim_res_blocks, self.final_res_block, self.final_conv_layer = self.build_i_block()

        # Declare loss functions and metrics for training, validation, and testing
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.test_auroc = tm.AUROC(average='weighted', pos_label=1)
        self.test_auprc = tm.AveragePrecision(pos_label=1)
        self.test_acc = tm.Accuracy(average='weighted', num_classes=self.num_classes, multiclass=True)
        self.test_f1 = tm.F1(average='weighted', num_classes=self.num_classes, multiclass=True)

        # Log hyperparameters
        self.save_hyperparameters()

    def build_gnn_block(self):
        """Define the layers for all NeiA GNN modules."""
        # Marshal all GNN layers, allowing the user to choose which kind of neighborhood averaging they would like
        if self.weighted_avg:
            gnn_layer = (NeiWAGraphConv(num_node_feats=self.num_node_input_feats,
                                        num_edge_feats=self.num_edge_input_feats,
                                        nbrhd_size=self.nbrhd_size,
                                        activ_fn=self.gnn_activ_fn,
                                        dropout=self.dropout_rate))
        else:
            gnn_layer = (NeiAGraphConv(num_node_feats=self.num_node_input_feats,
                                       num_edge_feats=self.num_edge_input_feats,
                                       nbrhd_size=self.nbrhd_size,
                                       activ_fn=self.gnn_activ_fn,
                                       dropout=self.dropout_rate))
        gnn_layers = [gnn_layer for _ in range(self.num_gnn_layers)]
        return nn.ModuleList(gnn_layers)

    def get_res_block(self):
        """Retrieve a residual block of a specific type (e.g. ResNet)."""
        res_block = nn.ModuleList([
            nn.Conv2d(in_channels=self.num_interact_hidden_channels,
                      out_channels=self.num_interact_hidden_channels,
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            self.interact_activ_fn,
            nn.Conv2d(in_channels=self.num_interact_hidden_channels,
                      out_channels=self.num_interact_hidden_channels,
                      kernel_size=(3, 3),
                      padding=(1, 1)),
        ])
        return res_block

    def build_i_block(self):
        """Define the layers of the interaction block for an interaction tensor."""
        # Marshal all interaction layers, beginning with the initial residual block
        init_res_block = nn.ModuleList([
            nn.Conv2d(self.num_node_input_feats * 2,
                      self.num_interact_hidden_channels,
                      kernel_size=(1, 1),
                      padding=(0, 0)),
            self.interact_activ_fn,
            nn.Conv2d(self.num_interact_hidden_channels,
                      self.num_interact_hidden_channels,
                      kernel_size=(1, 1),
                      padding=(0, 0)),
        ])
        # Unroll requested number of intermediate residual blocks
        interim_res_blocks = []
        for _ in range(self.num_interact_layers - 2):
            interim_res_block = self.get_res_block()
            interim_res_blocks.append(interim_res_block)
        interim_res_blocks = nn.ModuleList(interim_res_blocks)
        # Attach final residual block to project channel dimensionality down to original size
        final_res_block = nn.ModuleList([
            nn.Conv2d(self.num_interact_hidden_channels,
                      self.num_interact_hidden_channels,
                      kernel_size=(1, 1),
                      padding=(0, 0)),
            self.interact_activ_fn,
            nn.Conv2d(self.num_interact_hidden_channels,
                      self.num_node_input_feats * 2,
                      kernel_size=(1, 1),
                      padding=(0, 0))
        ])
        # Craft final convolution layer to project channel dimensionality down to 1, the target number of channels
        final_conv_layer = nn.Conv2d(self.num_node_input_feats * 2, 1, kernel_size=(1, 1), padding=(0, 0))
        return init_res_block, interim_res_blocks, final_res_block, final_conv_layer

    # ---------------------
    # Training
    # ---------------------
    def gnn_forward(self, node_feats: torch.Tensor, nbrhd_node_feats: torch.Tensor,
                    nbrhd_edge_feats: torch.Tensor, gnn_layer_id: int):
        """Make a forward pass through a single GNN layer."""
        # Convolve over graph nodes and edges
        new_node_feats = self.gnn_block[gnn_layer_id](node_feats, nbrhd_node_feats, nbrhd_edge_feats, self.device)
        return new_node_feats

    def interact_forward(self, interact_tensor: torch.Tensor):
        """Make a forward pass through the interaction blocks."""
        residual, logits = interact_tensor, interact_tensor
        # Convolve over the 3D "interaction" tensor given using the initial residual block
        for layer in self.init_res_block:
            logits = layer(logits)
        logits += residual
        residual = logits
        logits = self.interact_activ_fn(logits)
        # Convolve over the 3D "interaction" logits using the interim residual blocks
        for interim_res_block in self.interim_res_blocks:
            for layer in interim_res_block:
                logits = layer(logits)
            logits += residual
            residual = logits
            logits = self.interact_activ_fn(logits)
        # Convolve over the 3D "interaction" tensor given using the final residual block
        for layer in self.final_res_block:
            logits = layer(logits)
        logits += residual
        logits = self.interact_activ_fn(logits)
        # Project number of channels down to target size, 1
        logits = self.final_conv_layer(logits)
        return logits

    def forward(self, cmplx: dict, labels: torch.Tensor):
        """Make a forward pass through the entire siamese network."""
        # Make a copy of the complex's feature and index tensors to prevent feature overflow between epochs
        graph1_node_feats = cmplx['graph1_node_feats'].clone()
        graph2_node_feats = cmplx['graph2_node_feats'].clone()
        graph1_nbrhd_indices = cmplx['graph1_nbrhd_indices'].clone()
        graph2_nbrhd_indices = cmplx['graph2_nbrhd_indices'].clone()
        graph1_nbrhd_node_feats = cmplx['graph1_node_feats'].clone()
        graph2_nbrhd_node_feats = cmplx['graph2_node_feats'].clone()
        graph1_nbrhd_edge_feats = cmplx['graph1_edge_feats'].clone()
        graph2_nbrhd_edge_feats = cmplx['graph2_edge_feats'].clone()

        # Replace any leftover NaN values in edge features with zero
        if True in torch.isnan(cmplx['graph1_edge_feats']):
            graph1_nbrhd_edge_feats = torch.tensor(
                np.nan_to_num(graph1_nbrhd_edge_feats.cpu().numpy()), device=self.device
            )
        if True in torch.isnan(cmplx['graph2_edge_feats']):
            graph2_nbrhd_edge_feats = torch.tensor(
                np.nan_to_num(graph2_nbrhd_edge_feats.cpu().numpy()), device=self.device
            )

        # Secure layer-specific copy of node features to restrict each node's receptive field to the current hop
        graph1_layer_node_feats = cmplx['graph1_node_feats'].clone()
        graph2_layer_node_feats = cmplx['graph2_node_feats'].clone()
        # Convolve node features using a specified number of GNN layers
        for gnn_layer_id in range(len(self.gnn_block)):
            # Update node features in batches of residue-residue pairs in a node-unique manner
            unique_examples = max(len(torch.unique(labels[:, 0])), len(torch.unique(labels[:, 1])))
            for i in range(int(unique_examples / self.batch_size)):
                index = int(i * self.batch_size)
                # Get a batch of unique node IDs
                batch = labels[index: index + self.batch_size]
                graph1_batch_n_ids, graph2_batch_n_ids = batch[:, 0], batch[:, 1]
                g1_nbrhd_indices = graph1_nbrhd_indices[graph1_batch_n_ids].squeeze()
                g2_nbrhd_indices = graph2_nbrhd_indices[graph2_batch_n_ids].squeeze()
                # Get unique features selected for the batch
                g1_node_feats = graph1_node_feats[graph1_batch_n_ids]
                g2_node_feats = graph2_node_feats[graph2_batch_n_ids]
                g1_nbrhd_node_feats = graph1_nbrhd_node_feats[g1_nbrhd_indices].reshape(
                    -1, g1_nbrhd_indices.shape[-1], graph1_nbrhd_node_feats.shape[-1]
                )
                g2_nbrhd_node_feats = graph2_nbrhd_node_feats[g2_nbrhd_indices].reshape(
                    -1, g2_nbrhd_indices.shape[-1], graph2_nbrhd_node_feats.shape[-1]
                )
                g1_nbrhd_edge_feats = graph1_nbrhd_edge_feats[graph1_batch_n_ids]
                g2_nbrhd_edge_feats = graph2_nbrhd_edge_feats[graph2_batch_n_ids]
                # Forward propagate with weight-shared GNN layers using batch of residues
                updated_node_feats1 = self.gnn_forward(
                    g1_node_feats, g1_nbrhd_node_feats, g1_nbrhd_edge_feats, gnn_layer_id
                )
                updated_node_feats2 = self.gnn_forward(
                    g2_node_feats, g2_nbrhd_node_feats, g2_nbrhd_edge_feats, gnn_layer_id
                )
                # Update original node features according to updated node feature batch
                graph1_layer_node_feats[graph1_batch_n_ids] = updated_node_feats1
                graph2_layer_node_feats[graph2_batch_n_ids] = updated_node_feats2
            # Update original clone of node features for next hop
            graph1_node_feats = graph1_layer_node_feats.clone()
            graph2_node_feats = graph2_layer_node_feats.clone()
            graph1_nbrhd_node_feats = graph1_layer_node_feats.clone()
            graph2_nbrhd_node_feats = graph2_layer_node_feats.clone()

        # Interleave node features from both graphs to achieve the desired interaction tensor
        interact_tensor = construct_interact_tensor(graph1_node_feats, graph2_node_feats)

        # Predict residue-residue pair interactions using a convolution block (i.e. series of residual CNN blocks)
        logits = self.interact_forward(interact_tensor)

        # Return network prediction
        return logits.squeeze()  # Remove any trivial dimensions from logits

    def downsample_examples(self, examples: torch.tensor):
        """Randomly sample enough negative pairs to achieve requested positive-negative class ratio (via shuffling)."""
        examples = examples[torch.randperm(len(examples))]  # Randomly shuffle examples (during training)
        pos_examples = examples[examples[:, 2] == 1]  # Find out how many interacting pairs there are
        num_neg_pairs_to_sample = int(len(pos_examples) / self.pn_ratio)  # Determine negative sample size
        neg_examples = examples[examples[:, 2] == 0][:num_neg_pairs_to_sample]  # Sample negative pairs
        downsampled_examples = torch.cat((pos_examples, neg_examples))
        return downsampled_examples

    def training_step(self, batch, batch_idx):
        """Lightning calls this inside the training loop."""
        # Make a forward pass through the network for a batch of protein complexes
        cmplx = batch[0]
        examples = cmplx['examples']
        examples = self.downsample_examples(examples)
        logits = self(cmplx, examples)
        sampled_indices = examples[:, :2][:, 1] + logits.shape[1] * examples[:, :2][:, 0]  # 1d_idx = x + width * y
        flattened_logits = torch.flatten(logits)
        downsampled_logits = flattened_logits[sampled_indices]

        # Down-weight negative pairs to achieve desired PN weight, leaving positive pairs with a weight of one
        sample_weights = examples[:, 2].float()
        sample_weights[sample_weights == 0] = self.pn_ratio
        loss_fn = nn.BCEWithLogitsLoss(weight=sample_weights)  # Weight each class separately for a given complex
        loss = loss_fn(downsampled_logits, examples[:, 2].float())  # Calculate loss of a single complex

        # Log training step metric(s)
        self.log('train_bce', loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """Lightning calls this inside the validation loop."""
        # Make a forward pass through the network for a batch of protein complexes
        cmplx = batch[0]
        examples = cmplx['examples']
        logits = self(cmplx, examples)
        sampled_indices = examples[:, :2][:, 1] + logits.shape[1] * examples[:, :2][:, 0]  # 1d_idx = x + width * y
        flattened_logits = torch.flatten(logits)
        sampled_logits = flattened_logits[sampled_indices]

        # Calculate the complex loss and metrics
        loss = self.loss_fn(sampled_logits, examples[:, 2].float())  # Calculate loss of a single complex

        # Log validation step metric(s)
        self.log('val_bce', loss, sync_dist=True)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        """Lightning calls this inside the testing loop."""
        # Make a forward pass through the network for a batch of protein complexes
        cmplx = batch[0]
        examples = cmplx['examples']
        logits = self(cmplx, examples)
        sampled_indices = examples[:, :2][:, 1] + logits.shape[1] * examples[:, :2][:, 0]  # 1d_idx = x + width * y
        flattened_logits = torch.flatten(logits)
        sampled_logits = flattened_logits[sampled_indices]

        # Make predictions
        preds = torch.softmax(sampled_logits, dim=0)
        preds_rounded = torch.round(preds)
        int_labels = examples[:, 2].int()

        # Calculate the complex loss and metrics
        loss = self.loss_fn(sampled_logits, examples[:, 2].float())  # Calculate loss of a single complex
        test_acc = self.test_acc(preds_rounded, int_labels)  # Calculate Accuracy of a single complex
        test_f1 = self.test_f1(preds_rounded, int_labels)  # Calculate F1 score of a single complex
        test_auroc = self.test_auroc(preds, int_labels)  # Calculate AUROC of a complex
        test_auprc = self.test_auprc(preds, int_labels)  # Calculate AveragePrecision (i.e. AUPRC) of a complex

        # Log test step metric(s)
        self.log('test_bce', loss, sync_dist=True)

        return {
            'loss': loss, 'test_acc': test_acc, 'test_f1': test_f1, 'test_auroc': test_auroc, 'test_auprc': test_auprc
        }

    def test_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT):
        """Lightning calls this at the end of every test epoch."""
        # Tuplize scores for the current device (e.g. Rank 0)
        test_accs = torch.cat([output_dict['test_acc'].unsqueeze(0) for output_dict in outputs])
        test_f1s = torch.cat([output_dict['test_f1'].unsqueeze(0) for output_dict in outputs])
        test_aurocs = torch.cat([output_dict['test_auroc'].unsqueeze(0) for output_dict in outputs])
        test_auprcs = torch.cat([output_dict['test_auprc'].unsqueeze(0) for output_dict in outputs])
        # Concatenate scores over all devices (e.g. Rank 0 | ... | Rank N) - Warning: Memory Intensive
        test_accs = test_accs if self.multi_gpu_backend in ["dp"] else torch.cat([test_acc for test_acc in self.all_gather(test_accs)])
        test_f1s = test_f1s if self.multi_gpu_backend in ["dp"] else torch.cat([test_f1 for test_f1 in self.all_gather(test_f1s)])
        test_aurocs = test_aurocs if self.multi_gpu_backend in ["dp"] else torch.cat([test_auroc for test_auroc in self.all_gather(test_aurocs)])
        test_auprcs = test_auprcs if self.multi_gpu_backend in ["dp"] else torch.cat([test_auprc for test_auprc in self.all_gather(test_auprcs)])

        # Reset test TorchMetrics for all devices
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_auroc.reset()
        self.test_auprc.reset()

        # When logging only on rank 0, add 'rank_zero_only=True' to avoid deadlocks on synchronization
        if self.trainer.is_global_zero:
            self.log('med_test_acc', torch.median(test_accs), rank_zero_only=True)  # Log MedAccuracy of an epoch
            self.log('med_test_f1', torch.median(test_f1s), rank_zero_only=True)  # Log MedF1 of an epoch
            self.log('med_test_auroc', torch.median(test_aurocs), rank_zero_only=True)  # Log MedAUROC of an epoch
            self.log('med_test_auprc', torch.median(test_auprcs), rank_zero_only=True)  # Log epoch MedAveragePrecision

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # -----------------
        # Model arguments
        # -----------------
        parser.add_argument('--num_interact_hidden_channels', type=int, default=214,
                            help='Dimensionality of interaction module filters')
        return parser
