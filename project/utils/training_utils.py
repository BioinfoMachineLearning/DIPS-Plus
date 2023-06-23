import logging
import os
import pickle
from argparse import ArgumentParser
from typing import List

import dgl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from dgl.nn import pairwise_squared_distance
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from project.utils.constants import D3TO1, FEAT_COLS, ALLOWABLE_FEATS, DEFAULT_MISSING_HSAAC, HSAAC_DIM


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Atom3D (https://github.com/drorlab/atom3d/blob/master/benchmarking/pytorch_geometric/ppi_dataloader.py):
# -------------------------------------------------------------------------------------------------------------------------------------
def prot_df_to_dgl_graph_feats(df: pd.DataFrame, feat_cols: List, allowable_feats: List[List], knn: int):
    r"""Convert protein in dataframe representation to a graph compatible with DGL, where each node is a residue.

    :param df: Protein structure in dataframe format.
    :type df: pandas.DataFrame
    :param feat_cols: Columns of dataframe in which to find node feature values. For example, for residues use ``feat_cols=["element", ...]`` and for residues use ``feat_cols=["resname", ...], or both!``
    :type feat_cols: list[list[Any]]
    :param allowable_feats: List of lists containing all possible values of node type, to be converted into 1-hot node features.
        Any elements in ``feat_col`` that are not found in ``allowable_feats`` will be added to an appended "unknown" bin (see :func:`atom3d.util.graph.one_of_k_encoding_unk`).
    :param knn: Maximum number of nearest neighbors (i.e. edges) to allow for a given node.
    :type knn: int

    :return: tuple containing
        - knn_graph (dgl.DGLGraph): K-nearest neighbor graph for the structure DataFrame given.

        - pairwise_dists (torch.FloatTensor): Pairwise squared distances for the K-nearest neighbor graph's coordinates.

        - node_coords (torch.FloatTensor): Cartesian coordinates of each node.

        - node_feats (torch.FloatTensor): Features for each node, one-hot encoded by values in ``allowable_feats``.
    :rtype: Tuple
    """
    # Exit early if feat_cols or allowable_feats do not align in dimensionality
    if len(feat_cols) != len(allowable_feats):
        raise Exception('feat_cols does not match the length of allowable_feats')

    # Aggregate structure-based node features
    node_feats = torch.FloatTensor([])
    for i in range(len(feat_cols)):
        # Search through embedded 2D list for allowable values
        feat_vecs = [one_of_k_encoding_unk(feat, allowable_feats[i], feat_cols[i]) for feat in df[feat_cols[i]]]
        one_hot_feat_vecs = torch.FloatTensor(feat_vecs)
        node_feats = torch.cat((node_feats, one_hot_feat_vecs), 1)

    # Organize residue coordinates into a FloatTensor
    node_coords = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float32)

    # Define edges - KNN argument determines whether a residue-residue edge gets created in the resulting graph
    knn_graph = dgl.knn_graph(node_coords, knn)
    pairwise_dists = torch.topk(pairwise_squared_distance(node_coords), knn, 1, largest=False).values

    return knn_graph, pairwise_dists, node_coords, node_feats


def one_of_k_encoding_unk(feat, allowable_set, feat_col):
    """Converts input to 1-hot encoding given a set of (or sets of) allowable values. Additionally maps inputs not in the allowable set to the last element."""
    if len(allowable_set) == 0:  # e.g. RSA values
        return [feat]
    elif len(allowable_set) == 1 and type(allowable_set[0]) == list and len(allowable_set[0]) == 0:  # e.g. HSAAC values
        if len(feat) == 0:
            return DEFAULT_MISSING_HSAAC if feat_col == 'hsaac' else []  # Else means skip encoding amide_norm_vec
        if feat_col == 'hsaac' and len(feat) > HSAAC_DIM:  # Handle for edge case from postprocessing
            return np.array(DEFAULT_MISSING_HSAAC)
        return feat if feat_col == 'hsaac' or feat_col == 'sequence_feats' else []  # Else means skip encoding amide_norm_vec as a node feature
    else:  # e.g. Residue element type values
        if feat not in allowable_set:
            feat = allowable_set[-1]
        return list(map(lambda s: feat == s, allowable_set))


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for NeiA-PyTorch (https://github.com/amorehead/NeiA-PyTorch):
# -------------------------------------------------------------------------------------------------------------------------------------
def construct_filenames_frame_txt_filenames(mode: str, percent_to_use: float, filename_sampling: bool, root: str):
    """Build the file path of the requested filename DataFrame text file."""
    base_txt_filename = f'pairs-postprocessed' if mode == 'full' else f'pairs-postprocessed-{mode}'
    filenames_frame_txt_filename = base_txt_filename + f'-{int(percent_to_use * 100)}%-sampled.txt' \
        if filename_sampling else base_txt_filename + '.txt'
    filenames_frame_txt_filepath = os.path.join(root, filenames_frame_txt_filename)
    return base_txt_filename, filenames_frame_txt_filename, filenames_frame_txt_filepath


def build_filenames_frame_error_message(dataset: str, task: str, filenames_frame_txt_filepath: str):
    """Assemble the standard error message for a corrupt or missing filenames DataFrame text file."""
    return f'Unable to {task} {dataset} filenames text file' \
           f' (i.e. {filenames_frame_txt_filepath}).' \
           f' Please make sure it is downloaded and not corrupted.'


def min_max_normalize_tensor(tensor: torch.Tensor, device=None):
    """Normalize provided tensor to have values be in range [0, 1]."""
    min_value = min(tensor)
    max_value = max(tensor)
    tensor = torch.tensor([(value - min_value) / (max_value - min_value) for value in tensor], device=device)
    return tensor


def convert_df_to_dgl_graph(df: pd.DataFrame, input_file: str, knn: int, self_loops: bool) -> dgl.DGLGraph:
    r""" Transform a given DataFrame of residues into a corresponding DGL graph.

    Parameters
    ----------
    df : pandas.DataFrame
    input_file : str
    knn : int
    self_loops : bool

    Returns
    -------
    :class:`dgl.DGLGraph`

        Graph structure, feature tensors for each node and edge.

...     node_feats = graph.ndata['f']
...     node_coords = graph.ndata['x']
...     edge_weights = graph.edata['w']
...     residue_residue_angles = graph.edata['a']

        - ``ndata['f']``: feature tensors of the nodes
        - ``ndata['x']:`` Cartesian coordinate tensors of the nodes
        - ``ndata['f']``: feature tensors of the edges
    """
    # Derive node features, with edges being defined via a k-nearest neighbors approach and a maximum distance threshold
    struct_df = df[df['atom_name'] == 'CA']
    graph, _, node_coords, node_feats = prot_df_to_dgl_graph_feats(
        struct_df,  # Only use CA atoms when constructing the initial graph
        FEAT_COLS,
        ALLOWABLE_FEATS,
        knn
    )

    # Retrieve src and destination node IDs
    srcs = graph.edges()[0]
    dsts = graph.edges()[1]

    # Remove self-loops (if requested)
    if not self_loops:
        graph = dgl.remove_self_loop(graph)
        srcs = graph.edges()[0]
        dsts = graph.edges()[1]

    # Manually add isolated nodes (i.e. those with no connected edges) to the graph
    if len(node_feats) > graph.number_of_nodes():
        num_of_isolated_nodes = len(node_feats) - graph.number_of_nodes()
        raise Exception(f'{num_of_isolated_nodes} isolated node(s) detected in {input_file}')

    """Encode node features and labels in graph"""
    # Positional encoding for each node (used for Transformer-like GNNs)
    graph.ndata['f'] = min_max_normalize_tensor(graph.nodes()).reshape(-1, 1)  # [num_res_in_struct_df, 1]
    # One-hot features for each residue
    graph.ndata['f'] = torch.cat((graph.ndata['f'], node_feats), dim=1)  # [num_res_in_struct_df, num_node_feats]
    # Cartesian coordinates for each residue
    graph.ndata['x'] = node_coords  # [num_res_in_struct_df, 3]

    """Encode edge features and labels in graph"""
    # Positional encoding for each edge (used for sequentially-ordered inputs like proteins)
    graph.edata['f'] = torch.sin((graph.edges()[0] - graph.edges()[1]).float()).reshape(-1, 1)  # [num_edges, 1]
    # Normalized edge weights (according to Euclidean distance)
    edge_weights = min_max_normalize_tensor(torch.sum(node_coords[srcs] - node_coords[dsts] ** 2, 1)).reshape(-1, 1)
    graph.edata['f'] = torch.cat((graph.edata['f'], edge_weights), dim=1)  # [num_edges, 1]

    # Angle between the two amide normal vectors for a pair of residues, for all edge-connected residue pairs
    plane1 = struct_df[['amide_norm_vec']].iloc[dsts]
    plane2 = struct_df[['amide_norm_vec']].iloc[srcs]
    plane1.columns = ['amide_norm_vec']
    plane2.columns = ['amide_norm_vec']
    plane1 = torch.from_numpy(np.stack(plane1['amide_norm_vec'].values).astype('float32'))
    plane2 = torch.from_numpy(np.stack(plane2['amide_norm_vec'].values).astype('float32'))
    angles = np.array([
        torch.acos(torch.dot(vec1, vec2) / (torch.linalg.norm(vec1) * torch.linalg.norm(vec2)))
        for vec1, vec2 in zip(plane1, plane2)
    ])
    # Ensure amide plane normal vector angles on each edge are zeroed out rather than being left as NaN (in some cases)
    np.nan_to_num(angles, copy=False, nan=0.0, posinf=None, neginf=None)
    amide_angles = torch.from_numpy(np.nan_to_num(
        min_max_normalize_tensor(torch.from_numpy(angles)).cpu().numpy(),
        copy=True, nan=0.0, posinf=None, neginf=None
    )).reshape(-1, 1)  # [num_edges, 1]
    graph.edata['f'] = torch.cat((graph.edata['f'], amide_angles), dim=1)  # Amide-amide angles: [num_edges, 1]

    return graph


def build_complex_labels(bound_complex: any, df0: pd.DataFrame, df1: pd.DataFrame,
                         df0_index_to_node_id: dict, df1_index_to_node_id: dict, shuffle: bool):
    """ Construct the labels matrix for a given protein complex and mode (e.g. train, val, or test)."""
    # Get Cartesian product of CA atom row indices for both structures, making an array copy for future view calls
    index_pairs = np.transpose([
        np.tile(df0.index.values, len(df1.index.values)), np.repeat(df1.index.values, len(df0.index.values))
    ]).copy()

    # Get an array copy of pos_idx and neg_idx for (row) view calls
    pos_idx = bound_complex.pos_idx.copy()

    # Derive inter-protein node-node (i.e. residue-residue) interaction array (Interacting = 1, 0 otherwise)
    pos_labels = np.hstack((pos_idx, np.ones((len(pos_idx), 1), dtype=np.int64)))
    labels = pos_labels

    # Find residue-residue pairs not already included in pos_idx
    index_pair_rows = index_pairs.view([('', index_pairs.dtype)] * index_pairs.shape[1])
    pos_idx_rows = pos_idx.view([('', pos_idx.dtype)] * pos_idx.shape[1])
    unique_index_pairs = np.setdiff1d(index_pair_rows, pos_idx_rows) \
        .view(index_pairs.dtype).reshape(-1, index_pairs.shape[1]).copy()

    new_labels = np.hstack(
        (unique_index_pairs, np.zeros((len(unique_index_pairs), 1), dtype=np.int64)))

    # Derive inter-protein node-node (i.e. residue-residue) interaction matrix (Interacting = 1, 0 otherwise)
    labels = np.concatenate((labels, new_labels))

    # Shuffle rows corresponding to residue-residue pairs
    if shuffle:
        np.random.shuffle(labels)

    # Map DataFrame indices to graph node IDs for each structure
    labels[:, 0] = np.vectorize(df0_index_to_node_id.get)(labels[:, 0])
    labels[:, 1] = np.vectorize(df1_index_to_node_id.get)(labels[:, 1])

    # Return new labels matrix
    return torch.from_numpy(labels)


def process_complex_into_dict(raw_filepath: str, processed_filepath: str,
                              knn: int, self_loops: bool, check_sequence: bool):
    """Process protein complex into a dictionary representing both structures and ready for a given mode (e.g. val)."""
    # Retrieve specified complex
    bound_complex = pd.read_pickle(raw_filepath)

    # Isolate CA atoms in each structure's DataFrame
    df0 = bound_complex.df0[bound_complex.df0['atom_name'] == 'CA']
    df1 = bound_complex.df1[bound_complex.df1['atom_name'] == 'CA']

    # Ensure that the sequence of each DataFrame's residues matches its original FASTA sequence, character-by-character
    if check_sequence:
        df0_sequence = bound_complex.sequences['l_b']
        for i, (df_res_name, orig_res) in enumerate(zip(df0['resname'].values, df0_sequence)):
            if D3TO1[df_res_name] != orig_res:
                raise Exception(f'DataFrame 0 residue sequence does not match original FASTA sequence at position {i}')
        df1_sequence = bound_complex.sequences['r_b']
        for i, (df_res_name, orig_res) in enumerate(zip(df1['resname'].values, df1_sequence)):
            if D3TO1[df_res_name] != orig_res:
                raise Exception(f'DataFrame 1 residue sequence does not match original FASTA sequence at position {i}')

    # Convert each DataFrame into its DGLGraph representation, using all atoms to generate geometric features
    all_atom_df0, all_atom_df1 = bound_complex.df0, bound_complex.df1
    graph1 = convert_df_to_dgl_graph(all_atom_df0, raw_filepath, knn, self_loops)
    graph2 = convert_df_to_dgl_graph(all_atom_df1, raw_filepath, knn, self_loops)

    # Assemble the examples (containing labels) for the complex
    df0_index_to_node_id = {df0_index: idx for idx, df0_index in enumerate(df0.index.values)}
    df1_index_to_node_id = {df1_index: idx for idx, df1_index in enumerate(df1.index.values)}
    examples = build_complex_labels(bound_complex, df0, df1, df0_index_to_node_id, df1_index_to_node_id, shuffle=False)

    # Use pure PyTorch tensors to represent a given complex - Assemble tensors for storage in complex's dictionary
    graph1_node_feats = graph1.ndata['f']  # (n_nodes, n_node_feats)
    graph2_node_feats = graph2.ndata['f']

    graph1_node_coords = graph1.ndata['x']  # (n_nodes, 3)
    graph2_node_coords = graph2.ndata['x']

    # Collect the neighboring node and in-edge features for each of the first graph's nodes (in a consistent order)
    graph1_edge_feats = []
    graph1_nbrhd_indices = []
    for h_i in graph1.nodes():
        in_edge_ids_for_h_i = graph1.in_edges(h_i)
        in_edges_for_h_i = graph1.edges[in_edge_ids_for_h_i]
        graph1_edge_feats.append(in_edges_for_h_i.data['f'])
        dst_node_ids_for_h_i = in_edge_ids_for_h_i[0].reshape(-1, 1)
        graph1_nbrhd_indices.append(dst_node_ids_for_h_i)
    graph1_edge_feats = torch.stack(graph1_edge_feats)  # (n_nodes, nbrhd_size, n_edge_feats)
    graph1_nbrhd_indices = torch.stack(graph1_nbrhd_indices)  # (n_nodes, nbrhd_size, 1)

    # Collect the neighboring node and in-edge features for each of the second graph's nodes (in a consistent order)
    graph2_edge_feats = []
    graph2_nbrhd_indices = []
    for h_i in graph2.nodes():
        in_edge_ids_for_h_i = graph2.in_edges(h_i)
        in_edges_for_h_i = graph2.edges[in_edge_ids_for_h_i]
        graph2_edge_feats.append(in_edges_for_h_i.data['f'])
        dst_node_ids_for_h_i = in_edge_ids_for_h_i[0].reshape(-1, 1)
        graph2_nbrhd_indices.append(dst_node_ids_for_h_i)
    graph2_edge_feats = torch.stack(graph2_edge_feats)
    graph2_nbrhd_indices = torch.stack(graph2_nbrhd_indices)

    # Initialize the complex's new representation as a dictionary
    processed_complex = {
        'graph1_node_feats': torch.nan_to_num(graph1_node_feats),
        'graph2_node_feats': torch.nan_to_num(graph2_node_feats),
        'graph1_node_coords': torch.nan_to_num(graph1_node_coords),
        'graph2_node_coords': torch.nan_to_num(graph2_node_coords),
        'graph1_edge_feats': torch.nan_to_num(graph1_edge_feats),
        'graph2_edge_feats': torch.nan_to_num(graph2_edge_feats),
        'graph1_nbrhd_indices': graph1_nbrhd_indices,
        'graph2_nbrhd_indices': graph2_nbrhd_indices,
        'examples': examples,
        'complex': bound_complex.complex
    }

    # Write into processed_filepath
    processed_file_dir = os.path.join(*processed_filepath.split(os.sep)[: -1])
    os.makedirs(processed_file_dir, exist_ok=True)
    with open(processed_filepath, 'wb') as f:
        pickle.dump(processed_complex, f)


def zero_out_complex_features(cmplx: dict):
    """Zero-out the input features for a given protein complex dictionary (for an input-independent baseline)."""
    cmplx['graph1_node_feats'] = torch.zeros_like(cmplx['graph1_node_feats'])
    cmplx['graph2_node_feats'] = torch.zeros_like(cmplx['graph2_node_feats'])
    cmplx['graph1_edge_feats'] = torch.zeros_like(cmplx['graph1_edge_feats'])
    cmplx['graph2_edge_feats'] = torch.zeros_like(cmplx['graph2_edge_feats'])
    return cmplx


def construct_interact_tensor(graph1_feats: torch.Tensor, graph2_feats: torch.Tensor, pad=False, max_len=256):
    """Build the interaction tensor for given node representations, optionally padding up to the node count limit."""
    # Get descriptors and reshaped versions of the input feature matrices
    len_1, len_2 = graph1_feats.shape[0], graph2_feats.shape[0]
    x_a, x_b = graph1_feats.permute(1, 0).unsqueeze(0), graph2_feats.permute(1, 0).unsqueeze(0)
    if pad:
        x_a_num_zeros = max_len - x_a.shape[2]
        x_b_num_zeros = max_len - x_b.shape[2]
        x_a = F.pad(x_a, (0, x_a_num_zeros, 0, 0, 0, 0))  # Pad the start of 3D tensors
        x_b = F.pad(x_b, (0, x_b_num_zeros, 0, 0, 0, 0))  # Pad the end of 3D tensors
        len_1, len_2 = max_len, max_len
    # Interleave 2D input matrices into a 3D interaction tensor
    interact_tensor = torch.cat((torch.repeat_interleave(x_a.unsqueeze(3), repeats=len_2, dim=3),
                                 torch.repeat_interleave(x_b.unsqueeze(2), repeats=len_1, dim=2)), dim=1)
    return interact_tensor


def collect_args():
    """Collect all arguments required for training/testing."""
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # -----------------
    # Model arguments
    # -----------------
    parser.add_argument('--model_name', type=str, default='NeiA', help='Options are NeiA or NeiWA')
    parser.add_argument('--num_gnn_layers', type=int, default=1, help='Number of GNN layers')
    parser.add_argument('--num_interact_layers', type=int, default=3, help='Number of layers in interaction module')
    parser.add_argument('--metric_to_track', type=str, default='val_bce', help='Scheduling and early stop')

    # -----------------
    # Data arguments
    # -----------------
    parser.add_argument('--knn', type=int, default=20, help='Number of nearest neighbor edges for each node')
    parser.add_argument('--self_loops', action='store_true', dest='self_loops', help='Allow node self-loops')
    parser.add_argument('--no_self_loops', action='store_false', dest='self_loops', help='Disable self-loops')
    parser.add_argument('--pn_ratio', type=float, default=0.1,
                        help='Positive-negative class ratio to instate during training with DIPS-Plus')
    parser.add_argument('--dips_data_dir', type=str, default='datasets/DIPS/final/raw', help='Path to DIPS-Plus')
    parser.add_argument('--dips_percent_to_use', type=float, default=1.00,
                        help='Fraction of DIPS-Plus dataset splits to use')
    parser.add_argument('--db5_data_dir', type=str, default='datasets/DB5/final/raw', help='Path to DB5-Plus')
    parser.add_argument('--db5_percent_to_use', type=float, default=1.00,
                        help='Fraction of DB5-Plus dataset splits to use')
    parser.add_argument('--process_complexes', action='store_true', dest='process_complexes',
                        help='Check if all complexes for a dataset are processed and, if not, process those remaining')

    # -----------------
    # Logging arguments
    # -----------------
    parser.add_argument('--logger_name', type=str, default='TensorBoard', help='Which logger to use for experiments')
    parser.add_argument('--experiment_name', type=str, default=None, help='Logger experiment name')
    parser.add_argument('--project_name', type=str, default='NeiA-PyTorch', help='Logger project name')
    parser.add_argument('--entity', type=str, default='PyTorch', help='Logger entity (i.e. team) name')
    parser.add_argument('--run_id', type=str, default='', help='Logger run ID')
    parser.add_argument('--offline', action='store_true', dest='offline', help='Whether to log locally or remotely')
    parser.add_argument('--online', action='store_false', dest='offline', help='Whether to log locally or remotely')
    parser.add_argument('--tb_log_dir', type=str, default='tb_logs', help='Where to store TensorBoard log files')
    parser.set_defaults(offline=False)  # Default to using online logging mode

    # -----------------
    # Seed arguments
    # -----------------
    parser.add_argument('--seed', type=int, default=None, help='Seed for NumPy and PyTorch')

    # -----------------
    # Meta-arguments
    # -----------------
    parser.add_argument('--batch_size', type=int, default=1, help='Number of samples included in each data batch')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Decay rate of optimizer weight')
    parser.add_argument('--num_epochs', type=int, default=50, help='Maximum number of epochs to run for training')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout (forget) rate')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait until early stopping')
    parser.add_argument('--pad', action='store_true', dest='pad', help='Whether to zero pad interaction tensors')

    # -----------------
    # Miscellaneous
    # -----------------
    parser.add_argument('--max_hours', type=int, default=1, help='Maximum number of hours to allot for training')
    parser.add_argument('--max_minutes', type=int, default=55, help='Maximum number of minutes to allot for training')
    parser.add_argument('--multi_gpu_backend', type=str, default='ddp', help='Multi-GPU backend for training')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (e.g. -1 = all available GPUs)')
    parser.add_argument('--auto_choose_gpus', action='store_true', dest='auto_choose_gpus', help='Auto-select GPUs')
    parser.add_argument('--num_compute_nodes', type=int, default=1, help='Number of compute nodes to use')
    parser.add_argument('--gpu_precision', type=int, default=32, help='Bit size used during training (e.g. 16-bit)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU threads for loading data')
    parser.add_argument('--profiler_method', type=str, default=None, help='PL profiler to use (e.g. simple)')
    parser.add_argument('--ckpt_dir', type=str, default=f'{os.path.join(os.getcwd(), "checkpoints")}',
                        help='Directory in which to save checkpoints')
    parser.add_argument('--ckpt_name', type=str, default='', help='Filename of best checkpoint')
    parser.add_argument('--min_delta', type=float, default=5e-6, help='Minimum percentage of change required to'
                                                                      ' "metric_to_track" before early stopping'
                                                                      ' after surpassing patience')
    parser.add_argument('--accum_grad_batches', type=int, default=1, help='Norm over which to clip gradients')
    parser.add_argument('--grad_clip_val', type=float, default=0.5, help='Norm over which to clip gradients')
    parser.add_argument('--grad_clip_algo', type=str, default='norm', help='Algorithm with which to clip gradients')
    parser.add_argument('--stc_weight_avg', action='store_true', dest='stc_weight_avg', help='Smooth loss landscape')
    parser.add_argument('--find_lr', action='store_true', dest='find_lr', help='Find an optimal learning rate a priori')
    parser.add_argument('--input_indep', action='store_true', dest='input_indep', help='Whether to zero input for test')

    return parser


def process_args(args):
    """Process all arguments required for training/testing."""
    # ---------------------------------------
    # Seed fixing for random numbers
    # ---------------------------------------
    if not args.seed:
        args.seed = 42  # np.random.randint(100000)
    logging.info(f'Seeding everything with random seed {args.seed}')
    pl.seed_everything(args.seed)

    return args


def construct_pl_logger(args):
    """Return a specific Logger instance requested by the user."""
    if args.logger_name.lower() == 'wandb':
        return construct_wandb_pl_logger(args)
    else:  # Default to using TensorBoard
        return construct_tensorboard_pl_logger(args)


def construct_wandb_pl_logger(args):
    """Return an instance of WandbLogger with corresponding project and name strings."""
    return WandbLogger(name=args.experiment_name,
                       offline=args.offline,
                       project=args.project_name,
                       log_model=True,
                       entity=args.entity)


def construct_tensorboard_pl_logger(args):
    """Return an instance of TensorBoardLogger with corresponding project and experiment name strings."""
    return TensorBoardLogger(save_dir=args.tb_log_dir,
                             name=args.experiment_name)
