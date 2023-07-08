import collections as col
import copy
import dill
import gzip
import h5py
import logging
import os
import re
import requests
import shutil
import subprocess
import tempfile
import urllib.request as request
from contextlib import closing
from pathlib import Path
from typing import List

import atom3.database as db
import atom3.neighbors as nb
import atom3.pair as pa
import dgl
import dill as pickle
import hickle as hkl
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.PDB import Selection
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, DSSP
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.vectors import Vector
from Bio.SCOP.Raf import protein_letters_3to1
from atom3.complex import Complex
from atom3.utils import slice_list
from mpi4py import MPI
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler
from torch import FloatTensor
from tqdm import tqdm

from project.utils.constants import AMINO_ACIDS, AMINO_ACID_IDX, MAX_NODES_PER_JOB, \
    PSAIA_COLUMNS, PDB_PARSER, DEFAULT_DATASET_STATISTICS, ATOM_COUNT_LIMIT, RCSB_BASE_URL, FEAT_COLS, \
    ALLOWABLE_FEATS, DEFAULT_MISSING_SS, DEFAULT_MISSING_PROTRUSION_INDEX, DEFAULT_MISSING_RSA, \
    DEFAULT_MISSING_RD, DEFAULT_MISSING_HSAAC, DEFAULT_MISSING_CN, DEFAULT_MISSING_SEQUENCE_FEATS, \
    DEFAULT_MISSING_NORM_VEC, HSAAC_DIM, NUM_ALLOWABLE_NANS

try:
    from types import SliceType
except ImportError:
    SliceType = slice


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from DIPS (https://github.com/drorlab/DIPS):
# -------------------------------------------------------------------------------------------------------------------------------------

def process_pairs_to_keep(pair_filename: str, output_filename: str, to_keep_df):
    """Check if pair_filename should be to_keep. If yes, write it into output_filename. Otherwise, delete it if it is already in output_filename. If to_keep_df is not specified (i.e. it is empty), copy all pairs to the output dir."""
    exist = os.path.exists(output_filename)
    if to_keep_df.empty or __should_keep(pair_filename, to_keep_df):
        if not exist:
            # Write into output_filename if not exist
            shutil.copy(pair_filename, output_filename)
        return 1  # pair file was copied
    else:
        if exist:
            # Delete the output_filename
            os.remove(output_filename)
        return 0  # pair file wasn't copied


def __load_to_keep_files_into_dataframe(to_keep_filenames: List[str]):
    """ Load all file and intersect them into one pandas dataframe. The file
    heading indicates the criterias by which the pairs should be to_keep.
    They could be based on:
      pair_name (e.g. 2dj5.pdb1_0) (buried_surface_over_500.txt)
      pdb_name (e.g. 4rjv.pdb1) (seq_id_less_30.txt)
      pdb_code (e.g. 100d) (nmr_res_less_3.5.txt)
      (pdb_code chain) (101M A) (size_over_50.0.txt)

    The dataframe would have the following headers: ['pdb_code', 'struct_id'
    'pair_id', 'chain']. For example, 2dj5.pdb1_0 will have ['2dj5', 1, 0, None]
    as entry.
    """
    if len(to_keep_filenames) == 0:
        return pd.DataFrame()

    regex = re.compile(
        '(?P<pdb_code>\w{4})(\.pdb(?P<struct_id>\d+))*(_(?P<pair_id>\d+))*( (?P<chain>\w+))*')

    dfs = col.defaultdict(list)
    for filename in to_keep_filenames:
        data = []
        with open(filename, 'r') as f:
            logging.info("Processing to_keep file: {:}".format(filename))
            header = f.readline().rstrip()
            data += [regex.match(os.path.basename(line.rstrip()).lower()).groupdict() \
                     for line in f]
        df = pd.DataFrame(data, columns=['pdb_code', 'struct_id', 'pair_id', 'chain'])
        # Drop columns will all null
        df = df.dropna(axis=1, how='all')
        dfs[header].append(df)
    assert (len(dfs) > 0)

    # Combine dataframes with the same header
    dataframes = []
    for key in dfs:
        df = pd.concat(dfs[key])
        # Sort and remove duplicates
        df = df.sort_values(by=list(df.columns)).drop_duplicates()
        dataframes.append(df)

    # Merge the dataframes
    to_keep_df = dataframes[0]
    for df in dataframes[1:]:
        join_on = list(set(to_keep_df.columns) & set(df.columns))
        to_keep_df = to_keep_df.merge(df, left_on=join_on, right_on=join_on)
    return to_keep_df


def __should_keep(pair_filename: str, to_keep_df: pd.DataFrame):
    """Check if a given pair should be retained or discarded based on the provided criteria (i.e. filter '.txt' files)."""
    assert (not to_keep_df.empty)
    # pair_name example: 20gs.pdb1_0
    pair_name_regex = re.compile('(?P<pdb_code>\w{4})(\.pdb(?P<struct_id>\d*))*(_(?P<pair_id>\d+))')

    pair_name = db.get_pdb_name(pair_filename)
    pair_metadata = pair_name_regex.match(pair_name).groupdict()

    # Assign a struct_id of '1' if missing from original pair_name
    if not pair_metadata['struct_id']:
        pair_metadata['struct_id'] = '1'

    # The order to check is: pdb_code, struct_id, pair_id, chain
    if pair_metadata['pdb_code'] not in set(to_keep_df.pdb_code):
        return False
    # Check if we need to select based on struct_id
    slice = to_keep_df[to_keep_df.pdb_code == pair_metadata['pdb_code']]
    if 'struct_id' in slice.columns:
        if pair_metadata['struct_id'] not in set(slice.struct_id):
            return False
        slice = slice[slice.struct_id == pair_metadata['struct_id']]
    # Check if we need to select based on pair_id
    if 'pair_id' in slice.columns:
        if pair_metadata['pair_id'] not in set(slice.pair_id):
            return False
        slice = slice[slice.pair_id == pair_metadata['pair_id']]
    # Check if we need to select based on chain
    if 'chain' in slice.columns:
        pair = pa.read_pair_from_dill(pair_filename)
        pair_chains = set(pair.df0.chain) | set(pair.df1.chain)
        # Convert chain names to lowercase
        pair_chains = set([c.lower() for c in pair_chains])
        # All chains in the pair need to be to_keep_df to be valid
        if not pair_chains.issubset(set(slice.chain)):
            return False
    return True


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Atom3D (https://github.com/drorlab/atom3d/blob/master/benchmarking/pytorch_geometric/ppi_dataloader.py):
# -------------------------------------------------------------------------------------------------------------------------------------
def prot_df_to_dgl_graph_feats(df: pd.DataFrame, feat_cols: List, allowable_feats: List[List],
                               edge_dist_cutoff: float, edge_limit: int):
    r"""Convert protein in dataframe representation to a graph compatible with DGL, where each node is a residue.

    :param df: Protein structure in dataframe format.
    :type df: pandas.DataFrame
    :param feat_cols: Columns of dataframe in which to find node feature values. For example, for residues use ``feat_cols=["element", ...]`` and for residues use ``feat_cols=["resname", ...], or both!``
    :type feat_cols: list[list[Any]]
    :param allowable_feats: List of lists containing all possible values of node type, to be converted into 1-hot node features.
        Any elements in ``feat_col`` that are not found in ``allowable_feats`` will be added to an appended "unknown" bin (see :func:`atom3d.util.graph.one_of_k_encoding_unk`).
    :param edge_dist_cutoff: Maximum distance cutoff (in Angstroms) to define an edge between two residues, defaults to 15.0.
    :type edge_dist_cutoff: float
    :param edge_limit: Maximum number of edges allowed in a given graph, defaults to 15000.
    :type edge_limit: float

    :return: tuple containing
        - node_coords (torch.FloatTensor): Cartesian coordinates of each node.

        - node_feats (torch.FloatTensor): Features for each node, one-hot encoded by values in ``allowable_feats``.

        - srcs (torch.FloatTensor): Source edge indices

        - dsts (torch.FloatTensor): Destination edge indices
    :rtype: Tuple
    """
    # Exit early if feat_cols or allowable_feats do not align in dimensionality
    if len(feat_cols) != len(allowable_feats):
        logging.error('feat_cols does not match the length of allowable_feats')
        exit(1)

    # Structure-based node feature aggregation
    node_feats = FloatTensor([])
    for i in range(len(feat_cols)):
        # Search through embedded 2D list for allowable values
        feat_vecs = [one_of_k_encoding_unk(feat, allowable_feats[i], feat_cols[i]) for feat in df[feat_cols[i]]]
        one_hot_feat_vecs = FloatTensor(feat_vecs)
        node_feats = torch.cat((node_feats, one_hot_feat_vecs), 1)

    # Organize residue coordinates into a FloatTensor
    node_coords = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float32)

    # Edge aggregation - distance threshold determines whether a residue-residue edge gets created
    srcs, dsts = get_edges(node_coords, edge_dist_cutoff, edge_limit)

    return FloatTensor(node_coords), node_feats, srcs, dsts


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
# Following code derived from PAIRpred (https://combi.cs.colostate.edu/supplements/pairpred/):
# -------------------------------------------------------------------------------------------------------------------------------------
def get_coords(residues):
    """
    Get atom coordinates given a list of biopython residues
    """
    Coords = []
    for (idx, r) in enumerate(residues):
        v = [ak.get_coord() for ak in r.get_list()]
        Coords.append(v)
    return Coords


def get_res_letter(residue):
    """
    Get the letter code for a biopython residue object
    """
    r2name = residue.get_resname()
    if r2name in protein_letters_3to1:
        scode = protein_letters_3to1[r2name]
    else:
        scode = '-'
    return scode


def get_side_chain_vector(residue):
    """
    Find the average of the unit vectors to different atoms in the side chain
    from the c-alpha atom. For glycine the average of the N-Ca and C-Ca is
    used.
    Returns (C-alpha coordinate vector, side chain unit vector) for residue r
    """
    u = None
    gly = 0
    if is_aa(residue) and residue.has_id('CA'):
        ca = residue['CA'].get_coord()
        dv = np.array([ak.get_coord() for ak in residue.get_unpacked_list()[4:]])
        if len(dv) < 1:
            if residue.has_id('N') and residue.has_id('C'):
                dv = [residue['C'].get_coord(), residue['N'].get_coord()]
                dv = np.array(dv)
                gly = 1
            else:
                return None
        dv = dv - ca
        if gly:
            dv = -dv
        n = np.sum(np.abs(dv) ** 2, axis=-1) ** (1. / 2)
        v = dv / n[:, np.newaxis]
        v = v.mean(axis=0)
        u = (Vector(ca), Vector(v))
    return u


def get_similarity_matrix(coords, sg=2.0, thr=1e-3):
    """
    Instantiates the distance based similarity matrix (S). S is a tuple of
    lists (I,V). |I|=|V|=|R|. Each I[r] refers to the indices
    of residues in R which are "close" to the residue indexed by r in R, and V[r]
    contains a list of the similarity scores for the corresponding residues.
    The distance between two residues is defined to be the minimum distance of
    any of their atoms. The similarity score is evaluated as
        s = exp(-d^2/(2*sg^2))
    This ensures that the range of similarity values is 0-1. sg (sigma)
    determines the extent of the neighborhood.
    Two residues are defined to be close to one another if their similarity
    score is greater than a threshold (thr).
    Residues (or ligands) for which DSSP features are not available are not
    included in the distance calculations.
    """
    sg = 2 * (sg ** 2)
    I = [[] for k in range(len(coords))]
    V = [[] for k in range(len(coords))]
    for i in range(len(coords)):
        for j in range(i, len(coords)):
            d = spatial.distance.cdist(coords[i], coords[j]).min()
            s = np.exp(-(d ** 2) / sg)
            if s > thr:  # and not np.isnan(self.Phi[i]) and not np.isnan(self.Phi[j])
                I[i].append(j)
                V[i].append(s)
                if i != j:
                    I[j].append(i)
                    V[j].append(s)
    similarity_matrix = (I, V)
    coordinate_numbers = np.array([len(a) for a in similarity_matrix[0]])
    return similarity_matrix, coordinate_numbers


def get_hsacc(residues, similarity_matrix, raw_pdb_filename):
    """
    Compute the Half sphere exposure statistics
    The up direction is defined as the direction of the side chain and is
    calculated by taking average of the unit vectors to different side chain
    atoms from the C-alpha atom
    Anything within the up half sphere is counted as up and the rest as
    down
    """
    N = len(residues)
    Na = len(AMINO_ACIDS)
    UN = np.zeros(N)
    DN = np.zeros(N)
    UC = np.zeros((Na, N))
    DC = np.zeros((Na, N))
    for (i, r) in enumerate(residues):
        u = get_side_chain_vector(r)
        if u is None:
            UN[i] = np.nan
            DN[i] = np.nan
            UC[:, i] = np.nan
            DC[:, i] = np.nan
            logging.info(f'No side chain vector found for residue #{i} in PDB file {raw_pdb_filename}')
        else:
            idx = AMINO_ACID_IDX[get_res_letter(r)]
            UC[idx, i] = UC[idx, i] + 1
            DC[idx, i] = DC[idx, i] + 1
            n = similarity_matrix[0][i]
            for j in n:
                r2 = residues[j]
                if is_aa(r2) and r2.has_id('CA'):
                    v2 = r2['CA'].get_vector()
                    scode = get_res_letter(r2)
                    idx = AMINO_ACID_IDX[scode]
                    angle = u[1].angle((v2 - u[0]))
                    if angle < np.pi / 2.0:
                        UN[i] = UN[i] + 1
                        UC[idx, i] = UC[idx, i] + 1
                    else:
                        DN[i] = DN[i] + 1
                        DC[idx, i] = DC[idx, i] + 1
    UC = UC / (1.0 + UN)
    DC = DC / (1.0 + DN)
    return UC, DC


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DIPS-Plus (https://github.com/amorehead/DIPS-Plus):
# -------------------------------------------------------------------------------------------------------------------------------------
def get_global_node_rank(rank, size):
    """Recover a node's global MPI rank from its local MPI rank and the global MPI size (for a batch of MPI jobs)."""
    # Declare global node rank which is supplied as an argument
    global_node_rank = rank

    # Gather MPI-related information for a single node
    local_node_rank = MPI.COMM_WORLD.Get_rank()

    # Ascertain the true rank of a given job for the current node - Each Slurm job has access to at most 4 nodes
    node_id_batches = slice_list(list(range(size)), MAX_NODES_PER_JOB)
    node_id = node_id_batches[global_node_rank][local_node_rank]

    # Reestablish rank as global node_id
    rank = node_id
    return rank


def find_fasta_sequences_for_pdb_file(sequences: dict, pdb_filename: str, external_feats_dir: str,
                                      struct_idx: int, is_rcsb_complex: bool, original_pair: pd.DataFrame):
    """Extract from previously-generated FASTA files the residue sequences for a given PDB file."""
    # Extract required paths, file lists, and sequences
    pdb_code = db.get_pdb_code(pdb_filename)[1:3]
    pdb_full_name = db.get_pdb_name(pdb_filename)
    external_feats_subdir = os.path.join(external_feats_dir, pdb_code, 'work')
    if is_rcsb_complex:  # Construct RCSB FASTA file names indirectly out of necessity
        df = original_pair.df0 if struct_idx == 0 else original_pair.df1
        raw_rcsb_filename = os.path.split(pdb_filename)
        rcsb_fasta_filename = os.path.join(external_feats_subdir,
                                           raw_rcsb_filename[1] +
                                           f'-{df.iloc[0]["model"]}' + f'-{df.iloc[0]["chain"]}.fa')
        fasta_files = [rcsb_fasta_filename]
    else:  # Construct DB5 FASTA file names more directly using raw PDB filenames
        fasta_files = [os.path.join(external_feats_subdir, file) for file in os.listdir(external_feats_subdir)
                       if pdb_full_name in file and '.fa' in file]

    # Get only the first sequence from each FASTA sequence object
    sequence_list = [sequence.seq._data for fasta_file in fasta_files for sequence in SeqIO.parse(fasta_file, 'fasta')]

    # Give each sequence a left or right-bound key
    for fasta_file, sequence in zip(fasta_files, sequence_list):
        if struct_idx == 0:
            sequences['l_b'] = sequence
        else:
            sequences['r_b'] = sequence
    return sequences


def min_max_normalize_feature_array(features):
    """Independently for each column, normalize feature array values to be in range [0, 1]."""
    scaler = MinMaxScaler()
    scaler.fit(features)
    features_scaled = scaler.transform(features)
    return features_scaled


def min_max_normalize_feature_tensor(features):
    """Normalize provided feature tensor to have its values be in range [0, 1]."""
    min_value = min(features)
    max_value = max(features)
    features_std = torch.tensor([(value - min_value) / (max_value - min_value) for value in features])
    features_scaled = features_std * (max_value - min_value) + min_value
    return features_scaled


def get_dssp_dict_for_pdb_file(pdb_filename):
    """Run DSSP to calculate secondary structure features for a given PDB file."""
    dssp_dict = {}  # Initialize to default DSSP dict value
    try:
        dssp_tuple = dssp_dict_from_pdb_file(pdb_filename)
        dssp_dict = dssp_tuple[0]
    except Exception:
        logging.info("No DSSP features found for {:}".format(pdb_filename))
    return dssp_dict


def get_dssp_dict_for_pdb_model(pdb_model, raw_pdb_filename):
    """Run DSSP to calculate secondary structure features for a given PDB file."""
    dssp_dict = {}  # Initialize to default DSSP dict value
    try:
        dssp_dict = DSSP(pdb_model, raw_pdb_filename)
    except Exception:
        logging.info("No DSSP features found for {:}".format(pdb_model))
    return dssp_dict


def get_msms_rd_dict_for_pdb_model(pdb_model):
    """Run MSMS to calculate residue depth model for a given PDB model."""
    rd_dict = {}  # Initialize to default RD dict value
    try:
        rd_dict = ResidueDepth(pdb_model)
    except Exception:
        logging.info("No MSMS residue depth model found for {:}".format(pdb_model))
    return rd_dict


# Following function adapted from PAIRPred (https://combi.cs.colostate.edu/supplements/pairpred/)
def get_df_from_psaia_tbl_file(psaia_filename):
    """Parse through a given PSAIA .tbl output file to construct a new Pandas DataFrame."""
    psaia_dict = {}  # Initialize to default PSAIA DataFrame
    psaia_df = pd.DataFrame(columns=['average CX', 's_avg CX', 's-ch avg CX', 's-ch s_avg CX', 'max CX', 'min CX'])
    # Attempt to parse all the lines of a single PSAIA .tbl file for residue-level protrusion values
    try:
        stnxt = 0
        ln = 0
        for l in open(psaia_filename, "r"):
            ln = ln + 1
            ls = l.split()
            if stnxt:
                cid = ls[0]
                if cid == '*':  # PSAIA replaces cid ' ' in pdb files with *
                    cid = ' '
                resid = (cid, ls[1])  # cid, resid, resname is ignored
                rcx = tuple(map(float, ls[3:9]))
                psaia_dict[resid] = rcx
            elif len(ls) and ls[0] == 'chain':  # the line containing 'chain' is the last line before real data starts
                stnxt = 1
        # Construct a new DataFrame from the parsed dictionary
        psaia_df = pd.DataFrame.from_dict(psaia_dict).T
        psaia_df.columns = PSAIA_COLUMNS
    except Exception:
        logging.info("Error in parsing PSAIA .tbl file {:}".format(psaia_filename))
    return psaia_df


def get_hsaac_for_pdb_residues(residues, similarity_matrix, raw_pdb_filename):
    """Run BioPython to calculate half-sphere amino acid composition (HSAAC) for a given list of PDB residues."""
    hsaacs = np.array([DEFAULT_MISSING_HSAAC for _ in range(len(residues))])  # Initialize to default HSAACs value
    try:
        UC, DC = get_hsacc(residues, similarity_matrix, raw_pdb_filename)
        hsaacs = np.concatenate((UC, DC))  # Concatenate to get HSAAC
    except Exception:
        logging.info("No half-sphere amino acid compositions (HSAACs) found for PDB file {:}".format(raw_pdb_filename))
    return hsaacs


def get_dssp_value_for_residue(dssp_dict: dict, feature: str, chain: str, residue: int):
    """Return a secondary structure (SS) value or a relative solvent accessibility (RSA) value for a given chain-residue pair."""
    dssp_value = DEFAULT_MISSING_SS if feature == 'SS' else DEFAULT_MISSING_RSA  # Initialize to default DSSP feature
    try:
        if feature == 'SS':
            dssp_values = dssp_dict[chain, (' ', residue, ' ')]
            dssp_value = dssp_values[2]
        else:  # feature == 'RSA'
            dssp_values = dssp_dict[chain, (' ', residue, ' ')]
            dssp_value = dssp_values[3]
    except Exception:
        logging.info("No DSSP entry found for {:}".format((chain, (' ', residue, ' '))))
    return dssp_value


def get_msms_rd_value_for_residue(rd_dict: dict, chain: str, residue: int):
    """Return an alpha-carbon residue depth (RD) value for a given chain-residue pair."""
    ca_depth_value = DEFAULT_MISSING_RD  # Initialize to default RD value
    try:
        rd_value, ca_depth_value = rd_dict[chain, (' ', residue, ' ')]
    except Exception:
        logging.info("No MSMS residue depth entry found for {:}".format((chain, (' ', residue, ' '))))
    return ca_depth_value[0] if type(ca_depth_value) == list else ca_depth_value


def get_protrusion_index_for_residue(psaia_df: pd.DataFrame, chain: str, residue: int):
    """Return a protrusion index for a given chain-residue pair."""
    protrusion_index = DEFAULT_MISSING_PROTRUSION_INDEX  # Initialize to default protrusion index
    try:
        protrusion_index = psaia_df.loc[(chain, str(residue))].to_list()
    except Exception:
        logging.info("No protrusion index entry found for {:}".format((chain, (' ', residue, ' '))))
    return protrusion_index


def get_hsaac_for_residue(hsaac_matrix: np.array, residue_counter: int, chain: str, residue_id: int):
    """Return a half-sphere amino acid composition (HSAAC) for a given chain-residue pair."""
    hsaac = np.array(DEFAULT_MISSING_HSAAC)  # Initialize to default HSAAC value
    try:
        hsaac = hsaac_matrix[:, residue_counter]
    except Exception:
        logging.info(
            "No half-sphere amino acid composition entry found for {:}".format((chain, (' ', residue_id, ' '))))
    return np.array(DEFAULT_MISSING_HSAAC) if len(hsaac) > HSAAC_DIM else hsaac  # Handle HSAAC parsing edge case


def get_cn_value_for_residue(cn_values: np.array, residue_counter: int, chain: str, residue_id: int):
    """Return a coordinate number value for a given chain-residue pair."""
    cn_value = DEFAULT_MISSING_CN  # Initialize to default HSAAC value
    try:
        cn_value = cn_values[residue_counter]
    except Exception:
        logging.info("No coordinate number entry found for {:}".format((chain, (' ', residue_id, ' '))))
    return cn_value


def get_sequence_feats_for_residue(sequence_feats_df: pd.DataFrame, chain: str, residue_id: int):
    """Return all pre-generated sequence features (from profile HMM) for a given chain-residue pair."""
    sequence_feats = DEFAULT_MISSING_SEQUENCE_FEATS  # Initialize to default sequence features
    try:
        # Sequence features start at the 5th column and end at the third-to-last column
        sequence_feats = sequence_feats_df[sequence_feats_df['chain'].apply(
            lambda x: x.strip() == chain) & sequence_feats_df['residue'].apply(
            lambda x: x.strip() == str(residue_id))].to_numpy()[0, 4:-3]  # Grab first matching chain-residue pair
        # First twenty feature entries are emission probabilities for the residue, and the last seven are its transition probabilities
    except Exception:
        logging.info("No sequence feature entries found for {:}".format((chain, (' ', residue_id, ' '))))
    return sequence_feats


def get_norm_vec_for_residue(df: pd.DataFrame, ca_atom: pd.Series, chain: str, residue_id: int):
    """Return a normal vector for a given residue."""
    norm_vec = DEFAULT_MISSING_NORM_VEC  # Initialize to default norm vec value
    try:
        # Calculate normal vector for each residue's amide plane using relative coords of each Ca-Cb and Cb-N bond
        cb_atom = df[(df.chain == ca_atom.chain) &
                     (df.residue == ca_atom.residue) &
                     (df.atom_name == 'CB')]
        n_atom = df[(df.chain == ca_atom.chain) &
                    (df.residue == ca_atom.residue) &
                    (df.atom_name == 'N')]
        vec1 = ca_atom[['x', 'y', 'z']].to_numpy() - cb_atom[['x', 'y', 'z']].to_numpy()
        vec2 = cb_atom[['x', 'y', 'z']].to_numpy() - n_atom[['x', 'y', 'z']].to_numpy()
        norm_vec = np.cross(vec1, vec2)
    except Exception:
        logging.info("No normal vector entry found for {:}".format(chain, (' ', residue_id, ' ')))
    if len(norm_vec) == 0:  # Catch a missing normal vector, possibly from the residue missing a CB atom (e.g. Glycine)
        norm_vec = DEFAULT_MISSING_NORM_VEC
    return norm_vec


def postprocess_pruned_pairs(raw_pdb_dir: str, external_feats_dir: str, pair_filename: str,
                             output_filename: str, source_type: str):
    """Check if underlying PDB file for pair_filename contains DSSP-derivable features. If yes, postprocess its derived features and write them into three separate output_filenames. Otherwise, delete it if it is already in output_filename."""
    output_file_exists = os.path.exists(output_filename)
    pair, raw_pdb_filenames, should_keep = __should_keep_postprocessed(raw_pdb_dir, pair_filename, source_type)
    if should_keep:
        postprocessed_pair = postprocess_pruned_pair(raw_pdb_filenames, external_feats_dir, pair, source_type)
        if not output_file_exists:
            # Write into output_filenames if not exist
            with open(output_filename, 'wb') as f:
                pickle.dump(postprocessed_pair, f)
    else:
        if output_file_exists:
            # Delete the output_filenames
            os.remove(output_filename)


def postprocess_pruned_pair(raw_pdb_filenames: List[str], external_feats_dir: str, original_pair, source_type: str):
    """Construct a new Pair consisting of residues of structures with DSSP-derivable features and append DSSP secondary structure (SS) features to each protein structure dataframe as well."""
    chains_selected = [original_pair.df0['chain'][0], original_pair.df1['chain'][0]]
    df0_ss_values, df0_rsa_values, df0_rd_values, df0_protrusion_indices, \
    df0_hsaacs, df0_cn_values, df0_sequence_feats, df0_amide_norm_vecs, \
    df1_ss_values, df1_rsa_values, df1_rd_values, df1_protrusion_indices, \
    df1_hsaacs, df1_cn_values, df1_sequence_feats, df1_amide_norm_vecs, = [], [], [], [], [], [], [], [], \
                                                                          [], [], [], [], [], [], [], []
    single_raw_pdb_file_provided = len(list(set(raw_pdb_filenames))) == 1

    # Collect sequence and structure based features for each provided pair file (e.g. left-bound and right-bound files)
    sequences = {}
    dssp_dicts, rd_dicts, psaia_dfs, coordinate_numbers_list, hsaac_matrices, sequence_feats_dfs = [], [], [], \
                                                                                                   [], [], []
    for struct_idx, raw_pdb_filename in enumerate(raw_pdb_filenames):
        is_rcsb_complex = source_type.lower() in ['rcsb', 'evcoupling', 'casp_capri']

        # Extract the FASTA sequence(s) for a given PDB file
        sequences = find_fasta_sequences_for_pdb_file(sequences,
                                                      raw_pdb_filename,
                                                      external_feats_dir,
                                                      struct_idx,
                                                      is_rcsb_complex,
                                                      original_pair)

        # Avoid redundant feature fetching for RCSB complexes
        if (is_rcsb_complex and struct_idx == 0) or not is_rcsb_complex:
            # Derive BioPython structure and residues for the given PDB file
            structure = PDB_PARSER.get_structure(original_pair.complex, raw_pdb_filename)  # PDB structure
            # Filter out all hetero residues including waters to leave only amino and nucleic acids
            residues = [residue for residue in Selection.unfold_entities(structure, 'R')
                        if residue.get_id()[0] == ' ' and residue.get_parent().id in chains_selected]

            # Extract DSSP secondary structure (SS) and relative solvent accessibility (RSA) values for the 1st model
            dssp_dict = get_dssp_dict_for_pdb_model(structure[0], raw_pdb_filename)  # Only for 1st model

            # Extract residue depth (RD) values for each PDB model using MSMS
            rd_dict = get_msms_rd_dict_for_pdb_model(structure[0])  # RD only retrieved for first model

            # Get protrusion indices using PSAIA
            psaia_filepath = os.path.relpath(os.path.splitext(os.path.split(raw_pdb_filename)[-1])[0])
            psaia_filenames = [path for path in Path(external_feats_dir).rglob(f'{psaia_filepath}*.tbl')]
            if len(psaia_filenames) == 0:
                psaia_filenames = [path for path in Path(external_feats_dir).parent.rglob(f'{psaia_filepath}*.tbl')]
            psaia_filename = psaia_filenames[0]
            psaia_df = get_df_from_psaia_tbl_file(psaia_filename)

            # Extract half-sphere exposure (HSE) statistics for each PDB model (including HSAAC and CN values)
            similarity_matrix, coordinate_numbers = get_similarity_matrix(get_coords(residues))
            hsaac_matrix = get_hsaac_for_pdb_residues(residues, similarity_matrix, raw_pdb_filename)

            # Retrieve pre-generated sequence features (i.e. transition and emission probabilities via HH-suite3)
            seq_file_index = 'src0' if struct_idx == 0 else 'src1'
            file = os.path.split(original_pair.srcs[seq_file_index])[-1]
            sequence_feats_filepath = os.path.join(external_feats_dir, db.get_pdb_code(file)[1:3], file)
            sequence_feats_df = pd.read_pickle(sequence_feats_filepath)

            # Collect gathered features for later postprocessing below
            dssp_dicts.append(dssp_dict)
            rd_dicts.append(rd_dict)
            psaia_dfs.append(psaia_df)
            coordinate_numbers_list.append(coordinate_numbers)
            hsaac_matrices.append(hsaac_matrix)
            sequence_feats_dfs.append(sequence_feats_df)

    # -------------
    # DataFrame 0
    # -------------

    # Determine which feature data structures to pull features out of for the first DataFrame
    df0_dssp_dict = dssp_dicts[0]
    df0_rd_dict = rd_dicts[0]
    df0_hsaac_matrix = hsaac_matrices[0]
    df0_coordinate_numbers = coordinate_numbers_list[0]
    df0_raw_pdf_filename = raw_pdb_filenames[0]
    df0_psaia_df = psaia_dfs[0]
    df0_sequence_feats_df = sequence_feats_dfs[0]

    # Add SS and RSA values to the residues in the first dataframe, df0, of a pair of dataframes
    df0: pd.DataFrame = original_pair.df0

    # Iterate through each residue in the first structure and collect extracted features for training and reporting
    residue_counter = 0
    for index, row in df0.iterrows():
        # Aggregate features for each residues' alpha-carbon (CA) atom
        is_ca_atom = 'CA' in row.atom_name

        # Parse information from residue ID
        residue_id = row.residue.strip().lstrip("-+")
        residue_is_inserted = not residue_id.isdigit()
        residue_id = int(residue_id) if not residue_is_inserted else residue_id

        # Collect features for each residue
        dssp_ss_value_for_atom = get_dssp_value_for_residue(df0_dssp_dict, 'SS', row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_SS
        dssp_rsa_value_for_atom = get_dssp_value_for_residue(df0_dssp_dict, 'RSA', row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_RSA
        msms_rd_value_for_atom = get_msms_rd_value_for_residue(df0_rd_dict, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_RD
        protrusion_index_for_atom = get_protrusion_index_for_residue(df0_psaia_df, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_PROTRUSION_INDEX
        hsaac_for_atom = get_hsaac_for_residue(df0_hsaac_matrix, residue_counter, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_HSAAC
        cn_value_for_atom = get_cn_value_for_residue(
            df0_coordinate_numbers, residue_counter, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_CN
        sequence_feats_for_atom = get_sequence_feats_for_residue(
            df0_sequence_feats_df, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_SEQUENCE_FEATS
        norm_vec_for_atom = get_norm_vec_for_residue(original_pair.df0, row, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_NORM_VEC

        # Handle missing normal vectors
        if is_ca_atom and np.nan in norm_vec_for_atom:
            logging.info(f'Normal vector missing for df0 residue {row.residue}'
                         f'in chain {row.chain} in file {df0_raw_pdf_filename}')
            df0_amide_norm_vecs.append(np.array(norm_vec_for_atom))
        elif is_ca_atom:  # Normal vector was found successfully
            df0_amide_norm_vecs.append(norm_vec_for_atom[0])  # 2D array with a single inner array -> 1D array
        else:
            df0_amide_norm_vecs.append(np.array(norm_vec_for_atom))

        # Aggregate feature values
        df0_ss_values += dssp_ss_value_for_atom
        df0_rsa_values.append(dssp_rsa_value_for_atom)
        df0_rd_values.append(msms_rd_value_for_atom)
        df0_protrusion_indices.append(protrusion_index_for_atom)
        df0_hsaacs.append(hsaac_for_atom)
        df0_cn_values.append(cn_value_for_atom)
        df0_sequence_feats.append(sequence_feats_for_atom)

        # Report presence of inserted residues
        if is_ca_atom and residue_is_inserted:
            logging.info('Found inserted df0 residue entry for residue ' + row.resname + ': '
                         + '(\'' + row.chain + '\', \'' + row.residue + '\')')

        # Increment residue counter for each alpha-carbon encountered
        if 'CA' in row.atom_name:
            residue_counter += 1

    # Normalize df0 residue features to be in range [0, 1]
    df0_rd_values = min_max_normalize_feature_array(np.array(df0_rd_values).reshape(-1, 1))
    df0_protrusion_indices = min_max_normalize_feature_array(np.array(df0_protrusion_indices))
    df0_cn_values = min_max_normalize_feature_array(np.array(df0_cn_values).reshape(-1, 1))

    # Insert new df0 features
    df0.insert(5, 'ss_value', df0_ss_values, False)
    df0.insert(6, 'rsa_value', df0_rsa_values, False)
    df0.insert(7, 'rd_value', df0_rd_values, False)
    # Insert all protrusion index fields sequentially
    df0_col_idx = 8
    for struct_idx, col_name in enumerate(PSAIA_COLUMNS):
        df0.insert(df0_col_idx, col_name, df0_protrusion_indices[:, struct_idx], False)
        df0_col_idx += 1
    df0.insert(14, 'hsaac', df0_hsaacs, False)
    df0.insert(15, 'cn_value', df0_cn_values, False)
    df0.insert(16, 'sequence_feats', df0_sequence_feats, False)
    df0.insert(17, 'amide_norm_vec', df0_amide_norm_vecs, False)

    # -------------
    # DataFrame 1
    # -------------

    # Determine which feature data structures to pull features out of for the second DataFrame
    df1_dssp_dict = dssp_dicts[0] if single_raw_pdb_file_provided else dssp_dicts[1]
    df1_rd_dict = rd_dicts[0] if single_raw_pdb_file_provided else rd_dicts[1]
    df1_hsaac_matrix = hsaac_matrices[0] if single_raw_pdb_file_provided else hsaac_matrices[1]
    df1_coordinate_numbers = coordinate_numbers_list[0] if single_raw_pdb_file_provided else coordinate_numbers_list[1]
    df1_raw_pdf_filename = raw_pdb_filenames[0] if single_raw_pdb_file_provided else raw_pdb_filenames[1]
    df1_psaia_df = psaia_dfs[0] if single_raw_pdb_file_provided else psaia_dfs[1]
    df1_sequence_feats_df = sequence_feats_dfs[0] if single_raw_pdb_file_provided else sequence_feats_dfs[1]

    # Add SS and RSA values to the residues in the second dataframe, df1, of a pair of dataframes
    df1: pd.DataFrame = original_pair.df1

    # Iterate through each residue in the second structure and collect extracted features for training and reporting
    residue_counter = 0
    for index, row in df1.iterrows():
        # Aggregate features for each residues' alpha-carbon (CA) atom
        is_ca_atom = 'CA' in row.atom_name

        # Parse information from residue ID
        residue_id = row.residue.strip().lstrip("-+")
        residue_is_inserted = not residue_id.isdigit()
        residue_id = int(residue_id) if not residue_is_inserted else residue_id

        # Collect features for each residue
        dssp_ss_value_for_atom = get_dssp_value_for_residue(df1_dssp_dict, 'SS', row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_SS
        dssp_rsa_value_for_atom = get_dssp_value_for_residue(df1_dssp_dict, 'RSA', row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_RSA
        msms_rd_value_for_atom = get_msms_rd_value_for_residue(df1_rd_dict, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_RD
        protrusion_index_for_atom = get_protrusion_index_for_residue(df1_psaia_df, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_PROTRUSION_INDEX
        hsaac_for_atom = get_hsaac_for_residue(df1_hsaac_matrix, residue_counter, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_HSAAC
        cn_value_for_atom = get_cn_value_for_residue(
            df1_coordinate_numbers, residue_counter, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_CN
        sequence_feats_for_atom = get_sequence_feats_for_residue(
            df1_sequence_feats_df, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_SEQUENCE_FEATS
        norm_vec_for_atom = get_norm_vec_for_residue(original_pair.df1, row, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_NORM_VEC

        # Handle missing normal vectors
        if is_ca_atom and np.nan in norm_vec_for_atom:
            logging.info(f'Normal vector missing for df1 residue {row.residue}'
                         f'in chain {row.chain} in file {df1_raw_pdf_filename}')
            df1_amide_norm_vecs.append(np.array(norm_vec_for_atom))
        elif is_ca_atom:  # Normal vector was found successfully
            df1_amide_norm_vecs.append(norm_vec_for_atom[0])  # 2D array with a single inner array -> 1D array
        else:
            df1_amide_norm_vecs.append(norm_vec_for_atom)

        # Aggregate feature values
        df1_ss_values += dssp_ss_value_for_atom
        df1_rsa_values.append(dssp_rsa_value_for_atom)
        df1_rd_values.append(msms_rd_value_for_atom)
        df1_protrusion_indices.append(protrusion_index_for_atom)
        df1_hsaacs.append(hsaac_for_atom)
        df1_cn_values.append(cn_value_for_atom)
        df1_sequence_feats.append(sequence_feats_for_atom)

        # Report presence of inserted residues
        if is_ca_atom and residue_is_inserted:
            logging.info('Found inserted df1 residue entry for residue ' + row.resname + ': '
                         + '(\'' + row.chain + '\', \'' + row.residue + '\')')

        # Increment residue counter for each alpha-carbon encountered
        if is_ca_atom:
            residue_counter += 1

    # Normalize df1 residue features to be in range [0, 1]
    df1_rd_values = min_max_normalize_feature_array(np.array(df1_rd_values).reshape(-1, 1))
    df1_protrusion_indices = min_max_normalize_feature_array(np.array(df1_protrusion_indices))
    df1_cn_values = min_max_normalize_feature_array(np.array(df1_cn_values).reshape(-1, 1))

    # Insert new df1 features
    df1.insert(5, 'ss_value', df1_ss_values, False)
    df1.insert(6, 'rsa_value', df1_rsa_values, False)
    df1.insert(7, 'rd_value', df1_rd_values, False)
    # Insert all protrusion index fields sequentially
    df1_col_idx = 8
    for struct_idx, col_name in enumerate(PSAIA_COLUMNS):
        df1.insert(df1_col_idx, col_name, df1_protrusion_indices[:, struct_idx], False)
        df1_col_idx += 1
    df1.insert(14, 'hsaac', df1_hsaacs, False)
    df1.insert(15, 'cn_value', df1_cn_values, False)
    df1.insert(16, 'sequence_feats', df1_sequence_feats, False)
    df1.insert(17, 'amide_norm_vec', df1_amide_norm_vecs, False)

    # Reconstruct a Pair representing a complex of interacting proteins
    pair = pa.Pair(complex=original_pair.complex, df0=df0, df1=df1,
                   pos_idx=original_pair.pos_idx, neg_idx=original_pair.neg_idx,
                   srcs=original_pair.srcs, id=original_pair.id, sequences=sequences)
    return pair


def collect_dataset_statistics(output_dir: str):
    """Aggregate statistics for a postprocessed dataset."""
    dataset_statistics = DEFAULT_DATASET_STATISTICS
    # Look at each .dill file in the given output directory
    pair_filenames = [pair_filename.as_posix() for pair_filename in Path(output_dir).rglob('*.dill')]
    for i in tqdm(range(len(pair_filenames))):
        postprocessed_pair: pa.Pair = pd.read_pickle(pair_filenames[i])

        # Keep track of how many complexes have already been postprocessed
        dataset_statistics['num_of_processed_complexes'] += 1

        # -------------
        # DataFrame 0
        # -------------

        # Grab first structure's DataFrame (CA atoms by default)
        df0: pd.DataFrame = postprocessed_pair.df0[postprocessed_pair.df0['atom_name'].apply(lambda x: x == 'CA')]

        # Increment feature counters
        dataset_statistics['num_of_df0_interface_residues'] += len(postprocessed_pair.pos_idx[:, 0])
        dataset_statistics['num_of_valid_df0_ss_values'] += len(df0[df0['ss_value'] != '-'])
        dataset_statistics['num_of_valid_df0_rsa_values'] += len(df0[~df0['rsa_value'].isna()])
        dataset_statistics['num_of_valid_df0_rd_values'] += len(df0[~df0['rd_value'].isna()])
        num_nonzero_protrusion_indices = len(df0[(~df0['avg_cx'].isna()) & (~df0['s_avg_cx'].isna())
                                                 & (~df0['s_ch_avg_cx'].isna()) & (~df0['s_ch_s_avg_cx'].isna())
                                                 & (~df0['max_cx'].isna()) & (~df0['min_cx'].isna())])
        dataset_statistics['num_of_valid_df0_protrusion_indices'] += num_nonzero_protrusion_indices
        for hsaac_array in df0['hsaac']:
            if np.sum(np.isnan(hsaac_array)) == 0:
                dataset_statistics['num_of_valid_df0_hsaacs'] += 1
        dataset_statistics['num_of_valid_df0_cn_values'] += len(df0[~df0['cn_value'].isna()])
        for sequence_array in df0['sequence_feats']:
            if np.sum(np.isnan(sequence_array.astype(np.float))) == 0:
                dataset_statistics['num_of_valid_df0_sequence_feats'] += 1
        for amide_normal_vec in df0['amide_norm_vec']:
            if np.sum(np.isnan(amide_normal_vec.astype(np.float))) == 0:
                dataset_statistics['num_of_valid_df0_amide_normal_vecs'] += 1

        # Increment total residue count for first structure
        dataset_statistics['num_of_df0_residues'] += len(df0)

        # -------------
        # DataFrame 1
        # -------------

        # Grab second structure's DataFrame (CA atoms by default)
        df1: pd.DataFrame = postprocessed_pair.df1[postprocessed_pair.df1['atom_name'].apply(lambda x: x == 'CA')]

        # Increment feature counters
        dataset_statistics['num_of_df1_interface_residues'] += len(postprocessed_pair.pos_idx[:, 1])
        dataset_statistics['num_of_valid_df1_ss_values'] += len(df1[df1['ss_value'] != '-'])
        dataset_statistics['num_of_valid_df1_rsa_values'] += len(df1[~df1['rsa_value'].isna()])
        dataset_statistics['num_of_valid_df1_rd_values'] += len(df1[~df1['rd_value'].isna()])
        num_nonzero_protrusion_indices = len(df1[(~df1['avg_cx'].isna()) & (~df1['s_avg_cx'].isna())
                                                 & (~df1['s_ch_avg_cx'].isna()) & (~df1['s_ch_s_avg_cx'].isna())
                                                 & (~df1['max_cx'].isna()) & (~df1['min_cx'].isna())])
        dataset_statistics['num_of_valid_df1_protrusion_indices'] += num_nonzero_protrusion_indices
        for hsaac_array in df1['hsaac']:
            if np.sum(np.isnan(hsaac_array)) == 0:
                dataset_statistics['num_of_valid_df1_hsaacs'] += 1
        dataset_statistics['num_of_valid_df1_cn_values'] += len(df1[~df1['cn_value'].isna()])
        for sequence_array in df1['sequence_feats']:
            if np.sum(np.isnan(sequence_array.astype(np.float))) == 0:
                dataset_statistics['num_of_valid_df1_sequence_feats'] += 1
        for amide_normal_vec in df1['amide_norm_vec']:
            if np.sum(np.isnan(amide_normal_vec.astype(np.float))) == 0:
                dataset_statistics['num_of_valid_df1_amide_normal_vecs'] += 1

        # Increment total residue count for second structure
        dataset_statistics['num_of_df1_residues'] += len(df1)

        # Aggregate pair counts for logging final statistics
        num_unique_res_pairs = len(df0) * len(df1)
        dataset_statistics['num_of_pos_res_pairs'] += len(postprocessed_pair.pos_idx)
        dataset_statistics['num_of_neg_res_pairs'] += (num_unique_res_pairs - len(postprocessed_pair.pos_idx))
        dataset_statistics['num_of_res_pairs'] += num_unique_res_pairs

    return dataset_statistics


def determine_nan_fill_value(column: pd.Series, imputation_method='median'):
    """Determine whether to fill NaNs in a given column with the column's requested fill value or instead with zero."""
    cleaned_imputation_method = imputation_method.strip().lower()
    # Determine user-requested imputation (fill) value
    if cleaned_imputation_method == 'mean':
        imputation_value = column.mean()
    elif cleaned_imputation_method == 'min':
        imputation_value = column.min()
    elif cleaned_imputation_method == 'max':
        imputation_value = column.max()
    elif cleaned_imputation_method == 'zero':
        imputation_value = 0
    else:  # Default to replacing NaNs with the column's median value
        imputation_value = column.median()
    return imputation_value if column.isna().sum().sum() <= NUM_ALLOWABLE_NANS else 0


def impute_missing_feature_values(input_pair_filename: str, output_pair_filename: str,
                                  impute_atom_features: bool, advanced_logging: bool):
    """Impute missing feature values in a postprocessed dataset."""
    # Look at a .dill file in the given output directory
    postprocessed_pair: pa.Pair = pd.read_pickle(input_pair_filename)

    # -------------
    # DataFrames
    # -------------
    updated_dfs = []
    for df_idx, df in enumerate([postprocessed_pair.df0, postprocessed_pair.df1]):
        # Collect indices of requested atoms - all atom indices if atom feature imputation requested and only CAs if not
        atom_indices = df[df['atom_name'].apply(lambda x: x == 'CA')].index if not impute_atom_features else df.index

        # Grab structure's feature columns
        df.iloc[:, 6].replace('NA', np.nan, inplace=True)  # Replace NA (i.e. DSSP-stringified NaN value) with np.nan
        numeric_feat_cols = df.iloc[atom_indices, 6:14]
        hsaacs = []
        for hsaac in df.iloc[atom_indices, 14]:
            hsaacs.append(hsaac)
        hsaacs = pd.DataFrame(np.array(hsaacs))
        cns = df.iloc[atom_indices, 15]
        sequence_feats = pd.DataFrame(np.array([sequence_feats for sequence_feats in df.iloc[atom_indices, 16]]))
        amide_normal_vecs = pd.DataFrame(np.array([sequence_feats for sequence_feats in df.iloc[atom_indices, 17]]))

        # Initially inspect whether there are missing features in the structure
        numeric_feat_cols_have_nan = numeric_feat_cols.isna().values.any()
        hsaacs_have_nan = hsaacs.isna().values.any()
        cns_have_nan = cns.isna().values.any()
        sequence_feats_have_nan = sequence_feats.isna().values.any()
        amide_normal_vecs_have_nan = amide_normal_vecs.isna().values.any()
        nan_found = numeric_feat_cols_have_nan or \
                    hsaacs_have_nan or \
                    cns_have_nan or \
                    sequence_feats_have_nan or \
                    amide_normal_vecs_have_nan
        if nan_found:
            if advanced_logging:
                logging.info(f"""Before Feature Imputation:
                                | df{df_idx} (from {input_pair_filename}) contained at least one NaN value |
                                df{df_idx}_numeric_feat_cols_have_nan: {numeric_feat_cols_have_nan}
                                df{df_idx}_hsaacs_have_nan: {hsaacs_have_nan}
                                df{df_idx}_cns_have_nan: {cns_have_nan}
                                df{df_idx}_sequence_feats_have_nan: {sequence_feats_have_nan}
                                df{df_idx}_amide_normal_vecs_have_nan: {amide_normal_vecs_have_nan}""")

        # Impute structure's missing feature values uniquely for each column
        numeric_feat_cols = numeric_feat_cols.apply(lambda col: col.fillna(determine_nan_fill_value(col)), axis=0)
        df.iloc[atom_indices, 6:14] = numeric_feat_cols

        hsaacs = hsaacs.apply(lambda col: col.fillna(determine_nan_fill_value(col)), axis=0)
        for atom_index, df_hsaac in zip(atom_indices, hsaacs.values.tolist()):
            df.at[atom_index, 'hsaac'] = df_hsaac

        cns = cns.fillna(determine_nan_fill_value(cns))
        df.iloc[atom_indices, 15] = cns

        sequence_feats = sequence_feats.apply(lambda col: col.fillna(determine_nan_fill_value(col)), axis=0)
        for atom_index, seq_feats in zip(atom_indices, [np.array(lst) for lst in sequence_feats.values.tolist()]):
            df.at[atom_index, 'sequence_feats'] = seq_feats

        amide_normal_vecs = amide_normal_vecs.apply(lambda col: col.fillna(
            determine_nan_fill_value(col, imputation_method='zero')), axis=0)
        for atom_index, nv in zip(atom_indices, [np.array(lst) for lst in amide_normal_vecs.values.tolist()]):
            df.at[atom_index, 'amide_norm_vec'] = nv

        numeric_feat_cols_have_nan = numeric_feat_cols.isna().values.any()
        hsaacs_have_nan = hsaacs.isna().values.any()
        cns_have_nan = cns.isna().values.any()
        sequence_feats_have_nan = sequence_feats.isna().values.any()
        amide_normal_vecs_have_nan = amide_normal_vecs.isna().values.any()
        nan_found = numeric_feat_cols_have_nan or \
                    hsaacs_have_nan or \
                    cns_have_nan or \
                    sequence_feats_have_nan or \
                    amide_normal_vecs_have_nan
        if nan_found:
            raise Exception(f"""After Feature Imputation:
                            | df{df_idx} (from {input_pair_filename}) contained at least one NaN value |
                            df{df_idx}_numeric_feat_cols_have_nan: {numeric_feat_cols_have_nan}
                            df{df_idx}_hsaacs_have_nan: {hsaacs_have_nan}
                            df{df_idx}_cns_have_nan: {cns_have_nan}
                            df{df_idx}_sequence_feats_have_nan: {sequence_feats_have_nan}
                            df{df_idx}_amide_normal_vecs_have_nan: {amide_normal_vecs_have_nan}""")

        updated_dfs.append(df)

    # Reconstruct a feature-imputed Pair representing a complex of interacting proteins
    df0, df1 = updated_dfs[0], updated_dfs[1]
    feature_imputed_pair = pa.Pair(complex=postprocessed_pair.complex, df0=df0, df1=df1,
                                   pos_idx=postprocessed_pair.pos_idx, neg_idx=postprocessed_pair.neg_idx,
                                   srcs=postprocessed_pair.srcs, id=postprocessed_pair.id,
                                   sequences=postprocessed_pair.sequences)

    # Write into current pair_filename
    with open(output_pair_filename, 'wb') as f:
        pickle.dump(feature_imputed_pair, f)


def get_raw_pdb_filename_from_interim_filename(interim_filename: str, raw_pdb_dir: str, source_type: str):
    """Get raw pdb filename from interim filename."""
    pdb_name = interim_filename
    slash_tokens = pdb_name.split(os.path.sep)
    slash_dot_tokens = slash_tokens[-1].split(".")
    raw_pdb_filename = os.path.join(raw_pdb_dir, slash_tokens[-2], slash_dot_tokens[0]) + '.' + slash_dot_tokens[1] if \
        source_type.lower() in ['rcsb', 'evcoupling', 'casp_capri'] else \
        os.path.join(raw_pdb_dir, slash_dot_tokens[0].split('_')[0], slash_dot_tokens[0]) + '.' + slash_dot_tokens[1]
    return raw_pdb_filename


def __should_keep_postprocessed(raw_pdb_dir: str, pair_filename: str, source_type: str):
    """Determine if given pair filename corresponds to a pair of structures, both with DSSP-derivable secondary structure features."""
    # pair_name example: 20gs.pdb1_0
    raw_pdb_filenames = []
    pickle._dill._reverse_typemap['SliceType'] = slice  # Python 2 to 3 type conversion
    pair = pd.read_pickle(pair_filename)
    for i, interim_filename in enumerate(pair.srcs.values()):  # Unbound source filenames to be converted to bound ones
        # Identify if a given complex contains DSSP-derivable secondary structure features
        raw_pdb_filenames.append(get_raw_pdb_filename_from_interim_filename(interim_filename, raw_pdb_dir, source_type))
        pair_dssp_dict = get_dssp_dict_for_pdb_file(raw_pdb_filenames[i])
        if not pair_dssp_dict and source_type != 'db5':
            return pair, raw_pdb_filenames[i], False  # Discard pair missing DSSP-derivable secondary structure features
        if pair.df0.shape[0] > ATOM_COUNT_LIMIT or pair.df1.shape[0] > ATOM_COUNT_LIMIT:
            return pair, raw_pdb_filenames[i], False  # Discard pair exceeding atom count limit to reduce comp. complex.
    return pair, raw_pdb_filenames, True


def validate_input_filenames(complex_filename: str, struct1_filename: str, struct2_filename: str, unbound: bool):
    """Ensure that the user has entered either a valid complex file path or two valid structure file paths."""
    # Report an invalid file path for a protein complex if one was not provided
    complex_filename = '' if not complex_filename else complex_filename
    if not unbound and not os.path.exists(complex_filename):
        complex_filename = input(
            "The file path \"" + complex_filename + "\" does not exist. Please try again.\nComplex fn: ")
        while not os.path.exists(complex_filename):
            complex_filename = input(
                "The file path \"" + complex_filename + "\" does not exist. Please try again.\nComplex fn: ")

    # Report an invalid file path for a PDB file pair if one was not provided
    struct1_filename = '' if not struct1_filename else struct1_filename
    struct2_filename = '' if not struct2_filename else struct2_filename
    if unbound and not (os.path.exists(struct1_filename) and os.path.exists(struct2_filename)):
        if not os.path.exists(struct1_filename):
            struct1_filename = input(
                "The file path \"" + struct1_filename + "\" does not exist. Please try again.\nStruct1 fn: ")
            while not os.path.exists(struct1_filename):
                struct1_filename = input(
                    "The file path \"" + struct1_filename + "\" does not exist. Please try again.\nStruct1 fn: ")

        if not os.path.exists(struct2_filename):
            struct2_filename = input(
                "The file path \"" + struct2_filename + "\" does not exist. Please try again.\nStruct2 fn: ")
            while not os.path.exists(struct2_filename):
                struct2_filename = input(
                    "The file path \"" + struct2_filename + "\" does not exist. Please try again.\nStruct2 fn: ")


def extract_gz_archive(gz_archive_filename: str):
    """Run extraction logic to turn raw GZ archive into extracted data."""
    _, ext = os.path.splitext(gz_archive_filename)
    if not os.path.exists(_):
        with gzip.open(gz_archive_filename, 'rb') as f_in:
            with open(_, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                f_out.flush()  # Manually flush buffer


def extract_pairs_from_complex(neighbor_def: str, cutoff: int, complex_df: pd.DataFrame, complex_filename: str):
    """Find all chain pairs in a given complex."""
    get_neighbors = nb.build_get_neighbors(neighbor_def, cutoff)
    complex_df = complex_df[complex_df['model'] == complex_df['model'][0]]
    complex_obj = Complex(name=complex_df.pdb_name[0] if complex_df.pdb_name[0] else '',
                          bound_filenames=[], unbound_filenames=[])
    complex_obj.bound_filenames.append(complex_filename)
    pairs, num_chains = pa._get_all_chain_pairs(neighbor_def, complex_obj, complex_df,
                                                get_neighbors, complex_filename, False)
    return pairs, num_chains


def write_data_file_to_storage(output_fn: str, postprocessed_pair, df0: pd.DataFrame, df1: pd.DataFrame):
    """Cache derived inference data in non-volatile storage."""
    if not os.path.exists(output_fn):
        _, ext = os.path.splitext(output_fn)
        ext_ref_index = -5 if ext == '.dill' else -6  # -6 if ".torch"
        # Write into output_fn if not existent
        with open(output_fn, 'wb') as f:
            pickle.dump(postprocessed_pair, f)
        with open(output_fn[:ext_ref_index] + '_' + 'df0' + output_fn[ext_ref_index:], 'wb') as f:
            pickle.dump(df0, f)
        with open(output_fn[:ext_ref_index] + '_' + 'df1' + output_fn[ext_ref_index:], 'wb') as f:
            pickle.dump(df1, f)


def find_pruned_pairs_with_missing_pdb_files(logger, raw_pdb_dir: str, pair_filename: str):
    """Find pruned pairs for which their original PDB file is not already downloaded locally."""
    missing_pdb_filenames = []
    pickle._dill._reverse_typemap['SliceType'] = slice  # Python 2 to 3 type conversion
    pair = pd.read_pickle(pair_filename)
    for i, interim_filename in enumerate(pair.srcs.values()):  # Unbound source filenames to be converted to bound ones
        raw_pdb_filename = get_raw_pdb_filename_from_interim_filename(interim_filename, raw_pdb_dir, 'rcsb')
        current_pdb_dir = os.path.join(raw_pdb_dir, db.get_pdb_code(raw_pdb_filename)[1:3])
        if not os.path.exists(raw_pdb_filename):
            missing_pdb_filenames.append(raw_pdb_filename)
        if not os.path.exists(current_pdb_dir):
            logger.info(f'Making new dir {current_pdb_dir}')
            os.mkdir(current_pdb_dir)
    return missing_pdb_filenames


def fetch_missing_pdb_files(logger, missing_pdb_filenames: List[str]):
    """Fetch PDB files for pruned pairs for which their original PDB file was not already stored locally."""
    archive_ext = '.gz'
    for missing_pdb_filename in missing_pdb_filenames:
        pdb_code = db.get_pdb_code(missing_pdb_filename)[1:3]
        pdb_name = db.get_pdb_name(missing_pdb_filename) + archive_ext
        pdb_gz_full_filepath = missing_pdb_filename + archive_ext
        ftp_request_url = RCSB_BASE_URL + pdb_code + '/' + pdb_name
        # Download requested GZ archive with FTP
        logger.info(f'Fetching {pdb_name} from RCSB')
        with closing(request.urlopen(ftp_request_url)) as r:
            with open(pdb_gz_full_filepath, 'wb') as f:
                shutil.copyfileobj(r, f)
        logger.info(f'Done fetching {pdb_name} from RCSB')
        # Extract downloaded GZ archive to output_dir in correct subdirectory
        logger.info(f'Extracting {pdb_name}')
        extract_gz_archive(pdb_gz_full_filepath)
        logger.info(f'Done extracting {pdb_name}')


def download_missing_pruned_pair_pdbs(logger, raw_pdb_dir: str, pair_filename: str):
    """Find pruned pairs for which their original PDB file is not already downloaded locally and download their PDBs."""
    missing_pdb_filenames = find_pruned_pairs_with_missing_pdb_files(logger, raw_pdb_dir, pair_filename)
    missing_pdb_filenames = list(set(missing_pdb_filenames))
    fetch_missing_pdb_files(logger, missing_pdb_filenames)


def log_dataset_statistics(logger, dataset_statistics: dict):
    """Output pair postprocessing statistics."""
    logger.info(f'{dataset_statistics["num_of_processed_complexes"]} complexes copied')

    logger.info(f'{dataset_statistics["num_of_df0_residues"]} residues found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_df1_residues"]} residues found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_df0_interface_residues"]} residues found in df0 interfaces in total')
    logger.info(f'{dataset_statistics["num_of_df1_interface_residues"]} residues found in df1 interfaces in total')

    logger.info(f'{dataset_statistics["num_of_df0_interface_residues"] / dataset_statistics["num_of_df0_residues"]}'
                f' percent of total df0 residues found in interfaces on average')
    logger.info(f'{dataset_statistics["num_of_df1_interface_residues"] / dataset_statistics["num_of_df1_residues"]}'
                f' percent of total df1 residues found in interfaces on average')

    logger.info(f'{dataset_statistics["num_of_pos_res_pairs"]} positive residue pairs found in total')
    logger.info(f'{dataset_statistics["num_of_neg_res_pairs"]} negative residue pairs found in total')

    logger.info(f'{dataset_statistics["num_of_pos_res_pairs"] / dataset_statistics["num_of_res_pairs"]}'
                f' positive residue pairs found in complexes on average')
    logger.info(f'{dataset_statistics["num_of_neg_res_pairs"] / dataset_statistics["num_of_res_pairs"]}'
                f' negative residue pairs found in complexes on average')

    logger.info(f'{dataset_statistics["num_of_valid_df0_ss_values"]}'
                f' valid secondary structure (SS) values found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_ss_values"]}'
                f' valid secondary structure (SS) values found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_rsa_values"]}'
                f' valid relative solvent accessibility (RSA) values found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_rsa_values"]}'
                f' valid relative solvent accessibility (RSA) values found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_rd_values"]}'
                f' valid residue depth (RD) values found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_rd_values"]}'
                f' valid residue depth (RD) values found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_protrusion_indices"]}'
                f' valid protrusion indices found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_protrusion_indices"]}'
                f' valid protrusion indices found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_hsaacs"]}'
                f' valid half-sphere amino acid compositions (HSAACs) found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_hsaacs"]}'
                f' valid half-sphere amino acid compositions (HSAACs) found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_cn_values"]}'
                f' valid coordinate number (CN) values found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_cn_values"]}'
                f' valid coordinate number (CN) values found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_sequence_feats"]}'
                f' valid sequence feature arrays found for df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_sequence_feats"]}'
                f' valid sequence feature arrays found for df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_amide_normal_vecs"]}'
                f' valid amide normal vectors found for df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_amide_normal_vecs"]}'
                f' valid amide normal vectors found for df1 structures in total')


def convert_pair_pickle_to_hdf5(pickle_filepath: Path, hdf5_filepath: Path):
    # Load pickle file
    with open(str(pickle_filepath), 'rb') as f:
        pair_data = dill.load(f)

    # Save data to HDF5 file
    hkl_data = list(pair_data)
    hkl.dump(hkl_data, str(hdf5_filepath))


def convert_pair_hdf5_to_pickle(hdf5_filepath: Path) -> pa.Pair:
    # Load HDF5 file as pickle object
    hkl_data = hkl.load(str(hdf5_filepath))
    pair_data = pa.Pair(*hkl_data)
    return pair_data


def convert_pair_hdf5_to_hdf5_file(hdf5_filepath: Path) -> h5py.File:
    # Load HDF5 file as HDF5 file
    data = h5py.File(str(hdf5_filepath), 'r')
    return data


def remove_digits_from_filename(filename):
    pattern = r'\d+$'  # Matches one or more digits at the end of the string
    result = re.sub(pattern, '', filename)
    return result


def download_pdb_file(pdb_filename, save_path, obsolete_pdb=False):
    # Construct the URL for the specific PDB file
    pdb_url = f"https://files.rcsb.org/download/{pdb_filename}"

    # Send a GET request to the URL
    response = requests.get(pdb_url)

    # Check if the request was successful (HTTP status code 200)
    download_succeeded = False
    if response.status_code == 200:
        # Save the response content (PDB file) to the specified location
        download_succeeded = True
        with open(save_path, "wb") as file:
            file.write(response.content)
        print(f"Download of PDB file {pdb_filename} completed successfully.")
    else:
        print(f"Failed to download PDB file {pdb_filename}. Status code: {response.status_code}")

    if not download_succeeded and not obsolete_pdb:
        # Try downloading the default (non-biological assembly) version of an obsolete PDB structure
        download_pdb_file(remove_digits_from_filename(pdb_filename), save_path, obsolete_pdb=True)


def annotate_idr_residues(pickle_filepaths: List[Path]):
    # Process each pickle file input
    for pickle_filepath in pickle_filepaths:
        # Load pickle file
        try:
            with open(str(pickle_filepath), 'rb') as f:
                pair_data = dill.load(f)
        except Exception as e:
            logging.error(f"Failed to load pickle file {pickle_filepath} due to: {e}")
            with open("idr_annotation_errors.txt", "a") as f:
                f.write(str(pickle_filepath) + "\n")
            continue
        # Find IDR interface residues
        annotating_new_residues = False
        annotated_residue_sequences = copy.deepcopy(pair_data.sequences)
        for key, sequence in pair_data.sequences.items():
            key_to_be_annotated = key in ["l_b", "r_b", "l_u", "r_u"]
            idrs_to_be_annotated = (
                (f"{key}_idr_annotations" not in pair_data.sequences) or
                (f"{key}_idr_propensities" not in pair_data.sequences)
            )
            if key_to_be_annotated and idrs_to_be_annotated:
                annotating_new_residues = True
                fasta_filepath = os.path.join(tempfile.mkdtemp(), f'{key}.fasta')
                with open(fasta_filepath, 'w') as file:
                    file.write(f'>sequence\n{sequence}\n')
                # Run `flDPnn` to predict which residues reside in an IDR
                output_filepath = os.path.join(tempfile.mkdtemp(), "fldpnn_results.csv")
                cmd = [
                    'docker', 'run', '-i', 'sinaghadermarzi/fldpnn',
                    'fldpnn/dockerinout_nofunc', '<', fasta_filepath, '>', output_filepath
                ]
                try:
                    subprocess.run(' '.join(cmd), shell=True, check=True)
                    logging.info("Docker command executed successfully.")
                except subprocess.CalledProcessError as e:
                    logging.info(f"Error executing Docker command: {e}")
                # Parse output from `flDPnn`
                output_df = pd.read_csv(output_filepath, skiprows=1)
                annotated_residue_sequences[f"{key}_idr_annotations"] = output_df['Binary Prediction for Disorder'].tolist()
                annotated_residue_sequences[f"{key}_idr_propensities"] = output_df['Predicted Score for Disorder'].tolist()
                assert len(annotated_residue_sequences[f"{key}_idr_annotations"]) == len(pair_data.sequences[key]), "IDR annotations must match length of input sequence."
                assert len(annotated_residue_sequences[f"{key}_idr_propensities"]) == len(pair_data.sequences[key]), "IDR propensities must match length of input sequence."
        # Record IDR residue annotations and propensities within pickle file
        if annotating_new_residues:
            pair_data.sequences.update({
                key: value
                for key, value in annotated_residue_sequences.items()
            })
            with open(str(pickle_filepath), 'wb') as f:
                dill.dump(pair_data, f)


def process_raw_file_into_dgl_graphs(raw_filepath: str, new_graph_dir: str, processed_filepath: str,
                                     edge_dist_cutoff: float, edge_limit: int, self_loops: bool):
    """Process each postprocessed pair into a graph pair."""
    # Retrieve specified complex
    bound_complex = pd.read_pickle(raw_filepath)

    # Convert each DataFrame into its corresponding DGLGraph representations
    graph0 = convert_df_to_dgl_graph(bound_complex.df0, raw_filepath, edge_dist_cutoff, edge_limit, self_loops)
    graph1 = convert_df_to_dgl_graph(bound_complex.df1, raw_filepath, edge_dist_cutoff, edge_limit, self_loops)

    # Derive inter-protein node-node (i.e. residue-residue) interaction matrix (Interacting = 1 and Non-Interacting = 0)
    labels = np.zeros((len(bound_complex.df0), len(bound_complex.df1)))
    for idx in bound_complex.pos_idx:
        row1_row_num = bound_complex.df0.loc[[idx[0]]].index[0]
        row2_row_num = bound_complex.df1.loc[[idx[1]]].index[0]
        idx1 = bound_complex.df0.index.get_loc(row1_row_num)
        idx2 = bound_complex.df1.index.get_loc(row2_row_num)
        labels[idx1][idx2] = 1

    # Assemble and cache pair of graphs with its corresponding labels matrix
    graph_pair_payload = [graph0, graph1, labels]
    if not os.path.exists(new_graph_dir):
        os.mkdir(new_graph_dir)
    torch.save(graph_pair_payload, processed_filepath)


def make_dgl_graphs(output_dir: str, input_file: str, ext: str,
                    edge_dist_cutoff: float, edge_limit: int, self_loops: bool):
    """Convert all final 'raw' complexes into DGL graph pairs with their corresponding labels (interaction) matrix."""
    # Construct output file path for a DGL graph pair
    input_file_parts = input_file.split(os.sep)
    processed_filepath = os.path.join(output_dir, input_file_parts[-2],
                                      f'{os.path.splitext(input_file_parts[-1])[0]}' + f'.{ext}')
    # Create new DGL graph pair file
    if not os.path.exists(processed_filepath):
        process_raw_file_into_dgl_graphs(input_file,
                                         os.path.join(*processed_filepath.split(os.sep)[:-1]),
                                         processed_filepath,
                                         edge_dist_cutoff,
                                         edge_limit,
                                         self_loops)


def convert_df_to_dgl_graph(struct_df: pd.DataFrame, input_file: str, edge_dist_cutoff: float,
                            edge_limit: int, self_loops: bool) -> dgl.DGLGraph:
    """Transform a given DataFrame of residues into a corresponding DGL graph."""
    # Derive node features, with edges being defined via a k-nearest neighbors approach and a maximum distance threshold
    node_coords, node_feats, srcs, dsts = \
        prot_df_to_dgl_graph_feats(struct_df, FEAT_COLS, ALLOWABLE_FEATS, edge_dist_cutoff, edge_limit)

    # Construct DGLGraph from the edges derived above, initially making the graph unidirectional
    graph = dgl.graph((srcs, dsts))

    # Remove self-loops (if requested)
    if not self_loops:
        graph = dgl.remove_self_loop(graph)
        srcs = graph.edges()[0]
        dsts = graph.edges()[1]

    # Manually add isolated nodes (i.e. those with no connected edges) to the graph
    if len(node_feats) > graph.number_of_nodes():
        num_of_isolated_nodes = len(node_feats) - graph.number_of_nodes()
        logging.info(f'{num_of_isolated_nodes} isolated node(s) detected in {input_file}. Adding manually to graph...')
        graph.add_nodes(num_of_isolated_nodes)

    """Encode node features and labels in graph"""
    # One-hot features for each residue
    graph.ndata['h'] = node_feats  # [num_residues_in_struct_df, num_node_features]
    # Cartesian coordinates for each residue
    graph.ndata['x'] = node_coords  # [num_residues_in_struct_df, 3]

    """Encode edge features and labels in graph"""
    # Float edge weight scaled by Euclidean distance between source and destination node
    graph.edata['w'] = min_max_normalize_feature_tensor(
        ((node_coords[dsts] - node_coords[srcs]) ** 2).sum(dim=-1, keepdim=True))  # [num_edges, 1]
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
    np.nan_to_num(angles, copy=False, nan=0.0, posinf=None, neginf=None)
    graph.edata['a'] = min_max_normalize_feature_tensor(torch.from_numpy(angles))

    # Explicitly make graph bidirectional
    graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True)

    return graph


def get_edges(node_coords: FloatTensor, edge_dist_cutoff: float, edge_limit: int):
    """Query for neighbors of each residue within a certain distance to define edges in a corresponding DGL graph."""
    dist_map = torch.cdist(node_coords, node_coords).squeeze()
    cmap = dist_map <= edge_dist_cutoff
    if torch.sum(cmap) > edge_limit:
        dist_cutoff = torch.topk(torch.flatten(dist_map), edge_limit, largest=False)[0][-1].item()
        cmap = dist_map <= dist_cutoff
    edge_list = torch.where(cmap)
    srcs = edge_list[0]
    dsts = edge_list[1]
    return srcs, dsts
