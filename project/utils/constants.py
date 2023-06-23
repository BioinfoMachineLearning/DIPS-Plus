import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import CaPPBuilder

# Cluster-specific limit to the number of compute nodes available to each Slurm job
MAX_NODES_PER_JOB = 4

# Dataset-global node count limits to restrict computational learning complexity
ATOM_COUNT_LIMIT = 17500  # Default filter for both datasets when encoding complexes at an atom-based level

# From where we can get bound PDB complexes
RCSB_BASE_URL = 'ftp://ftp.wwpdb.org/pub/pdb/data/biounit/coordinates/divided/'

# The PDB codes of structures added between DB4 and DB5 (to be used for testing dataset)
DB5_TEST_PDB_CODES = ['3R9A', '4GAM', '3AAA', '4H03', '1EXB',
                      '2GAF', '2GTP', '3RVW', '3SZK', '4IZ7',
                      '4GXU', '3BX7', '2YVJ', '3V6Z', '1M27',
                      '4FQI', '4G6J', '3BIW', '3PC8', '3HI6',
                      '2X9A', '3HMX', '2W9E', '4G6M', '3LVK',
                      '1JTD', '3H2V', '4DN4', 'BP57', '3L5W',
                      '3A4S', 'CP57', '3DAW', '3VLB', '3K75',
                      '2VXT', '3G6D', '3EO1', '4JCV', '4HX3',
                      '3F1P', '3AAD', '3EOA', '3MXW', '3L89',
                      '4M76', 'BAAD', '4FZA', '4LW4', '1RKE',
                      '3FN1', '3S9D', '3H11', '2A1A', '3P57']

# Postprocessing logger dictionary
DEFAULT_DATASET_STATISTICS = dict(num_of_processed_complexes=0, num_of_df0_residues=0, num_of_df1_residues=0,
                                  num_of_df0_interface_residues=0, num_of_df1_interface_residues=0,
                                  num_of_pos_res_pairs=0, num_of_neg_res_pairs=0, num_of_res_pairs=0,
                                  num_of_valid_df0_ss_values=0, num_of_valid_df1_ss_values=0,
                                  num_of_valid_df0_rsa_values=0, num_of_valid_df1_rsa_values=0,
                                  num_of_valid_df0_rd_values=0, num_of_valid_df1_rd_values=0,
                                  num_of_valid_df0_protrusion_indices=0, num_of_valid_df1_protrusion_indices=0,
                                  num_of_valid_df0_hsaacs=0, num_of_valid_df1_hsaacs=0,
                                  num_of_valid_df0_cn_values=0, num_of_valid_df1_cn_values=0,
                                  num_of_valid_df0_sequence_feats=0, num_of_valid_df1_sequence_feats=0,
                                  num_of_valid_df0_amide_normal_vecs=0, num_of_valid_df1_amide_normal_vecs=0)

# Parsing utilities for PDB files (i.e. relevant for sequence and structure analysis)
PDB_PARSER = PDBParser()
CA_PP_BUILDER = CaPPBuilder()

# Dict for converting three letter codes to one letter codes
D3TO1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
RES_NAMES_LIST = list(D3TO1.keys())

# PSAIA features to encode as DataFrame columns
PSAIA_COLUMNS = ['avg_cx', 's_avg_cx', 's_ch_avg_cx', 's_ch_s_avg_cx', 'max_cx', 'min_cx']

# Constants for calculating half sphere exposure statistics
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY-'
AMINO_ACID_IDX = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))

# Default fill values for missing features
HSAAC_DIM = 42  # We have 2 + (2 * 20) HSAAC values from the two instances of the unknown residue symbol '-'
DEFAULT_MISSING_FEAT_VALUE = np.nan
DEFAULT_MISSING_SS = '-'
DEFAULT_MISSING_RSA = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_RD = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_PROTRUSION_INDEX = [DEFAULT_MISSING_FEAT_VALUE for _ in range(6)]
DEFAULT_MISSING_HSAAC = [DEFAULT_MISSING_FEAT_VALUE for _ in range(HSAAC_DIM)]
DEFAULT_MISSING_CN = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_SEQUENCE_FEATS = np.array([DEFAULT_MISSING_FEAT_VALUE for _ in range(27)])
DEFAULT_MISSING_NORM_VEC = [DEFAULT_MISSING_FEAT_VALUE for _ in range(3)]

# Default number of NaN values allowed in a specific column before imputing missing features of the column with zero
NUM_ALLOWABLE_NANS = 5

# Features to be one-hot encoded during graph processing and what their values could be
FEAT_COLS = [
    'resname',  # By default, leave out one-hot encoding of residues' type to decrease feature redundancy
    'ss_value',
    'rsa_value',
    'rd_value'
]
FEAT_COLS.extend(
    PSAIA_COLUMNS +
    ['hsaac',
     'cn_value',
     'sequence_feats',
     'amide_norm_vec',
     # 'element'  # For atom-level learning only
     ])

ALLOWABLE_FEATS = [
    # By default, leave out one-hot encoding of residues' type to decrease feature redundancy
    ["TRP", "PHE", "LYS", "PRO", "ASP", "ALA", "ARG", "CYS", "VAL", "THR",
     "GLY", "SER", "HIS", "LEU", "GLU", "TYR", "ILE", "ASN", "MET", "GLN"],
    ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-'],  # Populated 1D list means restrict column feature values by list values
    [],  # Empty list means take scalar value as is
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [[]],  # Doubly-nested, empty list means take first-level nested list as is
    [],
    [[]],
    [[]],
    # ['C', 'O', 'N', 'S']  # For atom-level learning only
]
