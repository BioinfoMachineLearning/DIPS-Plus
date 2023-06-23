# %% [markdown]
# # Feature generation for PDB file inputs

# %%
import os

import atom3.complex as comp
import atom3.conservation as con
import atom3.neighbors as nb
import atom3.pair as pair
import atom3.parse as parse
import dill as pickle

from pathlib import Path

from project.utils.utils import annotate_idr_residues, impute_missing_feature_values, postprocess_pruned_pair, process_raw_file_into_dgl_graphs

# %% [markdown]
# ### 1. Parse PDB file input to pair-wise features

# %%
pdb_filename = "project/datasets/Input/raw/pdb/2g/12gs.pdb1"  # note: an input PDB must be uncompressed (e.g., not in `.gz` archive format) using e.g., `gunzip`
output_pkl = "project/datasets/Input/interim/parsed/2g/12gs.pdb1.pkl"
complexes_dill = "project/datasets/Input/interim/complexes/complexes.dill"
pairs_dir = "project/datasets/Input/interim/pairs"
pkl_filenames = [output_pkl]
source_type = "rcsb"  # note: this default value will likely work for common use cases (i.e., those concerning bound-state PDB protein complex structure inputs)
neighbor_def = "non_heavy_res"
cutoff = 6  # note: distance threshold (in Angstrom) for classifying inter-chain interactions can be customized here
unbound = False  # note: if `source_type` is set to `rcsb`, this value should likely be `False`

for item in [
    Path(pdb_filename).parent,
    Path(output_pkl).parent,
    Path(complexes_dill).parent,
    pairs_dir,
]:
    os.makedirs(item, exist_ok=True)

# note: the following replicates the logic within `make_dataset.py` for a single PDB file input
parse.parse(
    # note: assumes the PDB file input (i.e., `pdb_filename`) is not compressed
    pdb_filename=pdb_filename,
    output_pkl=output_pkl
)
complexes = comp.get_complexes(filenames=pkl_filenames, type=source_type)
comp.write_complexes(complexes=complexes, output_dill=complexes_dill)
get_neighbors = nb.build_get_neighbors(criteria=neighbor_def, cutoff=cutoff)
get_pairs = pair.build_get_pairs(
    neighbor_def=neighbor_def,
    type=source_type,
    unbound=unbound,
    nb_fn=get_neighbors,
    full=False
)
complexes = comp.read_complexes(input_dill=complexes_dill)
pair.complex_to_pairs(
    complex=list(complexes['data'].values())[0],
    source_type=source_type,
    get_pairs=get_pairs,
    output_dir=pairs_dir
)

# %% [markdown]
# ### 2. Compute sequence-based features using external tools

# %%
psaia_dir = "~/Programs/PSAIA-1.0/bin/linux/psa"  # note: replace this with the path to your local installation of PSAIA
psaia_config_file = "project/datasets/builder/psaia_config_file_dips.txt"  # note: choose `psaia_config_file_dips.txt` according to the `source_type` selected above
file_list_file = os.path.join("project/datasets/Input/interim/external_feats/", 'PSAIA', source_type.upper(), 'pdb_list.fls')
num_cpus = 8
pkl_filename = "project/datasets/Input/interim/parsed/2g/12gs.pdb1.pkl"
output_filename = "project/datasets/Input/interim/external_feats/parsed/2g/12gs.pdb1.pkl"
hhsuite_db = "~/Data/Databases/pfamA_35.0/pfam"  # note: substitute the path to your local HHsuite3 database here
num_iter = 2
msa_only = False

for item in [
    Path(file_list_file).parent,
    Path(output_filename).parent,
]:
    os.makedirs(item, exist_ok=True)

# note: the following replicates the logic within `generate_psaia_features.py` and `generate_hhsuite_features.py` for a single PDB file input
with open(file_list_file, 'w') as file:
    file.write(f'{pdb_filename}\n')  # note: references the `pdb_filename` as defined previously
con.gen_protrusion_index(
    psaia_dir=psaia_dir,
    psaia_config_file=psaia_config_file,
    file_list_file=file_list_file,
)
con.map_profile_hmms(
    num_cpus=num_cpus,
    pkl_filename=pkl_filename,
    output_filename=output_filename,
    hhsuite_db=hhsuite_db,
    source_type=source_type,
    num_iter=num_iter,
    msa_only=msa_only,
)

# %% [markdown]
# ### 3. Compute structure-based features

# %%
from project.utils.utils import __should_keep_postprocessed


raw_pdb_dir = "project/datasets/Input/raw/pdb"
pair_filename = "project/datasets/Input/interim/pairs/2g/12gs.pdb1_0.dill"
source_type = "rcsb"
external_feats_dir = "project/datasets/Input/interim/external_feats/parsed"
output_filename = "project/datasets/Input/final/raw/2g/12gs.pdb1_0.dill"

unprocessed_pair, raw_pdb_filenames, should_keep = __should_keep_postprocessed(raw_pdb_dir, pair_filename, source_type)
if should_keep:
    # note: save `postprocessed_pair` to local storage within `project/datasets/Input/final/raw` for future reference as desired
    postprocessed_pair = postprocess_pruned_pair(
        raw_pdb_filenames=raw_pdb_filenames,
        external_feats_dir=external_feats_dir,
        original_pair=unprocessed_pair,
        source_type=source_type,
    )
    # write into output_filenames if not exist
    os.makedirs(Path(output_filename).parent, exist_ok=True)
    with open(output_filename, 'wb') as f:
        pickle.dump(postprocessed_pair, f)

# %% [markdown]
# ### 4. Embed deep learning-based IDR features

# %%
# note: ensures the Docker image for `flDPnn` is available locally before trying to run inference with the model
# !docker pull docker.io/sinaghadermarzi/fldpnn

input_pair_filename = "project/datasets/Input/final/raw/2g/12gs.pdb1_0.dill"
pickle_filepaths = [input_pair_filename]

annotate_idr_residues(
    pickle_filepaths=pickle_filepaths
)

# %% [markdown]
# ### 5. Impute missing feature values (optional)

# %%
input_pair_filename = "project/datasets/Input/final/raw/2g/12gs.pdb1_0.dill"
output_pair_filename = "project/datasets/Input/final/raw/2g/12gs.pdb1_0_imputed.dill"
impute_atom_features = False
advanced_logging = False

impute_missing_feature_values(
    input_pair_filename=input_pair_filename,
    output_pair_filename=output_pair_filename,
    impute_atom_features=impute_atom_features,
    advanced_logging=advanced_logging,
)

# %% [markdown]
# ### 6. Convert pair-wise features into graph inputs (optional)

# %%
raw_filepath = "project/datasets/Input/final/raw/2g/12gs.pdb1_0_imputed.dill"
new_graph_dir = "project/datasets/Input/final/processed/2g"
processed_filepath = "project/datasets/Input/final/processed/2g/12gs.pdb1.pt"
edge_dist_cutoff = 15.0
edge_limit = 5000
self_loops = True

os.makedirs(new_graph_dir, exist_ok=True)

process_raw_file_into_dgl_graphs(
    raw_filepath=raw_filepath,
    new_graph_dir=new_graph_dir,
    processed_filepath=processed_filepath,
    edge_dist_cutoff=edge_dist_cutoff,
    edge_limit=edge_limit,
    self_loops=self_loops,
)


