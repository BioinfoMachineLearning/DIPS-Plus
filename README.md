<div align="center">

# DIPS-Plus

The Enhanced Database of Interacting Protein Structures for Interface Prediction

[comment]: <> ([![Paper]&#40;http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)

[comment]: <> ([![Conference]&#40;http://img.shields.io/badge/NeurIPS-2021-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-35-2021&#41;)

</div>

## How to run creation tools

First, install and configure Conda environment:

```bash
# Clone project:
git clone https://github.com/amorehead/DIPS-Plus

# Change to project directory:
cd DIPS-Plus

# (If on HPC cluster) Download latest 64-bit Linux version of Miniconda and activate it:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  # Specify install directory
source miniconda/bin/activate  # Assuming environment created above is called 'miniconda'

# Create Conda environment using local 'environment.yml' file:
conda env create --name DIPS-Plus -f environment.yml

# Create Conda environment in a particular directory using local 'environment.yml' file:
conda env create --prefix MY-VENV-DIR -f environment.yml

# Activate Conda environment located in the current directory:
conda activate DIPS-Plus

# (Optional) Activate Conda environment located in another directory:
conda activate MY-VENV-DIR

# (Optional) Deactivate the currently-activated Conda environment:
conda deactivate

# Perform a full update on the Conda environment described in 'environment.yml':
conda env update -f environment.yml --prune

# (Optional) To remove this long prefix in your shell prompt, modify the env_prompt setting in your .condarc file with:
conda config --set env_prompt '({name})'
 ```

(If on HPC cluster) Install all project dependencies:

```bash
# Install project as a pip dependency in the Conda environment currently activated:
pip3 install -e .

# Install external pip dependencies in the Conda environment currently activated:
pip3 install -r requirements.txt
 ```

## Default DIPS-Plus directory structure

```
DIPS-Plus
│
└───project
│    │
│    └───datasets
│    │   │
│    │   └───builder
│    │   │
│    │   └───DB5
│    │   │   │
│    │   │   └───final
│    │   │   │   │
│    │   │   │   └───raw
│    │   │   │
│    │   │   └───interim
│    │   │   │   │
│    │   │   │   └───complexes
│    │   │   │   │
│    │   │   │   └───external_feats
│    │   │   │   │
│    │   │   │   └───pairs
│    │   │   │
│    │   │   └───raw
│    │   │   │
│    │   │   README
│    │   │
│    │   └───DIPS
│    │       │
│    │       └───filters
│    │       │
│    │       └───final
│    │       │   │
│    │       │   └───raw
│    │       │
│    │       └───interim
│    │       │   │
│    │       │   └───complexes
│    │       │   │
│    │       │   └───external_feats
│    │       │   │
│    │       │   └───pairs-pruned
│    │       │
│    │       └───raw
│    │           │
│    │           └───pdb
│    │
│    └───utils
│        constants.py
│        utils.py
│
.gitignore
environment.yml
LICENSE
README.md
requirements.txt
setup.cfg
setup.py
```

## How to compile DIPS-Plus from scratch

Retrieve protein complexes from the RCSB PDB and build out directory structure:

```bash
# Remove all existing training/testing sample lists
rm project/datasets/DIPS/final/raw/pairs-postprocessed.txt project/datasets/DIPS/final/raw/pairs-postprocessed-train.txt project/datasets/DIPS/final/raw/pairs-postprocessed-val.txt project/datasets/DIPS/final/raw/pairs-postprocessed-test.txt

# Create data directories (if not already created):
mkdir project/datasets/DIPS/raw project/datasets/DIPS/raw/pdb project/datasets/DIPS/interim project/datasets/DIPS/interim/external_feats project/datasets/DIPS/final project/datasets/DIPS/final/raw project/datasets/DIPS/final/processed

# Download the raw PDB files:
rsync -rlpt -v -z --delete --port=33444 --include='*.gz' --include='*.xz' --include='*/' --exclude '*' \
rsync.rcsb.org::ftp_data/biounit/coordinates/divided/ project/datasets/DIPS/raw/pdb

# Extract the raw PDB files:
python3 project/datasets/builder/extract_raw_pdb_gz_archives.py project/datasets/DIPS/raw/pdb

# Process the raw PDB data into associated pair files:
python3 project/datasets/builder/make_dataset.py project/datasets/DIPS/raw/pdb project/datasets/DIPS/interim --num_cpus 28 --source_type rcsb --bound

# Apply additional filtering criteria:
python3 project/datasets/builder/prune_pairs.py project/datasets/DIPS/interim/pairs project/datasets/DIPS/filters project/datasets/DIPS/interim/pairs-pruned --num_cpus 28

# Generate externally-sourced features:
python3 project/datasets/builder/generate_psaia_features.py "$PSAIADIR" "$PROJDIR"/project/datasets/builder/psaia_config_file_dips.txt "$PROJDIR"/project/datasets/DIPS/raw/pdb "$PROJDIR"/project/datasets/DIPS/interim/parsed "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned "$PROJDIR"/project/datasets/DIPS/interim/external_feats --source_type rcsb
python3 project/datasets/builder/generate_hhsuite_features.py "$PROJDIR"/project/datasets/DIPS/interim/parsed "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned "$HHSUITE_DB" "$PROJDIR"/project/datasets/DIPS/interim/external_feats --num_cpu_jobs 4 --num_cpus_per_job 8 --num_iter 2 --source_type rcsb --write_file

# Add new features to the filtered pairs, ensuring that the pruned pairs' original PDB files are stored locally for DSSP:
python3 project/datasets/builder/download_missing_pruned_pair_pdbs.py "$PROJDIR"/project/datasets/DIPS/raw/pdb "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned --num_cpus 32 --rank "$1" --size "$2"
python3 project/datasets/builder/postprocess_pruned_pairs.py "$PROJDIR"/project/datasets/DIPS/raw/pdb "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned "$PROJDIR"/project/datasets/DIPS/interim/external_feats "$PROJDIR"/project/datasets/DIPS/final/raw --num_cpus 32

# Partition dataset filenames, aggregate statistics, and impute missing features
python3 project/datasets/builder/partition_dataset_filenames.py "$PROJDIR"/project/datasets/DIPS/final/raw --source_type rcsb --filter_by_atom_count True --max_atom_count 17500 --rank "$1" --size "$2"
python3 project/datasets/builder/collect_dataset_statistics.py "$PROJDIR"/project/datasets/DIPS/final/raw --rank "$1" --size "$2"
python3 project/datasets/builder/log_dataset_statistics.py "$PROJDIR"/project/datasets/DIPS/final/raw --rank "$1" --size "$2"
python3 project/datasets/builder/impute_missing_feature_values.py "$PROJDIR"/project/datasets/DIPS/final/raw --impute_atom_features False --num_cpus 32 --rank "$1" --size "$2"

# Optionally convert each postprocessed (final 'raw') complex into a pair of DGL graphs (final 'processed') with labels
python3 project/datasets/builder/convert_complexes_to_graphs.py "$PROJDIR"/project/datasets/DIPS/final/raw "$PROJDIR"/project/datasets/DIPS/final/processed --num_cpus 32 --edge_dist_cutoff 15.0 --edge_limit 5000 --self_loops True --rank "$1" --size "$2"
```

## How to assemble DB5-Plus

Fetch prepared protein complexes from Dataverse:

```bash
# Download the prepared DB5 files:
wget -O project/datasets/DB5.tar.gz https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/H93ZKK/BXXQCG

# Extract downloaded DB5 archive:
tar -xzf project/datasets/DB5.tar.gz --directory project/datasets/

# Remove (now) redundant DB5 archive and other miscellaneous files:
rm project/datasets/DB5.tar.gz project/datasets/DB5/.README.swp
rm project/datasets/DB5.tar.gz project/datasets/DB5/.README.swp
rm -rf project/datasets/DB5/interim project/datasets/DB5/processed

# Create relevant interim and final data directories:
mkdir project/datasets/DB5/interim project/datasets/DB5/interim/external_feats
mkdir project/datasets/DB5/final project/datasets/DB5/final/raw project/datasets/DB5/final/processed

# Construct DB5 dataset pairs:
python3 project/datasets/builder/make_dataset.py "$PROJDIR"/project/datasets/DB5/raw "$PROJDIR"/project/datasets/DB5/interim --num_cpus 32 --source_type db5 --unbound

# Generate externally-sourced features:
python3 project/datasets/builder/generate_psaia_features.py "$PSAIADIR" "$PROJDIR"/project/datasets/builder/psaia_config_file_db5.txt "$PROJDIR"/project/datasets/DB5/raw "$PROJDIR"/project/datasets/DB5/interim/parsed "$PROJDIR"/project/datasets/DB5/interim/parsed "$PROJDIR"/project/datasets/DB5/interim/external_feats --source_type db5
python3 project/datasets/builder/generate_hhsuite_features.py "$PROJDIR"/project/datasets/DB5/interim/parsed "$PROJDIR"/project/datasets/DB5/interim/parsed "$HHSUITE_DB" "$PROJDIR"/project/datasets/DB5/interim/external_feats --num_cpu_jobs 4 --num_cpus_per_job 8 --num_iter 2 --source_type db5 --write_file

# Add new features to the filtered pairs:
python3 project/datasets/builder/postprocess_pruned_pairs.py "$PROJDIR"/project/datasets/DB5/raw "$PROJDIR"/project/datasets/DB5/interim/pairs "$PROJDIR"/project/datasets/DB5/interim/external_feats "$PROJDIR"/project/datasets/DB5/final/raw --num_cpus 32 --source_type db5

# Partition dataset filenames, aggregate statistics, and impute missing features
python3 project/datasets/builder/partition_dataset_filenames.py "$PROJDIR"/project/datasets/DB5/final/raw --source_type db5 --rank "$1" --size "$2"
python3 project/datasets/builder/collect_dataset_statistics.py "$PROJDIR"/project/datasets/DB5/final/raw --rank "$1" --size "$2"
python3 project/datasets/builder/log_dataset_statistics.py "$PROJDIR"/project/datasets/DB5/final/raw --rank "$1" --size "$2"
python3 project/datasets/builder/impute_missing_feature_values.py "$PROJDIR"/project/datasets/DB5/final/raw --impute_atom_features False --num_cpus 32 --rank "$1" --size "$2"

# Optionally convert each postprocessed (final 'raw') complex into a pair of DGL graphs (final 'processed') with labels
python3 project/datasets/builder/convert_complexes_to_graphs.py "$PROJDIR"/project/datasets/DB5/final/raw "$PROJDIR"/project/datasets/DB5/final/processed --num_cpus 32 --edge_dist_cutoff 15.0 --edge_limit 5000 --self_loops True --rank "$1" --size "$2"
```

## How to reassemble DIPS-Plus' "interim" external features

We split the (tar.gz) archive into eight separate parts with
'split -b 4096M interim_external_feats_dips.tar.gz "interim_external_feats_dips.tar.gz.part"'
to upload it to Zenodo, so to recover the original archive:

```bash
# Reassemble external features archive with 'cat'
cat interim_external_feats_dips.tar.gz.parta* >interim_external_feats_dips.tar.gz
```

## Python 2 to 3 pickle file solution

While using Python 3 in this project, you may encounter the following error if you try to postprocess '.dill' pruned
pairs that were created using Python 2.

ModuleNotFoundError: No module named 'dill.dill'

1. To resolve it, ensure that the 'dill' package's version is greater than 0.3.2.
2. If the problem persists, edit the pickle.py file corresponding to your Conda environment's Python 3 installation (
   e.g. ~/DIPS-Plus/venv/lib/python3.8/pickle.py) and add the statement

```python
if module == 'dill.dill': module = 'dill._dill'
```

to the end of the

```python
if self.proto < 3 and self.fix_imports:
```

block in the Unpickler class' find_class() function
(e.g. line 1577 of Python 3.8.5's pickle.py).

### Citation

```
@article{morehead2021dips,
  title = {DIPS-Plus: The Enhanced Database of Interacting Protein Structures for Interface Prediction},
  author = {Alex Morehead, Chen Chen, Ada Sedova, and Jianlin Cheng},
  year = {2021},
  eprint = {N/A},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```
