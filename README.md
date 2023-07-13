<div align="center">

# DIPS-Plus

The Enhanced Database of Interacting Protein Structures for Interface Prediction

[![Paper](http://img.shields.io/badge/paper-arxiv.2106.04362-B31B1B.svg)](https://arxiv.org/abs/2106.04362)  [![CC BY 4.0][cc-by-shield]][cc-by] [![Primary Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5134732.svg)](https://doi.org/10.5281/zenodo.5134732) [![Supplementary Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8140981.svg)](https://doi.org/10.5281/zenodo.8140981)

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

[comment]: <> ([![Conference]&#40;http://img.shields.io/badge/NeurIPS-2021-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-35-2021&#41;)

[<img src="https://twixes.gallerycdn.vsassets.io/extensions/twixes/pypi-assistant/1.0.3/1589834023190/Microsoft.VisualStudio.Services.Icons.Default" width="50"/>](https://pypi.org/project/DIPS-Plus/)

</div>

## Versioning

* Version 1.0.0: Initial release of DIPS-Plus and DB5-Plus (DOI: 10.5281/zenodo.4815267)
* Version 1.1.0: Minor updates to DIPS-Plus and DB5-Plus' tar archives (DOI: 10.5281/zenodo.5134732)
  * DIPS-Plus' final 'raw' tar archive now includes standardized 80%-20% lists of filenames for training and validation, respectively
  * DB5-Plus' final 'raw' tar archive now includes (optional) standardized lists of filenames for training and validation, respectively
  * DB5-Plus' final 'raw' tar archive now also includes a corrected (i.e. de-duplicated) list of filenames for its 55 test complexes
    * Benchmark results included in our paper were run after this issue was resolved
    * However, if you ran experiments using DB5-Plus' filename list for its test complexes, please re-run them using the latest list
* Version 1.2.0: Minor additions to DIPS-Plus tar archives, including new residue-level intrinsic disorder region annotations and raw Jackhmmer-small BFD MSAs (Supplementary Data DOI: 10.5281/zenodo.8071136)
* Version 1.3.0: Minor additions to DIPS-Plus tar archives, including new FoldSeek-based structure-focused training and validation splits, residue-level (scalar) disorder propensities, and a Graphein-based featurization pipeline (Supplementary Data DOI: 10.5281/zenodo.8140981)

## How to set up

First, download Mamba (if not already downloaded):
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # Accept all terms and install to the default location
rm Mambaforge-$(uname)-$(uname -m).sh  # (Optionally) Remove installer after using it
source ~/.bashrc  # Alternatively, one can restart their shell session to achieve the same result
```

Then, create and configure Mamba environment:

```bash
# Clone project:
git clone https://github.com/BioinfoMachineLearning/DIPS-Plus
cd DIPS-Plus

# Create Conda environment using local 'environment.yml' file:
mamba env create -f environment.yml
conda activate DIPS-Plus  # Note: One still needs to use `conda` to (de)activate environments

# Install local project as package:
pip3 install -e .
```

To install PSAIA for feature generation, install GCC 10 for PSAIA:

```bash
# Install GCC 10 for Ubuntu 20.04:
sudo apt install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/ppa
sudo apt update
sudo apt install gcc-10 g++-10

# Or install GCC 10 for Arch Linux/Manjaro:
yay -S gcc10
```

Then install QT4 for PSAIA:

```bash
# Install QT4 for Ubuntu 20.04:
sudo add-apt-repository ppa:rock-core/qt4
sudo apt update
sudo apt install libqt4* libqtcore4 libqtgui4 libqtwebkit4 qt4* libxext-dev

# Or install QT4 for Arch Linux/Manjaro:
yay -S qt4
```

Conclude by compiling PSAIA from source:

```bash
# Select the location to install the software:
MY_LOCAL=~/Programs

# Download and extract PSAIA's source code:
mkdir "$MY_LOCAL"
cd "$MY_LOCAL"
wget http://complex.zesoi.fer.hr/data/PSAIA-1.0-source.tar.gz
tar -xvzf PSAIA-1.0-source.tar.gz

# Compile PSAIA (i.e., a GUI for PSA):
cd PSAIA_1.0_source/make/linux/psaia/
qmake-qt4 psaia.pro
make

# Compile PSA (i.e., the protein structure analysis (PSA) program):
cd ../psa/
qmake-qt4 psa.pro
make

# Compile PIA (i.e., the protein interaction analysis (PIA) program):
cd ../pia/
qmake-qt4 pia.pro
make

# Test run any of the above-compiled programs:
cd "$MY_LOCAL"/PSAIA_1.0_source/bin/linux
# Test run PSA inside a GUI:
./psaia/psaia
# Test run PIA through a terminal:
./pia/pia
# Test run PSA through a terminal:
./psa/psa
```

Lastly, install Docker following the instructions from https://docs.docker.com/engine/install/

## How to generate protein feature inputs
In our [feature generation notebook](notebooks/feature_generation.ipynb), we provide examples of how users can generate the protein features described in our [accompanying manuscript](https://arxiv.org/abs/2106.04362) for individual protein inputs.

## How to use data
In our [data usage notebook](notebooks/data_usage.ipynb), we provide examples of how users might use DIPS-Plus (or DB5-Plus) for downstream analysis or prediction tasks. For example, to train a new NeiA model with DB5-Plus as its cross-validation dataset, first download DB5-Plus' raw files and process them via the `data_usage` notebook:

```bash
mkdir -p project/datasets/DB5/final
wget https://zenodo.org/record/5134732/files/final_raw_db5.tar.gz -O project/datasets/DB5/final/final_raw_db5.tar.gz
tar -xzf project/datasets/DB5/final/final_raw_db5.tar.gz -C project/datasets/DB5/final/

# To process these raw files for training and subsequently train a model:
python3 notebooks/data_usage.py
```

## How to split data using FoldSeek
We provide users with the [ability](https://github.com/BioinfoMachineLearning/DIPS-Plus/blob/75775d98f0923faf11fb50639eb58cd510a10ffd/project/datasets/builder/partition_dataset_filenames.py#L486) to perform structure-based splits of the complexes in DIPS-Plus using FoldSeek. This script is designed to allow users to customize how stringent one would like FoldSeek's searches to be for structure-based splitting. Moreover, we provide standardized structure-based splits of DIPS-Plus' complexes in the corresponding [supplementary Zenodo data record](https://zenodo.org/record/8140981).

## How to featurize DIPS-Plus complexes using Graphein
In the new [graph featurization script](https://github.com/BioinfoMachineLearning/DIPS-Plus/blob/main/project/datasets/builder/add_new_feature.py), we provide an example of how users may install new Expasy protein scale features using the Graphein library. The script is designed to be amenable to simple user customization such that users can use this script to insert arbitrary new Graphein-based features into each DIPS-Plus complex's pair file, for downstream tasks.

## Standard DIPS-Plus directory structure

```
DIPS-Plus
│
└───project
     │
     └───datasets
         │
         └───DB5
         │   │
         │   └───final
         │   │   │
         │   │   └───processed  # task-ready features for each dataset example
         │   │   │
         │   │   └───raw  # generic features for each dataset example
         │   │
         │   └───interim
         │   │   │
         │   │   └───complexes  # metadata for each dataset example
         │   │   │
         │   │   └───external_feats  # features curated for each dataset example using external tools
         │   │   │
         │   │   └───pairs  # pair-wise features for each dataset example
         │   │
         │   └───raw  # raw PDB data downloads for each dataset example
         │
         └───DIPS
             │
             └───filters  # filters to apply to each (un-pruned) dataset example
             │
             └───final
             │   │
             │   └───processed  # task-ready features for each dataset example
             │   │
             │   └───raw  # generic features for each dataset example
             │
             └───interim
             │   │
             │   └───complexes  # metadata for each dataset example
             │   │
             │   └───external_feats  # features curated for each dataset example using external tools
             │   │
             │   └───pairs-pruned  # filtered pair-wise features for each dataset example
             │   │
             │   └───parsed  # pair-wise features for each dataset example after initial parsing
             │
             └───raw
                 │
                 └───pdb  # raw PDB data downloads for each dataset example
```

## How to compile DIPS-Plus from scratch

Retrieve protein complexes from the RCSB PDB and build out directory structure:

```bash
# Remove all existing training/testing sample lists
rm project/datasets/DIPS/final/raw/pairs-postprocessed.txt project/datasets/DIPS/final/raw/pairs-postprocessed-train.txt project/datasets/DIPS/final/raw/pairs-postprocessed-val.txt project/datasets/DIPS/final/raw/pairs-postprocessed-test.txt

# Create data directories (if not already created):
mkdir project/datasets/DIPS/raw project/datasets/DIPS/raw/pdb project/datasets/DIPS/interim project/datasets/DIPS/interim/pairs-pruned project/datasets/DIPS/interim/external_feats project/datasets/DIPS/final project/datasets/DIPS/final/raw project/datasets/DIPS/final/processed

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
python3 project/datasets/builder/generate_hhsuite_features.py "$PROJDIR"/project/datasets/DIPS/interim/parsed "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned "$HHSUITE_DB" "$PROJDIR"/project/datasets/DIPS/interim/external_feats --num_cpu_jobs 4 --num_cpus_per_job 8 --num_iter 2 --source_type rcsb --write_file  # Note: After this, one needs to re-run this command with `--read_file` instead

# Generate multiple sequence alignments (MSAs) using a smaller sequence database (if not already created using the standard BFD):
DOWNLOAD_DIR="$HHSUITE_DB_DIR" && ROOT_DIR="${DOWNLOAD_DIR}/small_bfd" && SOURCE_URL="https://storage.googleapis.com/alphafold-databases/reduced_dbs/bfd-first_non_consensus_sequences.fasta.gz" && BASENAME=$(basename "${SOURCE_URL}") && mkdir --parents "${ROOT_DIR}" && aria2c "${SOURCE_URL}" --dir="${ROOT_DIR}" && pushd "${ROOT_DIR}" && gunzip "${ROOT_DIR}/${BASENAME}" && popd  # e.g., Download the small BFD
python3 project/datasets/builder/generate_hhsuite_features.py "$PROJDIR"/project/datasets/DIPS/interim/parsed "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned "$HHSUITE_DB_DIR"/small_bfd "$PROJDIR"/project/datasets/DIPS/interim/external_feats --num_cpu_jobs 4 --num_cpus_per_job 8 --num_iter 2 --source_type rcsb --generate_msa_only --write_file  # Note: After this, one needs to re-run this command with `--read_file` instead

# Identify interfaces within intrinsically disordered regions (IDRs) #
# (1) Pull down the Docker image for `flDPnn`
docker pull docker.io/sinaghadermarzi/fldpnn
# (2) For all sequences in the dataset, predict which interface residues reside within IDRs
python3 project/datasets/builder/annotate_idr_interfaces.py "$PROJDIR"/project/datasets/DIPS/final/raw --num_cpus 16

# Add new features to the filtered pairs, ensuring that the pruned pairs' original PDB files are stored locally for DSSP:
python3 project/datasets/builder/download_missing_pruned_pair_pdbs.py "$PROJDIR"/project/datasets/DIPS/raw/pdb "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned --num_cpus 32 --rank "$1" --size "$2"
python3 project/datasets/builder/postprocess_pruned_pairs.py "$PROJDIR"/project/datasets/DIPS/raw/pdb "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned "$PROJDIR"/project/datasets/DIPS/interim/external_feats "$PROJDIR"/project/datasets/DIPS/final/raw --num_cpus 32

# Partition dataset filenames, aggregate statistics, and impute missing features
python3 project/datasets/builder/partition_dataset_filenames.py "$PROJDIR"/project/datasets/DIPS/final/raw --source_type rcsb --filter_by_atom_count True --max_atom_count 17500 --rank "$1" --size "$2"
python3 project/datasets/builder/collect_dataset_statistics.py "$PROJDIR"/project/datasets/DIPS/final/raw --rank "$1" --size "$2"
python3 project/datasets/builder/log_dataset_statistics.py "$PROJDIR"/project/datasets/DIPS/final/raw --rank "$1" --size "$2"
python3 project/datasets/builder/impute_missing_feature_values.py "$PROJDIR"/project/datasets/DIPS/final/raw --impute_atom_features False --advanced_logging False --num_cpus 32 --rank "$1" --size "$2"

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
python3 project/datasets/builder/impute_missing_feature_values.py "$PROJDIR"/project/datasets/DB5/final/raw --impute_atom_features False --advanced_logging False --num_cpus 32 --rank "$1" --size "$2"

# Optionally convert each postprocessed (final 'raw') complex into a pair of DGL graphs (final 'processed') with labels
python3 project/datasets/builder/convert_complexes_to_graphs.py "$PROJDIR"/project/datasets/DB5/final/raw "$PROJDIR"/project/datasets/DB5/final/processed --num_cpus 32 --edge_dist_cutoff 15.0 --edge_limit 5000 --self_loops True --rank "$1" --size "$2"
```

## How to reassemble DIPS-Plus' "interim" external features

We split the (tar.gz) archive into eight separate parts with
'split -b 4096M interim_external_feats_dips.tar.gz "interim_external_feats_dips.tar.gz.part"'
to upload it to the dataset's primary Zenodo record, so to recover the original archive:

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

## Citation
If you find DIPS-Plus useful in your research, please cite:

```bibtex
@misc{morehead2021dipsplus,
      title={DIPS-Plus: The Enhanced Database of Interacting Protein Structures for Interface Prediction}, 
      author={Alex Morehead and Chen Chen and Ada Sedova and Jianlin Cheng},
      year={2021},
      eprint={2106.04362},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```