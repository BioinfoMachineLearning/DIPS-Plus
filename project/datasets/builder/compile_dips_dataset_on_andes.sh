#!/bin/bash

####################### Batch Headers #########################
#SBATCH -A BIF132
#SBATCH -p batch
#SBATCH -J make_dips_dataset
#SBATCH -t 0-24:00
#SBATCH --mem 224G
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
###############################################################

# Remote paths #
export PROJID=bif132
export PROJDIR=/gpfs/alpine/scratch/"$USER"/$PROJID/Repositories/Lab_Repositories/DIPS-Plus
export PSAIADIR=/ccs/home/"$USER"/Programs/PSAIA_1.0_source/bin/linux/psa

# Default to using the Big Fantastic Database (BFD) of protein sequences (approx. 270GB compressed)
export HHSUITE_DB=/gpfs/alpine/scratch/$USER/$PROJID/Data/Databases/bfd_metaclust_clu_complete_id30_c90_final_seq

# Remote Conda environment #
source "$PROJDIR"/miniconda3/bin/activate

# Load CUDA module for DGL
module load cuda/10.2.89

# Load GCC 10 module for OpenMPI support
module load gcc/10.3.0

# Run dataset compilation scripts
cd "$PROJDIR"/project || exit
rm "$PROJDIR"/project/datasets/DIPS/final/raw/pairs-postprocessed.txt "$PROJDIR"/project/datasets/DIPS/final/raw/pairs-postprocessed-train.txt "$PROJDIR"/project/datasets/DIPS/final/raw/pairs-postprocessed-val.txt "$PROJDIR"/project/datasets/DIPS/final/raw/pairs-postprocessed-test.txt
mkdir "$PROJDIR"/project/datasets/DIPS/raw "$PROJDIR"/project/datasets/DIPS/raw/pdb "$PROJDIR"/project/datasets/DIPS/interim "$PROJDIR"/project/datasets/DIPS/interim/external_feats "$PROJDIR"/project/datasets/DIPS/interim/external_feats/PSAIA "$PROJDIR"/project/datasets/DIPS/interim/external_feats/PSAIA/RCSB "$PROJDIR"/project/datasets/DIPS/final "$PROJDIR"/project/datasets/DIPS/final/raw
rsync -rlpt -v -z --delete --port=33444 --include='*.gz' --include='*/' --exclude '*' rsync.rcsb.org::ftp_data/biounit/coordinates/divided/ "$PROJDIR"/project/datasets/DIPS/raw/pdb

python3 "$PROJDIR"/project/datasets/builder/extract_raw_pdb_gz_archives.py "$PROJDIR"/project/datasets/DIPS/raw/pdb --rank "$1" --size "$2"

python3 "$PROJDIR"/project/datasets/builder/make_dataset.py "$PROJDIR"/project/datasets/DIPS/raw/pdb "$PROJDIR"/project/datasets/DIPS/interim --num_cpus 32 --rank "$1" --size "$2" --source_type rcsb --bound

python3 "$PROJDIR"/project/datasets/builder/prune_pairs.py "$PROJDIR"/project/datasets/DIPS/interim/pairs "$PROJDIR"/project/datasets/DIPS/filters "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned --num_cpus 32 --rank "$1" --size "$2"

python3 "$PROJDIR"/project/datasets/builder/generate_psaia_features.py "$PSAIADIR" "$PROJDIR"/project/datasets/builder/psaia_config_file_dips.txt "$PROJDIR"/project/datasets/DIPS/raw/pdb "$PROJDIR"/project/datasets/DIPS/interim/parsed "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned "$PROJDIR"/project/datasets/DIPS/interim/external_feats --source_type rcsb --rank "$1" --size "$2"
srun python3 "$PROJDIR"/project/datasets/builder/generate_hhsuite_features.py "$PROJDIR"/project/datasets/DIPS/interim/parsed "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned "$HHSUITE_DB" "$PROJDIR"/project/datasets/DIPS/interim/external_feats --rank "$1" --size "$2" --num_cpu_jobs 4 --num_cpus_per_job 8 --num_iter 2 --source_type rcsb --write_file

# Retroactively download the PDB files corresponding to complexes that made it through DIPS-Plus' RCSB complex pruning to reduce storage requirements
python3 "$PROJDIR"/project/datasets/builder/download_missing_pruned_pair_pdbs.py "$PROJDIR"/project/datasets/DIPS/raw/pdb "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned --num_cpus 32 --rank "$1" --size "$2"
srun python3 "$PROJDIR"/project/datasets/builder/postprocess_pruned_pairs.py "$PROJDIR"/project/datasets/DIPS/raw/pdb "$PROJDIR"/project/datasets/DIPS/interim/pairs-pruned "$PROJDIR"/project/datasets/DIPS/interim/external_feats "$PROJDIR"/project/datasets/DIPS/final/raw --num_cpus 32 --rank "$1" --size "$2"

python3 "$PROJDIR"/project/datasets/builder/partition_dataset_filenames.py "$PROJDIR"/project/datasets/DIPS/final/raw --source_type rcsb --filter_by_atom_count True --max_atom_count 17500 --rank "$1" --size "$2"
python3 "$PROJDIR"/project/datasets/builder/collect_dataset_statistics.py "$PROJDIR"/project/datasets/DIPS/final/raw --rank "$1" --size "$2"
python3 "$PROJDIR"/project/datasets/builder/log_dataset_statistics.py "$PROJDIR"/project/datasets/DIPS/final/raw --rank "$1" --size "$2"
python3 "$PROJDIR"/project/datasets/builder/impute_missing_feature_values.py "$PROJDIR"/project/datasets/DIPS/final/raw --impute_atom_features False --num_cpus 32 --rank "$1" --size "$2"

# Optionally convert each postprocessed (final 'raw') complex into a pair of DGL graphs (final 'processed') with labels
python3 "$PROJDIR"/project/datasets/builder/convert_complexes_to_graphs.py "$PROJDIR"/project/datasets/DIPS/final/raw "$PROJDIR"/project/datasets/DIPS/final/processed --num_cpus 32 --edge_dist_cutoff 15.0 --edge_limit 5000 --self_loops True --rank "$1" --size "$2"
