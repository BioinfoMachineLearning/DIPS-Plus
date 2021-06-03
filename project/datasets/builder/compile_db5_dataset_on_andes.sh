#!/bin/bash

####################### Batch Headers #########################
#SBATCH -A BIP198
#SBATCH -p batch
#SBATCH -J make_db5_dataset
#SBATCH -t 0-12:00
#SBATCH --mem 224G
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
###############################################################

# Remote paths #
export PROJID=bif132
export PROJDIR=/gpfs/alpine/scratch/"$USER"/$PROJID/Repositories/Lab_Repositories/DIPS-Plus
export PSAIADIR=/ccs/home/"$USER"/Programs/PSAIA_1.0_source/bin/linux/psa
export OMP_NUM_THREADS=8

# Default to using the Big Fantastic Database (BFD) of protein sequences (approx. 270GB compressed)
export HHSUITE_DB=/gpfs/alpine/scratch/$USER/$PROJID/Data/Databases/bfd_metaclust_clu_complete_id30_c90_final_seq

# Remote Conda environment #
source "$PROJDIR"/miniconda3/bin/activate

# Load CUDA module for DGL
module load cuda/10.2.89

# Run dataset compilation scripts
cd "$PROJDIR"/project || exit
wget -O "$PROJDIR"/project/datasets/DB5.tar.gz https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/H93ZKK/BXXQCG
tar -xzf "$PROJDIR"/project/datasets/DB5.tar.gz --directory "$PROJDIR"/project/datasets/
rm "$PROJDIR"/project/datasets/DB5.tar.gz "$PROJDIR"/project/datasets/DB5/.README.swp
rm -rf "$PROJDIR"/project/datasets/DB5/interim "$PROJDIR"/project/datasets/DB5/processed
mkdir "$PROJDIR"/project/datasets/DB5/interim "$PROJDIR"/project/datasets/DB5/interim/external_feats "$PROJDIR"/project/datasets/DB5/interim/external_feats/PSAIA "$PROJDIR"/project/datasets/DB5/interim/external_feats/PSAIA/DB5 "$PROJDIR"/project/datasets/DB5/final "$PROJDIR"/project/datasets/DB5/final/raw

python "$PROJDIR"/project/datasets/builder/make_dataset.py "$PROJDIR"/project/datasets/DB5/raw "$PROJDIR"/project/datasets/DB5/interim --num_cpus 32 --rank "$1" --size "$2" --source_type db5

python "$PROJDIR"/project/datasets/builder/generate_psaia_features.py "$PSAIADIR" "$PROJDIR"/project/datasets/builder/psaia_config_file_db5.txt "$PROJDIR"/project/datasets/DB5/raw "$PROJDIR"/project/datasets/DB5/interim/parsed "$PROJDIR"/project/datasets/DB5/interim/parsed "$PROJDIR"/project/datasets/DB5/interim/external_feats --source_type db5 --rank "$1" --size "$2"
srun python "$PROJDIR"/project/datasets/builder/generate_hhsuite_features.py "$PROJDIR"/project/datasets/DB5/interim/parsed "$PROJDIR"/project/datasets/DB5/interim/parsed "$HHSUITE_DB" "$PROJDIR"/project/datasets/DB5/interim/external_feats --rank "$1" --size "$2" --num_cpu_jobs 4 --num_cpus_per_job 8 --num_iter 2 --source_type db5 --write_file

srun python "$PROJDIR"/project/datasets/builder/postprocess_pruned_pairs.py "$PROJDIR"/project/datasets/DB5/raw "$PROJDIR"/project/datasets/DB5/interim/pairs "$PROJDIR"/project/datasets/DB5/interim/external_feats "$PROJDIR"/project/datasets/DB5/final/raw --num_cpus 32 --rank "$1" --size "$2" --source_type db5

python "$PROJDIR"/project/datasets/builder/partition_dataset_filenames.py "$PROJDIR"/project/datasets/DB5/final/raw --source_type db5 --rank "$1" --size "$2"
python "$PROJDIR"/project/datasets/builder/collect_dataset_statistics.py "$PROJDIR"/project/datasets/DB5/final/raw --rank "$1" --size "$2"
python "$PROJDIR"/project/datasets/builder/log_dataset_statistics.py "$PROJDIR"/project/datasets/DB5/final/raw --rank "$1" --size "$2"
python "$PROJDIR"/project/datasets/builder/impute_missing_feature_values.py "$PROJDIR"/project/datasets/DB5/final/raw --impute_atom_features False --num_cpus 32 --rank "$1" --size "$2"

# Optionally convert each postprocessed (final 'raw') complex into a pair of DGL graphs (final 'processed') with labels
python "$PROJDIR"/project/datasets/builder/convert_complexes_to_graphs.py "$PROJDIR"/project/datasets/DB5/final/raw "$PROJDIR"/project/datasets/DB5/final/processed --num_cpus 32 --edge_dist_cutoff 15.0 --edge_limit 5000 --self_loops True --rank "$1" --size "$2"
