#!/bin/bash

####################### Batch Headers #########################
#SBATCH -A BIF135
#SBATCH -p batch
#SBATCH -J make_evcoupling_dataset
#SBATCH -t 0-24:00
#SBATCH --mem 224G
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
###############################################################

# Remote paths #
export PROJDIR=/gpfs/alpine/scratch/"$USER"/bif135/Repositories/Lab_Repositories/DIPS-Plus
export PSAIADIR=/ccs/home/"$USER"/Programs/PSAIA_1.0_source/bin/linux/psa
export OMP_NUM_THREADS=8

# Remote Conda environment #
source "$PROJDIR"/miniconda3/bin/activate
conda activate DIPS-Plus

# Load CUDA module for DGL
module load cuda/10.2.89

# Default to using the Big Fantastic Database (BFD) of protein sequences (approx. 270GB compressed)
export HHSUITE_DB=/gpfs/alpine/scratch/$USER/bif132/Data/Databases/bfd_metaclust_clu_complete_id30_c90_final_seq

# Run dataset compilation scripts
cd "$PROJDIR"/project || exit

srun python3 "$PROJDIR"/project/datasets/builder/generate_hhsuite_features.py "$PROJDIR"/project/datasets/EVCoupling/interim/parsed "$PROJDIR"/project/datasets/EVCoupling/interim/parsed "$HHSUITE_DB" "$PROJDIR"/project/datasets/EVCoupling/interim/external_feats --rank "$1" --size "$2" --num_cpu_jobs 4 --num_cpus_per_job 8 --num_iter 2 --source_type evcoupling --read_file

#srun python3 "$PROJDIR"/project/datasets/builder/postprocess_pruned_pairs.py "$PROJDIR"/project/datasets/EVCoupling/raw "$PROJDIR"/project/datasets/EVCoupling/interim/pairs "$PROJDIR"/project/datasets/EVCoupling/interim/external_feats "$PROJDIR"/project/datasets/EVCoupling/final/raw --num_cpus 32 --rank "$1" --size "$2" --source_type EVCoupling

#python3 "$PROJDIR"/project/datasets/builder/partition_dataset_filenames.py "$PROJDIR"/project/datasets/EVCoupling/final/raw --source_type EVCoupling --rank "$1" --size "$2"
#python3 "$PROJDIR"/project/datasets/builder/collect_dataset_statistics.py "$PROJDIR"/project/datasets/EVCoupling/final/raw --rank "$1" --size "$2"
#python3 "$PROJDIR"/project/datasets/builder/log_dataset_statistics.py "$PROJDIR"/project/datasets/EVCoupling/final/raw --rank "$1" --size "$2"
#python3 "$PROJDIR"/project/datasets/builder/impute_missing_feature_values.py "$PROJDIR"/project/datasets/EVCoupling/final/raw --impute_atom_features False --num_cpus 32 --rank "$1" --size "$2"

# Optionally convert each postprocessed (final 'raw') complex into a pair of DGL graphs (final 'processed') with labels
#python3 "$PROJDIR"/project/datasets/builder/convert_complexes_to_graphs.py "$PROJDIR"/project/datasets/EVCoupling/final/raw "$PROJDIR"/project/datasets/EVCoupling/final/processed --num_cpus 32 --edge_dist_cutoff 15.0 --edge_limit 5000 --self_loops True --rank "$1" --size "$2"
