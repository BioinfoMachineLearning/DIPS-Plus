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
