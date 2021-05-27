#!/bin/bash

# Where the project is stored
export PROJID=bif132
export PROJDIR=/gpfs/alpine/scratch/"$USER"/$PROJID/Repositories/Lab_Repositories/DIPS-Plus

# Which copies of the BFD to use for the available nodes' search batches (i.e. in range ['_1', '_2', ..., '_31', '_32'])
BFD_COPY_IDS=("_1" "_2" "_3" "_4" "_5" "_6" "_7" "_8" "_9" "_10" "_11" "_12" "_17" "_21" "_25" "_29")
NUM_BFD_COPIES=${#BFD_COPY_IDS[@]}

# Whether to launch multi-node compilation jobs for DIPS or instead for DB5
compile_dips=true

if [ "$compile_dips" = true ]; then
  # Job 1 - DIPS
  echo Submitting job 1 for compile_dips_dataset_on_andes.sh with parameters: 0 "$NUM_BFD_COPIES"
  sbatch "$PROJDIR"/project/datasets/builder/compile_dips_dataset_on_andes.sh 0 "$NUM_BFD_COPIES"

  # Job 2 - DIPS
  echo Submitting job 2 for compile_dips_dataset_on_andes.sh with parameters: 1 "$NUM_BFD_COPIES"
  sbatch "$PROJDIR"/project/datasets/builder/compile_dips_dataset_on_andes.sh 1 "$NUM_BFD_COPIES"

  # Job 3 - DIPS
  echo Submitting job 3 for compile_dips_dataset_on_andes.sh with parameters: 2 "$NUM_BFD_COPIES"
  sbatch "$PROJDIR"/project/datasets/builder/compile_dips_dataset_on_andes.sh 2 "$NUM_BFD_COPIES"

  # Job 4 - DIPS
  echo Submitting job 4 for compile_dips_dataset_on_andes.sh with parameters: 3 "$NUM_BFD_COPIES"
  sbatch "$PROJDIR"/project/datasets/builder/compile_dips_dataset_on_andes.sh 3 "$NUM_BFD_COPIES"
else
  # Job 1 - DB5
  echo Submitting job 1 for compile_db5_dataset_on_andes.sh with parameters: 0 "$NUM_BFD_COPIES"
  sbatch "$PROJDIR"/project/datasets/builder/compile_db5_dataset_on_andes.sh 0 "$NUM_BFD_COPIES"

  # Job 2 - DB5
  echo Submitting job 2 for compile_db5_dataset_on_andes.sh with parameters: 1 "$NUM_BFD_COPIES"
  sbatch "$PROJDIR"/project/datasets/builder/compile_db5_dataset_on_andes.sh 1 "$NUM_BFD_COPIES"

  # Job 3 - DB5
  echo Submitting job 3 for compile_db5_dataset_on_andes.sh with parameters: 2 "$NUM_BFD_COPIES"
  sbatch "$PROJDIR"/project/datasets/builder/compile_db5_dataset_on_andes.sh 2 "$NUM_BFD_COPIES"

  # Job 4 - DB5
  echo Submitting job 4 for compile_db5_dataset_on_andes.sh with parameters: 3 "$NUM_BFD_COPIES"
  sbatch "$PROJDIR"/project/datasets/builder/compile_db5_dataset_on_andes.sh 3 "$NUM_BFD_COPIES"
fi
