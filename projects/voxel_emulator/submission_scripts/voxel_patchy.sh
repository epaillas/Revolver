#!/bin/bash
#SBATCH --account=desi_g
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus 4
#SBATCH --array=0-10

N_PHASE=50
START_PHASE=$((SLURM_ARRAY_TASK_ID * N_PHASE + 1))
WEIGHT_TYPE=default_FKP
ZLIM="0.45 0.6"
NTHREADS=4

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=$NTHREADS
export NUMEXPR_MAX_THREADS=$NTHREADS
CODE_PATH=/global/u1/e/epaillas/code/Revolver/projects/voxel_emulator/voxel_patchy.py

# srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi -c 256 \
python $CODE_PATH \
--start_phase $START_PHASE \
--n_phase $N_PHASE \
--weight_type $WEIGHT_TYPE \
--zlim $ZLIM \
--nthreads $NTHREADS \
--save_voids \
--use_gpu \