#!/bin/bash
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -c 256
#SBATCH --array=0-8

N_PHASE=10
START_PHASE=$((SLURM_ARRAY_TASK_ID * N_PHASE + 1))
WEIGHT_TYPE=default
ZLIM="0.45 0.6"
NTHREADS=256

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=$NTHREADS
export NUMEXPR_MAX_THREADS=$NTHREADS
export OMP_PLACES=threads
CODE_PATH=/global/u1/e/epaillas/code/Revolver/projects/voxel_emulator/voxel_nseries.py

# srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi -c 256 \
python $CODE_PATH \
--start_phase $START_PHASE \
--n_phase $N_PHASE \
--weight_type $WEIGHT_TYPE \
--zlim $ZLIM \
--nthreads $NTHREADS \
--save_voids \