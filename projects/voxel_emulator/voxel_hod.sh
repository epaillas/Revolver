#!/bin/bash
#SBATCH --account=desi_g
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus 4
#SBATCH --constraint=gpu
#SBATCH --array=130-181


N_HOD=100
START_HOD=0
N_COSMO=1
START_COSMO=$((SLURM_ARRAY_TASK_ID * N_COSMO))
NTHREADS=128

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=$NTHREADS
export NUMEXPR_MAX_THREADS=$NTHREADS
CODE_PATH=/global/u1/e/epaillas/code/Revolver/projects/voxel_emulator/voxel_hod.py
JOB_FLAGS="-N 1 -C gpu -t 12:00:00 --gpus 1 --qos regular --account desi_g"

srun $JOB_FLAGS python $CODE_PATH \
--start_cosmo $START_COSMO \
--n_cosmo $N_COSMO \
--start_hod $START_HOD \
--n_hod $N_HOD \
--nthreads $NTHREADS \
--save_voids \