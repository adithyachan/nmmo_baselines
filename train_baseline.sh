#!/bin/bash

# Usage: sbatch train_simple.sh experiment_name --arg1=value1 --arg2=value2 ...


#SBATCH --comment=carperai
#SBATCH --partition=g40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=40G
#SBATCH --chdir=/fsx/home-daveey/nmmo-baselines/
#SBATCH --output=sbatch/%j.out
#SBATCH --error=sbatch/%j.error
#SBATCH --requeue
#SBATCH --export=PYTHONUNBUFFERED=1
job_name=$3
#SBATCH --job-name="$job_name"

source /fsx/home-daveey/miniconda3/etc/profile.d/conda.sh && \
conda activate nmmo && \
ulimit -c unlimited && \
ulimit -s unlimited && \
ulimit -a && \
stdbuf -oL -eL python -u -O -m main \
--rollout.num_envs=4 \
--rollout.num_buffers=4  \
--rollout.num_steps=32 \
--wandb.entity=daveey \
--wandb.project=nmmo \
--train.experiments_dir=/fsx/home-daveey/experiments \
--train.num_steps=100000000 \
--train.experiment_name="$1" \
"${@:3}"
