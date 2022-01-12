#! /bin/bash
#SBATCH -J batch_run
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH -t 500:00
#SBATCH --gres=gpu:2

python ./batch_run.py