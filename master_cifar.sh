#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 18            # number of cores 
#SBATCH -t 0-06:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH --gpus=a100:1
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

python experiment_configs/cifar10/ensembles/train_ensemble_members.py
python experiment_configs/cifar10/ensembles/eval_ensembles.py

python experiment_configs/cifar10/one_dimensional_subspaces/train_lines.py
python experiment_configs/cifar10/one_dimensional_subspaces/train_lines_layerwise.py
python experiment_configs/cifar10/one_dimensional_subspaces/train_curves.py

python experiment_configs/cifar10/one_dimensional_subspaces/eval_lines.py
python experiment_configs/cifar10/one_dimensional_subspaces/eval_lines_layerwise.py
python experiment_configs/cifar10/one_dimensional_subspaces/eval_curves.py