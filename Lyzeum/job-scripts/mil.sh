#!/bin/bash
#! Which partition (queue) should be used
#SBATCH -p gpu-long
#! Number of required nodes
#SBATCH -N 1
#! Number of MPI ranks running per node
#SBATCH --ntasks-per-node=6
#! Number of GPUs per node if required
#SBATCH --gres=gpu:1
#!How much wallclock time will be required (HH:MM:SS)
#SBATCH --time=72:00:00

module purge
module load miniconda3
source activate jdenv

patch_dir="/nfs/st01/hpc-cmih-cbs31/jd949/train_patches/patches_256_128"
output_dir="/nfs/st01/hpc-cmih-cbs31/jd949/output-data/"

echo "$output_dir"

# python ../scripts/train_mil.py $patch_dir --epochs 10 --cv-folds 5 --save-dir "$output_dir"

python ../scripts/train_mil.py $patch_dir --epochs 10 --valid-split 0.0 --save-dir "$output_dir"
