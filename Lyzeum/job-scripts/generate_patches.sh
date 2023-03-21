#!/bin/bash
#! Which partition (queue) should be used
#SBATCH -p skylake
#! Number of required nodes
#SBATCH -N 1
#! Number of MPI ranks running per node
#SBATCH --ntasks-per-node=12
#! Number of GPUs per node if required
#SBATCH --gres=gpu:0
#!How much wallclock time will be required (HH:MM:SS)
#SBATCH --time=12:00:00

module purge
module load miniconda3
source activate jdenv

patch_dir="/nfs/st01/hpc-cmih-cbs31/jd949/train_patches/patches_256_128"
output_dir="/nfs/st01/hpc-cmih-cbs31/jd949/output-data/"

python ../scripts/extract_wsi_patches.py $spreadsheet $wsi_dir $out_dir
