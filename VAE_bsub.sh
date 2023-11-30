#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J VAE_23/11/30
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o VAE_231130.out
#BSUB -e VAE_231130.err

module load python3/3.11.4
module load h5py
python3 "231113_VAE.py"