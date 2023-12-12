#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J VAE_EPOCHS
#BSUB -n 1
#BSUB -W 2:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o 2EPOCHS.out
#BSUB -e 2EPOCHS.err

module load python3/3.11.4
module load h5py
python3 "VAE_lognormal_Final_wRELU.py"
