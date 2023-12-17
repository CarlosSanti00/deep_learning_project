#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J VAE_to_AE
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o autoencoder.out
#BSUB -e autoencoder.err

module load python3/3.11.4
module load h5py
python3 "VAE_to_AE.py"
