#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J VAE_to_AE 
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o FFNN_100.out
#BSUB -e FFNN_100.err

module load python3/3.11.4
module load h5py
python3 "FFNN_VAE_to_AE.py"
