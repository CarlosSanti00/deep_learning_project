#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J VAE_Ana
#BSUB -n 1
#BSUB -W 05:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o VAE_Ana.out
#BSUB -e VAE_Ana.err

module load python3/3.11.4
module load h5py
python3 "ANA_VAE_metrics.py"
