#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J VAE_Ana_try
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o ../Log_out_files/dist.out
#BSUB -e ../Log_out_files/dist.err

module load python3/3.11.4
module load h5py
python3 "distributions.py"
