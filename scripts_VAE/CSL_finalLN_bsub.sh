#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J VAE_final
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o ../Log_out_files/VAE_final.out
#BSUB -e ../Log_out_files/VAE_final.err

module load python3/3.11.4
module load h5py
python3 "CSL_Final_LN_VAE.py"
