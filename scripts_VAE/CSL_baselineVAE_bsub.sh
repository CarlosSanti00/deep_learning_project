#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Baseline_VAE
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o ../Log_out_files/Baseline_VAE.out
#BSUB -e ../Log_out_files/Baseline_VAE.err

module load python3/3.11.4
module load h5py
python3 "CSL_Baseline_VAE.py"
