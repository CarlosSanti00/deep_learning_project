#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J baseline_VAE_2
#BSUB -n 1
#BSUB -W 04:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o ../Log_out_files/baseline.out
#BSUB -e ../Log_out_files/baseline.err

module load python3/3.11.4
module load h5py
python3 "baseline.py"
