#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J FFNN_all
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o FFNN_all.out
#BSUB -e FFNN_all.err

module load python3/3.11.4
module load h5py
python3 "CSL_FFNN_all_231204.py"