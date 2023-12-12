#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J FFNN_all
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o FFNN_final.out
#BSUB -e FFNN_final.err

module load python3/3.11.4
module load h5py
python3 "FFNN_Final.py"
