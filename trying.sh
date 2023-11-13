#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Try
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o Try0.out
#BSUB -e Try.err

module load python3/3.11.4
module load h5py
python3 "Tries.py"