#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J FFNN_baseline
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o FFNN_BS2_256-32_lr1e-2.out
#BSUB -e FFNN_BS2_256-32_lr1e-2.err

module load python3/3.11.4
module load h5py
python3 "CSL_FFNN_BS2.py"