#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J ReadData_archs4
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o ReadData_archs4_231112.out
#BSUB -e ReadData_archs4_231112.err

module load python3/3.6.2
python3 "231112_read_archs4.py"
