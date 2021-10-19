#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-08:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=200G                         # Memory total in MB (for all cores)
#SBATCH -o hostname_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e hostname_%j.err                 # File to which STDERR will be written, including job ID (%j)

# You can change hostname to any command you would like to run

source activate dl_tf1
python 01.compute_importance_per_drug.py $1
