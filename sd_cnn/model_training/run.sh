#!/bin/bash
#SBATCH -c 8                               # Request one core
#SBATCH -t 0-03:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_beam                           # Partition to run in
#SBATCH --mem=400G                         # Memory total in MB (for all cores)
#SBATCH -o hostname_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e hostname_%j.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --gres=gpu:1                                           # You can change the filenames given with -o and -e to any filenames you'd like
 
# You can change hostname to any command you would like to run

module load gcc/6.2.0 cuda/10.1
python $1 $2

