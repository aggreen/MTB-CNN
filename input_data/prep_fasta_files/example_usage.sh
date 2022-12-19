#!/bin/bash
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=30G
#SBATCH -o errors_%j.out
#SBATCH -e errors_%j.err


srun perl snpConcatenater_w_exclusion_frompilonvcf_2.9_edit_20201119.pl exclude.BED IDfail.tab INDEL REGION 1672457-1675011 pos > FabG1-inhA.fasta
