# MTB-CNN
CNN models for antibiotic resistance prediction in M. tuberculosis genomes. From Green &amp; Yoon et al, Nat Comms 2022

https://doi.org/10.1038/s41467-022-31236-0

## Installation

Software can be installed by downloading the codebase. Required dependencies are listed on a per-subdirectory basis in environment_reqs.txt. Installation of the codebase and all required dependencies takes minutes.

## System requirements

The MD-CNN, SD-CNN, and WDNN were trained on an NVIDIA GeForce GTX Titan X GPU. Model training takes 6-12 hours. 

The LogReg + L2 training was performed on a CPU with 50GB RAM.

DeepLIFT calculations were performed on an NVIDIA GeForce GTX Titan X GPU.

Downstream data analysis was performed on a MacBook Pro laptop running macOS Catalina v. 10.15.7

## Instructions for running

#### Input sequences

All sequences input to the model were aligned to the M. tuberculosis H37Rv reference genome, with fasta nucleotide sequences extracted for the positions listed in Table 3 of the published manuscript. The sequences for each locus are input as a multiple nucleotide sequence alignment - meaning that gaps are present in the input reference sequence if any of the input sequences has an insertion at that position. All provided input fasta sequence files are aligned accordingly.

To input new sequences to the model, the new sequences need to be appended to the existing input fasta sequence files so that the frame of the alignment is unchanged. We recommend doing this with MAFFT v7.490, `mafft --add <new_fasta_alignment_file> --keeplength <previous_fasta_alignment_file> > <output_alignment>`

#### Executing models

All scripts in the model_training subdirectories are executed with `python <script.py> <parameter_file.txt>`, with parameter files in yaml format. Paths within the parameter file should be adjusted for your local environment. 

## List of subdirectories
```
data_analysis: contains jupyter notebooks for comparing models and running analysis on DeepLIFT importance scores

md_cnn: code for running multidrug convolutional neural network (MD-CNN) model and corresponding DeepLIFT analysis

output_data: output accuracy files created by running the models on train and test data, list of known resistance variants, output DeepLIFT scores

sd_cnn: code for running single drug convolutional neural network (SD-CNN) models and corresponding DeepLIFT analysis

regression_l2: code for running regression + L2 regularization baseline

wdnn: code for running previous state of art wide and deep neural network (WDNN)

input_data: input fasta and phenotype data for training, test and CRyPTIC datasets, as well as in silico mutagenized strains

```

