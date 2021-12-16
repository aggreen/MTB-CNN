# MTB-CNN
CNN models for antibiotic resistance prediction in M. tuberculosis genomes. From Green &amp; Yoon et al 

preprint: https://www.biorxiv.org/content/10.1101/2021.12.06.471431v1

## Installation

Software can be installed by downloading the codebase. Required dependencies are listed on a per-subdirectory basis in environment_reqs.txt. Installation of the codebase and all required dependencies takes minutes.

## System requirements

The MD-CNN, SD-CNN, and WDNN were trained on an NVIDIA GeForce GTX Titan X GPU. Model training takes 6-12 hours. 

The LogReg + L2 training was performed on a CPU with 50GB RAM.

DeepLIFT calculations were performed on an NVIDIA GeForce GTX Titan X GPU.

Downstream data analysis was performed on a MacBook Pro laptop running macOS Catalina v. 10.15.7

## Instructions for running



## List of subdirectories
```
data_analysis: contains jupyter notebooks for comparing models and running analysis on DeepLIFT importance scores

md_cnn: code for running multidrug convolutional neural network (MD-CNN) model and corresponding DeepLIFT analysis

output_data: output accuracy files created by running the models on train and test data, list of known resistance variants, output DeepLIFT scores

sd_cnn: code for running single drug convolutional neural network (SD-CNN) models and corresponding DeepLIFT analysis

regression_l2: code for running regression + L2 regularization baseline

wdnn: code for running previous state of art wide and deep neural network (WDNN)

```

