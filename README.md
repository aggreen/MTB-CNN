# MTB-CNN
CNN models for antibiotic resistance prediction in M. tuberculosis genomes. From Green &amp; Yoon et al 

preprint: https://www.biorxiv.org/content/10.1101/2021.12.06.471431v1


## List of subdirectories
```
data_analysis: contains jupyter notebooks for comparing models and running analysis on DeepLIFT importance scores

md_cnn: code for running multidrug convolutional neural network (MD-CNN) model and corresponding DeepLIFT analysis

output_data: output accuracy files created by running the models on train and test data, list of known resistance variants, output DeepLIFT scores

sd_cnn: code for running single drug convolutional neural network (SD-CNN) models and corresponding DeepLIFT analysis

regression_l2: code for running regression + L2 regularization baseline

wdnn: code for running previous state of art wide and deep neural network (WDNN)

```

All subdirectories contain an environment_reqs.txt file with the list of packages in the conda environment (this may vary between subdirectories)
