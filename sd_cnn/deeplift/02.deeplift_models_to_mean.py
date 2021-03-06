'''
Postprocessing on deeplift importance scores
Author: Anna G. Green

'''
from __future__ import print_function
import warnings
import glob
import sparse
import numpy as np
import pandas as pd

# Read in phenotype data
df_geno_pheno = pd.read_pickle("../focus_cnn/multitask_geno_pheno_train_test.pkl")
df_geno_pheno = df_geno_pheno.query("category=='set1_original_10202'")
df_geno_pheno = df_geno_pheno.reset_index(drop=True)

print(len(df_geno_pheno))
# Get lis of all output files
output_files = glob.glob("output/*npz")

# For each file (each drug)
for file in output_files:

    # sparse load
    scores = sparse.load_npz(file)
    drug = file.split("/")[-1].split("_")[0]

    # get only the resistant strains
    print(file, drug, scores.shape)
    subset_df_geno_pheno = df_geno_pheno.loc[np.logical_or(df_geno_pheno[drug]=="R", df_geno_pheno[drug]=="S"),:].reset_index(drop=True)
    resistant_strains = subset_df_geno_pheno.loc[df_geno_pheno[drug]=="R",:].index

    if len(resistant_strains)==0:
        continue

    assert len(subset_df_geno_pheno) == scores.shape[0]

    # take only isolates that are resistant
    scores_subset = scores[resistant_strains, :, :].todense()
    print("shape of the scores", scores_subset.shape)

    # Take max, median, and mean of saliency at each position
    max_score = np.max(np.abs(scores_subset), axis=0)
    median_score = np.median(scores_subset, axis=0)
    mean_score = np.mean(scores_subset, axis=0)

    # save to file
    np.save(f"output/{drug}_max.npy", max_score)
    np.save(f"output/{drug}_median.npy", median_score)
    np.save(f"output/{drug}_mean.npy", mean_score)
