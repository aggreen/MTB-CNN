
import sys
import glob
import os
import yaml
import sparse
import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd
from tb_cnn_codebase import *

geno_df = make_genotype_df("fasta_files/")
df_genos = geno_df.query('index =="MT_H37Rv"')

print('making one hot encoding for...')
for column in df_genos.columns:
    print("...", column)
    df_genos[column + "_one_hot"] = df_genos[column].apply(np.vectorize(get_one_hot))

X = create_X(df_genos)

print(X.shape)

np.save( "h37rv_input.npy", X)
