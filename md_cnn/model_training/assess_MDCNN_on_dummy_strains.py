#!/usr/bin/env python
# coding: utf-8
"""
Code for using a trained MD-CNN model to predict phenotypes for "dummy" strains,
ie sequences made in silico
Authors:
	Michael Chen (original MD-CNN)
	Anna G. Green
	Chang-ho Yoon
"""

import sys
import glob
import os
import yaml
import sparse
import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models

from tb_cnn_codebase import *


def run():

    def get_conv_nn():

		#TODO: replace X.shape with passed argument
        model = models.Sequential()
		#TODO: add filter size argument
        model.add(layers.Conv2D(
            64, (5, filter_size),
            data_format='channels_last',
            activation='relu',
            input_shape = X.shape[1:]
        ))
        model.add(layers.Lambda(lambda x: K.squeeze(x, 1)))
        model.add(layers.Conv1D(64, 12, activation='relu'))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Conv1D(32, 3, activation='relu'))
        model.add(layers.Conv1D(32, 3, activation='relu'))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu', name='d1'))
        model.add(layers.Dense(256, activation='relu', name='d2'))
        model.add(layers.Dense(13, activation='sigmoid', name='d4'))

        opt = Adam(learning_rate=np.exp(-1.0 * 9))

        model.compile(optimizer=opt,
                      loss=masked_multi_weighted_bce,
                      metrics=[masked_weighted_accuracy])


        return model

    class myCNN:
        def __init__(self):
            self.model = get_conv_nn()
            self.epochs = N_epochs

        def fit_model(self, X_train, y_train, X_val=None, y_val=None):
            if X_val is not None and y_val is not None:
                history = self.model.fit(X_train, y_train, epochs=self.epochs,
                                         validation_data=(X_val, y_val), batch_size=128)
                print('\nhistory dict:', history.history)
                return pd.DataFrame.from_dict(data=history.history)
            else:
                history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=128)
                print('\nhistory dict:', history.history)
                return pd.DataFrame.from_dict(data=history.history)

        def predict(self, X_val):
            return np.squeeze(self.model.predict(X_val))

    _, input_file = sys.argv

    # load kwargs from config file (input_file)
    kwargs = yaml.safe_load(open(input_file, "r"))
    print(kwargs)

    output_path = kwargs["output_path"]
    N_epochs = kwargs["N_epochs"]
    filter_size = kwargs["filter_size"]
    pkl_file = kwargs["pkl_file"]

    # Proceed via same protocol as training model on real sequences -
    # Give model path to fasta files of dummy seqs and construct pd.DataFrame
    # and then X data
    if os.path.isfile(pkl_file):
        print("pickle file already exists, proceeding with modeling")
    else:
        print("creating genotype phenotype pickle")
        make_geno_pheno_pkl(**kwargs)

    # Get data from pickle
    df_geno_pheno = pd.read_pickle(pkl_file)

    print("read in the pkl")
    print("the shape of the geno pheno df", df_geno_pheno.shape)

    pkl_file_sparse= kwargs['pkl_file_sparse']

    # Create the X input from the pd.DataFrame
    if os.path.isfile(pkl_file_sparse):
        print("X input already exists, loading X")
        X_sparse = sparse.load_npz(pkl_file_sparse)

    else:
        print("creating X pickle")
        X_all = create_X(df_geno_pheno)

        X_sparse = sparse.COO(X_all)
        sparse.save_npz(pkl_file_sparse, X_sparse, compressed=False)

    drugs = ['RIFAMPICIN', 'ISONIAZID', 'PYRAZINAMIDE',
             'ETHAMBUTOL', 'STREPTOMYCIN', 'LEVOFLOXACIN',
             'CAPREOMYCIN', 'AMIKACIN', 'MOXIFLOXACIN',
             'OFLOXACIN', 'KANAMYCIN', 'ETHIONAMIDE',
             'CIPROFLOXACIN']


    num_drugs = len(drugs)

    ### Load in the saved model
    saved_model_path = kwargs['saved_model_path']

    if os.path.isdir(saved_model_path):
        model = models.load_model(saved_model_path, custom_objects={
            'masked_weighted_accuracy': masked_weighted_accuracy,
            "masked_multi_weighted_bce": masked_multi_weighted_bce
        })

    ## Load in the saved threshold data
    threshold_data = pd.read_csv(kwargs['threshold_file'])
    drug_to_threshold = {x:y for x,y in zip(threshold_data.drug, threshold_data.threshold)}

    # Quantitative prediction for X data
    y_prediction = model.predict(X_sparse.todense())
    print(X_sparse.shape, y_prediction.shape)
    print("predictions complete")

    # Phenotype file must be a dataframe containing the names of the dummy strains
    # in a column called Isolate
    prediction_df = pd.read_csv(kwargs['phenotype_file'])
    prediction_df = prediction_df[["Isolate"]]

    # Create an empty column for each drug
    for drug in drugs:
        prediction_df[drug] = np.nan

    # Add in the correct prediction
    prediction_df.loc[:,drugs] = y_prediction

    # Save
    prediction_df.to_csv(f"{output_path}_dummy_strain_prediction.csv")

run()
