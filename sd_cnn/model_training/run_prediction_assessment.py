#!/usr/bin/env python
# coding: utf-8
"""
Code for running CNN on MTB data to predict ABR phenotypes
Authors:
	Michael Chen (original version)
	Anna G. Green
	Chang-ho Yoon
"""

import sys
import glob
import os
#import ruamel.yaml as yaml
import yaml
import sparse
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tb_cnn_codebase import *
from sklearn.metrics import roc_auc_score, average_precision_score


def run():

    def get_conv_nn():

		#TODO: replace X.shape with passed argument
        model = models.Sequential()
		#TODO: add filter size argument
        model.add(layers.Conv2D(
            64, (5, filter_size),
            data_format='channels_last',
            activation='relu',
            input_shape = X.shape[1::]
        ))
        model.add(layers.Conv2D(64, (1,12), activation='relu', name='conv1d'))
        model.add(layers.MaxPooling2D((1,3), name='max_pooling1d'))
        model.add(layers.Conv2D(32, (1,3), activation='relu', name='conv1d_1'))
        model.add(layers.Conv2D(32, (1,3), activation='relu', name='conv1d_2'))
        model.add(layers.MaxPooling2D((1,3), name='max_pooling1d_1'))
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(256, activation='relu', name='d1'))
        model.add(layers.Dense(256, activation='relu', name='d2'))
        model.add(layers.Dense(1, activation='sigmoid', name='d4'))

        print(model.summary())

        opt = Adam(lr=np.exp(-1.0 * 9))

        model.compile(optimizer=opt,
                      loss=masked_multi_weighted_bce,
                      metrics=[masked_weighted_accuracy])

        return model

    class myCNN:
        def __init__(self):
            self.model = get_conv_nn()
            self.epochs = N_epochs

        def fit_model(self, X_train, y_train, X_val=None, y_val=None, weights=[1.,1.]):
            if X_val is not None and y_val is not None:
                history = self.model.fit(
                    X_train, y_train,
                    epochs=self.epochs,
                    validation_data=(X_val, y_val),
                    batch_size=128
                )
                print('\nhistory dict:', history.history)
                return pd.DataFrame.from_dict(data=history.history)
            else:
                history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=128)
                print('\nhistory dict:', history.history)
                return pd.DataFrame.from_dict(data=history.history)

        def predict(self, X_val):
            return np.squeeze(self.model.predict(X_val))

        def save(self, saved_model_path):
            return self.model.save(saved_model_path)

    _, input_file = sys.argv

    # load kwargs from config file (input_file)
    kwargs = yaml.safe_load(open(input_file, "r"))
    print(kwargs)
    output_path = kwargs["output_path"]
    N_epochs = kwargs["N_epochs"]
    filter_size = kwargs["filter_size"]
    pkl_file = kwargs["pkl_file"]
    DRUG = kwargs["drug"]
    locus_list = kwargs["locus_list"]

    # # Get data from pickle
    df_geno_pheno = pd.read_pickle(pkl_file)
    print("read in the pkl")

    pkl_file_sparse = kwargs['pkl_file_sparse_train']

    #
    if os.path.isfile(pkl_file_sparse) and False:
        print("X input already exists, loading X")
        X_sparse_train = sparse.load_npz(pkl_file_sparse)
        X_sparse_test = sparse.load_npz(kwargs["pkl_file_sparse_test"])
        X_sparse = np.concat([X_sparse_test, X_sparse_train])
        print("Total X shape", X_sparse.shape)

    else:
        print("creating X pickle")

        df_geno_pheno["Isolate"] = df_geno_pheno.index
        columns_to_keep = ["index","Isolate", "category", DRUG] + [x+"_one_hot" for x in locus_list]

        df_geno_pheno_subset = df_geno_pheno[columns_to_keep]
        del df_geno_pheno
        print(df_geno_pheno_subset)
        df_geno_pheno_subset = df_geno_pheno_subset.loc[
            np.logical_or(df_geno_pheno_subset[DRUG]=='R',df_geno_pheno_subset[DRUG]=="S")
        ]
        print(df_geno_pheno_subset.shape)
        df_geno_pheno_subset.to_csv(output_path + "_df_geno_pheno.csv")
        X_all = create_X(df_geno_pheno_subset)

        X_sparse = sparse.COO(X_all)
        sparse.save_npz("X_all.pkl", X_sparse, compressed=False)
        X_sparse = sparse.load_npz("X_all.pkl.npz")

        sparse.save_npz(pkl_file_sparse, X_sparse, compressed=False)

    num_drugs = 1

    ### Train the model on the entire training set - no CV splits
    saved_model_path = kwargs['saved_model_path']

    if os.path.isfile(saved_model_path):
        model = models.load_model(saved_model_path, custom_objects={
            'masked_weighted_accuracy': masked_weighted_accuracy,
            "masked_multi_weighted_bce": masked_multi_weighted_bce
        })
    else:
        print("NO MODELLLLL WHY", saved_model_path)


    threshold_data = pd.read_csv(kwargs['threshold_file'])
    print(threshold_data)
    drug_to_threshold = {x:y for x,y in zip([DRUG], threshold_data.threshold)}

    # Quantitative prediction for X data
    y_prediction = model.predict(X_sparse.todense())
    print(X_sparse.shape, y_prediction.shape)
    print("predictions complete")

    prediction_df = pd.read_csv(output_path + "_df_geno_pheno.csv")
    prediction_df = prediction_df[["Isolate"]]
    print(prediction_df.shape)

    print(y_prediction.shape)
    prediction_df.loc[:,DRUG] = y_prediction

    prediction_df.to_csv(f"{output_path}_strain_auc.csv")

run()
