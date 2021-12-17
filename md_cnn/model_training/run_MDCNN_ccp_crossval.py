#!/usr/bin/env python
# coding: utf-8
"""
Runs multitask model with conv-conv-pool architecture, 5 fold cross validation on training/validation set
This is the architecture used for the final MD-CNN model

Authors:
	Michael Chen (original version)
	Anna G. Green
	Chang Ho Yoon
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
        """
        Define convolutional neural network architecture

        NB filter_size is a global variable (int) given by the kwargs
        """

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
        """
        Class for handling CNN functionality

        """
        def __init__(self):
            self.model = get_conv_nn()
            self.epochs = N_epochs

        def fit_model(self, X_train, y_train, X_val=None, y_val=None):
            """
            X_train: np.ndarray
                n_strains x 5 (one-hot) x longest locus length x no. of loci
                Genotypes of isolates used for training
            y_train: np.ndarray
                Labels for isolates used for training

            X_val: np.ndarray (optional, default=None)
                Optional genotypes of isolates in validation set

            y_val: np.ndarray (optional, default=None)
                Optional labels for isolates in validation set

            Returns
            -------
            pd.DataFrame:
                training history (accuracy, loss, validation accuracy, and validation loss) per epoch

            """
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

    _, input_file = sys.argv

    # load kwargs from config file (input_file)
    kwargs = yaml.safe_load(open(input_file, "r"))
    print(kwargs)
    output_path = kwargs["output_path"]
    N_epochs = kwargs["N_epochs"]
    filter_size = kwargs["filter_size"]
    pkl_file = kwargs["pkl_file"]

    # Determine whether pickle already exists
    if os.path.isfile(pkl_file):
        print("pickle file already exists, proceeding with modeling")
    else:
        print("creating genotype phenotype pickle")
        make_geno_pheno_pkl(**kwargs)

    # # Get data from pickle
    df_geno_pheno = pd.read_pickle(pkl_file)
    print("read in the pkl")

    pkl_file_sparse_train = kwargs['pkl_file_sparse_train']
    pkl_file_sparse_test = kwargs['pkl_file_sparse_test']
    #
    if os.path.isfile(pkl_file_sparse_train) and os.path.isfile(pkl_file_sparse_test):
        print("X input already exists, loading X")
        X_sparse_train = sparse.load_npz(pkl_file_sparse_train)
        #X_sparse_test = sparse.load_npz(pkl_file_sparse_test)

    else:
        print("creating X pickle")
        X_all = create_X(df_geno_pheno)

        X_sparse = sparse.COO(X_all)

        X_all = X_sparse.todense()
        assert (X_all.shape[0] == len(df_geno_pheno))

        df_geno_pheno = df_geno_pheno.reset_index(drop=True)

        train_indices = df_geno_pheno.query("category=='set1_original_10202'").index
        test_indices = df_geno_pheno.query("category!='set1_original_10202'").index

        print("splitting X pkl")
        X_sparse_train = X_sparse[train_indices, :]
        X_sparse_test = X_sparse[test_indices, :]
        del X_sparse

        #X_sparse_train = sparse.COO(X_train)
        sparse.save_npz(pkl_file_sparse_train, X_sparse_train, compressed=False)

        #X_sparse_test = sparse.COO(X_test)
        sparse.save_npz(pkl_file_sparse_test, X_sparse_test, compressed=False)

    drugs = ['RIFAMPICIN', 'ISONIAZID', 'PYRAZINAMIDE',
             'ETHAMBUTOL', 'STREPTOMYCIN', 'LEVOFLOXACIN',
             'CAPREOMYCIN', 'AMIKACIN', 'MOXIFLOXACIN',
             'OFLOXACIN', 'KANAMYCIN', 'ETHIONAMIDE',
             'CIPROFLOXACIN']

    y_all_train, y_array = rs_encoding_to_numeric(df_geno_pheno.query("category=='set1_original_10202'"), drugs)

    num_drugs = len(drugs)

    # obtain phenotype data for CNN
    y_all_train = y_all_train[drugs].values.astype(np.int)

    # obtain isolates with at least 1 resistance status to length of drugs
    ind_with_phenotype = np.where(y_all_train.sum(axis=1) != -num_drugs)

    X = X_sparse_train[ind_with_phenotype]
    print("the shape of X is {}".format(X.shape))

    y = y_all_train[ind_with_phenotype]
    print("the shape of y is {}".format(y.shape))

    alpha_matrix_path = kwargs["alpha_file"]
    alpha_matrix = load_alpha_matrix(alpha_matrix_path, y_array, df_geno_pheno)
    del df_geno_pheno


    ### Perform 5-fold cross validation
    cv_splits = 5

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=1)

    column_names = ['Algorithm', 'Drug', 'AUC', 'AUC_PR', "threshold", "spec", "sens"]
    results = pd.DataFrame(columns=column_names)
    i = 0

    for train_idx, (train, val) in enumerate(cv.split(X, y)):
        model = myCNN()
        X_train = X[train, :].todense()
        X_val = X[val, :].todense()
        y_train = y[train, :]
        y_val = y[val, :]

        print('fitting..')
        history = model.fit_model(X_train, alpha_matrix[train, :], X_val, alpha_matrix[val, :])
        history.to_csv(output_path + "history_cv_split" + str(train_idx) + ".csv")
        print('predicting..')
        y_pred = model.predict(X_val)

        for idx, drug in enumerate(drugs):
            non_missing_val = np.where(y_val[:, idx] != -1)[0]
            auc_y = np.reshape(y_val[non_missing_val, idx], (len(non_missing_val), 1))
            auc_preds = np.reshape(y_pred[non_missing_val, idx], (len(non_missing_val), 1))
            val_auc = roc_auc_score(auc_y, auc_preds)
            val_auc_pr = average_precision_score(1 - y_val[non_missing_val, idx], 1 - y_pred[non_missing_val, idx])
            val__ = get_threshold_val(y_val[non_missing_val, idx], y_pred[non_missing_val, idx])
            val_threshold = float(val__["threshold"])
            val_spec = val__['spec']
            val_sens = val__['sens']

            results.loc[i] = ['CNN', drug, val_auc, val_auc_pr, val_threshold, val_spec, val_sens]

            i += 1

    K.clear_session()

    results.to_csv(output_path + "_auc.csv")


run()
