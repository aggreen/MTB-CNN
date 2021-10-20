#!/usr/bin/env python
# coding: utf-8
"""
Runs SD-CNN training on cross-validation set
- assesses accuracy for each CV split
- saves table of loss per epoch

Authors:
	Michael Chen (original version)
	Anna G. Green
	Chang-ho Yoon
"""

import sys
import glob
import os
import yaml
import sparse
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models

from tb_cnn_codebase import *

num_drugs = 1

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
        model.add(layers.Dense(1, activation='sigmoid', name='d4'))

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
            """
            Returns
            -------
            predicted labels for given X
            """
            return np.squeeze(self.model.predict(X_val))

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

    # Ensure that output directory exists
    output_dir = "/".join(output_path.split("/")[0:-1])
    if not os.path.isdir(output_dir):
        print(f"WARNING: output directory {output_dir} does not exist, creating")
        os.system(f"mkdir {output_dir}")

    # Determine whether the genotype/phenotype pickle already exists
    if os.path.isfile(pkl_file):
        print("pickle file already exists, proceeding with modeling")
    else:
        print("creating genotype phenotype pickle")
        make_geno_pheno_pkl(**kwargs)

    # Get data from pickle
    df_geno_pheno = pd.read_pickle(pkl_file)
    print("read in the pkl")

    # Create numpy arrays for training and test sets
    pkl_file_sparse_train = kwargs['pkl_file_sparse_train']
    pkl_file_sparse_test = kwargs['pkl_file_sparse_test']

    # If already exists, continue
    if os.path.isfile(pkl_file_sparse_train) and os.path.isfile(pkl_file_sparse_test) and os.path.isfile(output_path + "_df_geno_pheno.csv"):
        print("X input already exists, loading training data")
        X_sparse_train = sparse.load_npz(pkl_file_sparse_train)


    else:
        print("creating X pickle")

        # Discard columns not relevant to study
        columns_to_keep = ["index", "category", DRUG] + [x+"_one_hot" for x in locus_list]
        df_geno_pheno_subset = df_geno_pheno[columns_to_keep]
        del df_geno_pheno

        # Drop isolates that don't have  measured phenotype for drug of interest
        df_geno_pheno_subset = df_geno_pheno_subset.loc[
            np.logical_or(df_geno_pheno_subset[DRUG]=='R',df_geno_pheno_subset[DRUG]=="S")
        ]
        print(df_geno_pheno_subset.shape)
        df_geno_pheno = df_geno_pheno_subset.reset_index(drop=True)
        df_geno_pheno.to_csv(output_path + "_df_geno_pheno.csv")

        # Create the X matrix
        X_all = create_X(df_geno_pheno_subset)
        X_sparse = sparse.COO(X_all)

        # Get the training and test indices
        train_indices = df_geno_pheno.query("category=='set1_original_10202'").index
        test_indices = df_geno_pheno.query("category!='set1_original_10202'").index

        print("splitting X pkl")
        X_sparse_train = X_sparse[train_indices, :]
        X_sparse_test = X_sparse[test_indices, :]
        print("training set shape", X_sparse_train.shape)
        print("test set shape", X_sparse_test.shape)
        del X_sparse

        # Save the sparsified outputs
        sparse.save_npz(pkl_file_sparse_train, X_sparse_train, compressed=False)
        sparse.save_npz(pkl_file_sparse_test, X_sparse_test, compressed=False)

    ## get the y values for isolates in the training set ('set1_original_10202')
    df_geno_pheno = pd.read_csv(output_path + "_df_geno_pheno.csv", index_col=0)
    y_all_train, y_array = rs_encoding_to_numeric(df_geno_pheno.query("category=='set1_original_10202'"), DRUG)
    y_array = y_array.reshape(-1,1)
    y_all_train = y_all_train.values.astype(np.int)

    # obtain isolates with at least 1 resistance status to length of drugs
    ind_with_phenotype = np.arange(0, len(y_all_train))

    X = X_sparse_train[ind_with_phenotype]
    print("the shape of X is {}".format(X.shape))

    y = y_all_train[ind_with_phenotype].reshape(-1,1)
    print("the shape of y is {}".format(y.shape))

    # Read in or create the alpha matrix
    alpha_matrix_path = kwargs["alpha_file"]
    if os.path.isfile(alpha_matrix_path):
        print("alpha matrix already exists, loading alpha matrix")
        alpha_matrix = alpha_matrix = np.loadtxt(alpha_matrix_path, delimiter=',')
    else:
        print("creating alpha matrix")
        if "weight_of_sensitive_class" in kwargs:
            print('creating alpha matrix with weight', kwargs["weight_of_sensitive_class"])
            alpha_matrix = alpha_mat(y_array, df_geno_pheno, kwargs["weight_of_sensitive_class"])
        else:
            alpha_matrix = alpha_mat(y_array, df_geno_pheno)
        np.savetxt(alpha_matrix_path, alpha_matrix, delimiter=',')
    del df_geno_pheno
    print("the shape of the alpha_matrix: {}".format(alpha_matrix.shape))
    alpha_matrix = alpha_matrix.reshape(-1,1)

    ###### Training the model
    cv_splits = 5
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=1)

    column_names = ['Algorithm', 'Drug', 'AUC', 'AUC_PR', "threshold", "spec", "sens"]
    results = pd.DataFrame(columns=column_names)
    i = 0

    for train_idx, (train, val) in enumerate(cv.split(X, y)):

        # Initialize model, split train and validation
        model = myCNN()
        X_train = X[train, :].todense()
        X_val = X[val, :].todense()
        y_train = y[train, :]
        y_val = y[val, :]

        # Fit model and save history
        print('fitting..')
        history = model.fit_model(X_train, alpha_matrix[train, :], X_val, alpha_matrix[val, :])
        history.to_csv(output_path + "history_cv_split" + str(train_idx) + ".csv")
        print('predicting..')
        y_pred = model.predict(X_val)

        # Compute statistics on validation set
        auc_y = y_val.reshape(-1,1)
        auc_preds = y_pred.reshape(-1,1)
        val_auc = roc_auc_score(auc_y, auc_preds)
        val_auc_pr = average_precision_score(1 - y_val, 1 - y_pred)
        val__ = get_threshold_val(y_val, y_pred)
        val_threshold = float(val__["threshold"])
        val_spec = val__['spec']
        val_sens = val__['sens']

        # Save for current iteration
        results.loc[i] = ['CNN', DRUG, val_auc, val_auc_pr, val_threshold, val_spec, val_sens]
        i += 1

    K.clear_session()
    results.to_csv(output_path + "_auc.csv")

run()
