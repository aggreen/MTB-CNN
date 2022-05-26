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
num_drugs = 1

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
        model.add(layers.Dense(1, activation='sigmoid', name='d4'))

        opt = Adam(learning_rate=np.exp(-1.0 * 9))

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

    ## Compute the performance for the training set
    def compute_drug_auc_table(y_train, y_train_pred, threshold):
        column_names = ['Algorithm', 'Drug', "num_sensitive", "num_resistant",'AUC', 'AUC_PR', "threshold", "spec", "sens"]
        results = pd.DataFrame(columns=column_names)

        for idx, drug in enumerate([DRUG]):
            print("evaluating for the test set,", drug)

            # Calculate the threshold from the TRAINING data, not the test data
            val_threshold = float(threshold)

            non_missing_val = np.where(y_train[:, idx] != -1)[0]

            if len(non_missing_val)==0:
                results.loc[idx] = ['CNN', drug, 0, 0, np.nan, np.nan, val_threshold, np.nan, np.nan]
                continue

            auc_y = np.reshape(y_train[non_missing_val, idx], (len(non_missing_val), 1)).astype(int)
            auc_preds = np.reshape(y_train_pred[non_missing_val, idx], (len(non_missing_val), 1))

            num_sensitive = np.sum(auc_y==1)
            num_resistant = np.sum(auc_y==0)

            if num_sensitive==0 or num_resistant==0:
                results.loc[idx] = ['CNN', drug, num_sensitive, num_resistant, np.nan, np.nan, val_threshold, np.nan, np.nan]
                continue

            val_auc = roc_auc_score(auc_y, auc_preds)

            y_average_precision = (1 - y_train[non_missing_val, idx]).astype(int)
            pred_average_precision = (1 - y_train_pred[non_missing_val, idx]).astype(int)
            val_auc_pr = average_precision_score(y_average_precision, pred_average_precision)

            binary_prediction = np.array(y_train_pred[non_missing_val, idx] > val_threshold).astype(int)

            #print("N positive", N_condition_positive, "N negative", N_condition_negative)
            # Specificity = #TN / # Condition Negative
            # Remember that in RS encoding to numeric, resistant==0
            val_spec = np.sum(np.logical_and(binary_prediction == 1, y_train[non_missing_val, idx] == 1)) / num_sensitive
            # Sensitivity = #TP / # Condition Positive
            val_sens = np.sum(np.logical_and(binary_prediction == 0, y_train[non_missing_val, idx] == 0)) / num_resistant

            #print(['CNN', drug, num_sensitive, num_resistant, val_auc, val_auc_pr, val_threshold, val_spec, val_sens])
            results.loc[idx] = ['CNN', drug, num_sensitive, num_resistant, val_auc, val_auc_pr, val_threshold, val_spec, val_sens]

        return results


    ### Prepare the test data - held out strains
    def compute_y_pred(df_geno_pheno_test):

        df_geno_pheno_test = df_geno_pheno_test.fillna('-1')
        y_all_test, y_all_test_array = rs_encoding_to_numeric(df_geno_pheno_test, DRUG)
        #y_all_test = y_all_test.reshape(-1,1)
        y_all_test_array = y_all_test_array.reshape(-1,1)

        ind_with_phenotype_test = y_all_test.index[y_all_test != -1] + int(df_geno_pheno_test.index[0])
        ind_with_phenotype_test_0index = y_all_test.index[y_all_test != -1]

        X = X_sparse_test[ind_with_phenotype_test]
        print("the shape of X_test is {}".format(X.shape))

        # y_train_val = y_all_train_val[ind_with_phenotype_train_val]
        y_test = y_all_test_array[ind_with_phenotype_test_0index]
        del y_all_test_array
        del y_all_test
        # print("the shape of y_train_val is {}".format(y_train_val.shape))
        print("the shape of y_test is {}".format(y_test.shape))

        print('Predicting for test data...')
        y_pred = model.predict(X.todense())

        return y_pred, y_test


    _, input_file = sys.argv

    # load kwargs from config file (input_file)
    kwargs = yaml.safe_load(open(input_file, "r"))
    print(kwargs)

    output_path = kwargs["output_path"]
    N_epochs = kwargs["N_epochs"]
    filter_size = kwargs["filter_size"]
    pkl_file_sparse = kwargs['pkl_file_sparse']
    DRUG = kwargs["drug"]
    locus_list = kwargs["locus_list"]

    # Ensure that output directory exists
    output_dir = "/".join(output_path.split("/")[0:-1])
    if not os.path.isdir(output_dir):
        print(f"WARNING: output directory {output_dir} does not exist, creating")
        os.system(f"mkdir {output_dir}")

    # Determine whether pickle already exists
    if os.path.isfile(kwargs["pkl_file"]):
        print("pickle file already exists, proceeding with modeling")
    else:
        print("creating genotype phenotype pickle")
        make_geno_pheno_pkl(**kwargs)

    df_geno_pheno = pd.read_pickle(kwargs["pkl_file"])
    y_all_train, y_array = rs_encoding_to_numeric(df_geno_pheno, DRUG)


    if os.path.isfile(pkl_file_sparse):
        print("X input already exists, loading X")
        X_sparse = sparse.load_npz(pkl_file_sparse)


    else:
        print("creating X pickle")
        columns_to_keep = ["Isolate", DRUG] + [x+"_one_hot" for x in locus_list]
        print(list(df_geno_pheno.columns))
        df_geno_pheno_subset = df_geno_pheno[columns_to_keep]
        del df_geno_pheno
        print(df_geno_pheno_subset)
        df_geno_pheno_subset = df_geno_pheno_subset.loc[
            np.logical_or(df_geno_pheno_subset[DRUG]=='R',df_geno_pheno_subset[DRUG]=="S")
        ]
        print(df_geno_pheno_subset.shape)

        X_all = create_X(df_geno_pheno_subset)
        df_geno_pheno_subset.to_csv(output_path + "_df_geno_pheno.csv")

        X_sparse = sparse.COO(X_all)
        sparse.save_npz(pkl_file_sparse, X_sparse, compressed=False)

    # Read in isolates
    df_geno_pheno = pd.read_csv(output_path + "_df_geno_pheno.csv", index_col=0)
    y_all_train, y_array = rs_encoding_to_numeric(df_geno_pheno, DRUG)
    y_array = y_array.reshape(-1,1)
    num_drugs = 1

    # obtain phenotype data for CNN
    y_all_train = y_all_train.values.astype(np.int)

    # obtain isolates with at least 1 resistance status to length of drugs
    ind_with_phenotype = np.arange(0, len(y_all_train))
    print(X_sparse.shape, y_array.shape)

    X = X_sparse[ind_with_phenotype]
    print("the shape of X is {}".format(X.shape))

    y = y_all_train[ind_with_phenotype].reshape(-1,1)
    print("the shape of y is {}".format(y.shape))

    alpha_matrix_path = kwargs["alpha_file"]

    ### Train the model on the entire training set - no CV splits
    saved_model_path = kwargs['saved_model_path']

    if os.path.isfile(saved_model_path):
        model = models.load_model(saved_model_path, custom_objects={
            'masked_weighted_accuracy': masked_weighted_accuracy,
            "masked_multi_weighted_bce": masked_multi_weighted_bce
        })
    else:
        print("Did not find model", saved_model_path)
    #
    # ## Get the thresholds for evaluation
    print("Predicting for training data...")
    y_train_pred = model.predict(X.todense())
    y_train = y_array[ind_with_phenotype]

    print(y_train.shape)
    #
    # # Get the optimal prediction threshold
    # val__ = get_threshold_val(y_train.reshape(-1,1), y_train_pred.reshape(-1,1))
    # threshold_data = pd.DataFrame(val__, index=[0])
    # threshold_data.to_csv(kwargs['threshold_file'])

    threshold_data = pd.read_csv(kwargs['threshold_file'])

    threshold = threshold_data.threshold[0]

    results = compute_drug_auc_table(y_train, y_train_pred, threshold)
    results.to_csv(f"{output_path}_drug_auc.csv")

    prediction_df = pd.read_csv(output_path + "_df_geno_pheno.csv")
    prediction_df = prediction_df[["Isolate"]]
    print(prediction_df.shape)
    print(y_train_pred.shape)
    prediction_df.loc[:,DRUG] = y_train_pred

    prediction_df.to_csv(f"{output_path}_strain_auc.csv")


run()
