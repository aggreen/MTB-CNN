#!/usr/bin/env python
# coding: utf-8
"""
Code to train MDCNN on entire training set and save model using TENSORFLOW 1
for Deeplift compatibility

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
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tb_cnn_codebase_tf1 import *
from sklearn.metrics import roc_auc_score, average_precision_score

drugs = ['RIFAMPICIN', 'ISONIAZID', 'PYRAZINAMIDE',
             'ETHAMBUTOL', 'STREPTOMYCIN', 'LEVOFLOXACIN',
             'CAPREOMYCIN', 'AMIKACIN', 'MOXIFLOXACIN',
             'OFLOXACIN', 'KANAMYCIN', 'ETHIONAMIDE',
             'CIPROFLOXACIN']
num_drugs = len(drugs)

def run():

    def get_conv_nn():

		#TODO: replace X.shape with passed argument
        model = models.Sequential()
		#TODO: add filter size argument
        model.add(layers.Conv2D(
            64, (4, filter_size),
            data_format='channels_last',
            activation='relu',
            input_shape = (4, 10291, 18)
        ))
        model.add(layers.Conv2D(64, (1,12), activation='relu', name='conv1d'))
        model.add(layers.MaxPooling2D((1,3), name='max_pooling1d'))
        model.add(layers.Conv2D(32, (1,3), activation='relu', name='conv1d_1'))
        model.add(layers.Conv2D(32, (1,3), activation='relu', name='conv1d_2'))
        model.add(layers.MaxPooling2D((1,3), name='max_pooling1d_1'))
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(256, activation='relu', name='d1'))
        model.add(layers.Dense(256, activation='relu', name='d2'))
        model.add(layers.Dense(13, activation='sigmoid', name='d4'))

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

        def save(self, saved_model_path):
            return self.model.save(saved_model_path)

        def save_weights(self, saved_weights_path):
            return self.model.save_weights(saved_weights_path)

    ## Compute the performance for the training set
    def compute_drug_auc_table(y_train, y_train_pred, drug_to_threshold):
        column_names = ['Algorithm', 'Drug', "num_sensitive", "num_resistant",'AUC', 'AUC_PR', "threshold", "spec", "sens"]
        results = pd.DataFrame(columns=column_names)

        for idx, drug in enumerate(drugs):
            print("evaluating for the test set,", drug)

            # Calculate the threshold from the TRAINING data, not the test data
            val_threshold = float(drug_to_threshold[drug])

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
            N_condition_negative = np.sum(y_train[non_missing_val, idx] == 0)
            N_condition_positive = np.sum(y_train[non_missing_val, idx] == 1)
            print("N positive", N_condition_positive, "N negative", N_condition_negative)
            # Specificity = #TN / # Condition Negative
            # Remember that in RS encoding to numeric, resistant==0
            val_spec = np.sum(np.logical_and(binary_prediction == 0, y_train[non_missing_val, idx] == 0)) / N_condition_negative
            # Sensitivity = #TP / # Condition Positive
            val_sens = np.sum(np.logical_and(binary_prediction == 1, y_train[non_missing_val, idx] == 1)) / N_condition_positive
            #print(['CNN', drug, num_sensitive, num_resistant, val_auc, val_auc_pr, val_threshold, val_spec, val_sens])
            results.loc[idx] = ['CNN', drug, num_sensitive, num_resistant, val_auc, val_auc_pr, val_threshold, val_spec, val_sens]

        return results


    ### Prepare the test data - held out strains
    def compute_y_pred(df_geno_pheno_test):

        df_geno_pheno_test = df_geno_pheno_test.fillna('-1')
        y_all_test, y_all_test_array = rs_encoding_to_numeric(df_geno_pheno_test, drugs)

        ind_with_phenotype_test = y_all_test.index[y_all_test.sum(axis=1) != -num_drugs] + int(df_geno_pheno_test.index[0])
        ind_with_phenotype_test_0index = y_all_test.index[y_all_test.sum(axis=1) != -num_drugs]

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
    pkl_file_sparse_train = kwargs['pkl_file_sparse_train']
    pkl_file_sparse_test = kwargs['pkl_file_sparse_test']

    df_geno_pheno = pd.read_pickle(kwargs["pkl_file"])
    y_all_train, y_array = rs_encoding_to_numeric(df_geno_pheno.query("category=='set1_original_10202'"), drugs)

    #
    if os.path.isfile(pkl_file_sparse_train) and os.path.isfile(pkl_file_sparse_test):
        print("X input already exists, loading X")
        X_sparse_train = sparse.load_npz(pkl_file_sparse_train)
        X_sparse_test = sparse.load_npz(pkl_file_sparse_test)

    else:
        print("creating X pickle")
        X_all = create_X(df_geno_pheno)
        X_sparse = sparse.COO(X_all)
        del X_all
        sparse.save_npz("X_all.pkl", X_sparse, compressed=False)

        print("splitting X into train and test")
        X_sparse = sparse.load_npz("X_all.pkl.npz")
        X_sparse_test, X_sparse_train = split_into_traintest(X_sparse, df_geno_pheno, "set1_original_10202")

    ## obtain isolates with at least 1 resistance status to length of drugs
    ind_with_phenotype = np.where(y_all_train.sum(axis=1) != -num_drugs)

    X = X_sparse_train[ind_with_phenotype]

    # Because of Deeplift we need to remove the 'gap' column
    X = X[:, 0:4, :, :]

    print("the shape of X is {}".format(X.shape))

    y = y_array[ind_with_phenotype]
    print("the shape of y is {}".format(y.shape))

    ### Train the model on the entire training set - no CV splits
    saved_model_path = kwargs['saved_model_path']

    # if os.path.isdir(saved_model_path):
    #     model = models.load_model(saved_model_path, custom_objects={
    #         'masked_weighted_accuracy': masked_weighted_accuracy,
    #         "masked_multi_weighted_bce": masked_multi_weighted_bce
    #     })
    # else:
    model = myCNN()
    X_train = X.todense()
    print('fitting..')
    alpha_matrix = load_alpha_matrix(kwargs["alpha_file"], y, df_geno_pheno, **kwargs)
    history = model.fit_model(X_train, alpha_matrix)
    history.to_csv(output_path + "history.csv")
    model.save(saved_model_path)
    #

    model.save_weights(saved_model_path + "weights.h5")
    exit()

run()
