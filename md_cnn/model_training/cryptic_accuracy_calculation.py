#!/usr/bin/env python
# coding: utf-8
"""
Runs multitask model with conv-conv-pool architecture:
- training on entire train set
- accuracy evaluation on held-out test set
This is the architecture used for the final MD-CNN model

Authors:
	Michael Chen (original version)
	Anna G. Green
	Chang-Ho Yoon
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
from tb_cnn_codebase_cryptic import *

drugs = ['RIFAMPICIN', 'ISONIAZID',
             'ETHAMBUTOL', 'LEVOFLOXACIN',
             'AMIKACIN', 'MOXIFLOXACIN',
             'KANAMYCIN', 'ETHIONAMIDE',
             ]
num_drugs = len(drugs)

original_drugs =  ['RIFAMPICIN', 'ISONIAZID', 'PYRAZINAMIDE',
             'ETHAMBUTOL', 'STREPTOMYCIN', 'LEVOFLOXACIN',
             'CAPREOMYCIN', 'AMIKACIN', 'MOXIFLOXACIN',
             'OFLOXACIN', 'KANAMYCIN', 'ETHIONAMIDE',
             'CIPROFLOXACIN']

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

        def save(self, saved_model_path):
            return self.model.save(saved_model_path)

## Compute the performance for the training set
    def compute_drug_auc_table(y_train, y_train_pred, drug_to_threshold):
        """
        Computes the AUC, sensitivity, specificity, for given threshold

        Parameters
        ----------
        y_train: np.array
            actual values for y
        y_train_pred: np.array
            predicted values for y
        drug_to_threshold: dict of str->float
            The prediction threshold for each drug
        Returns
        -------
        pd.DataFrame with columns: 'Algorithm', 'Drug', "num_sensitive", "num_resistant",'AUC', "threshold", "spec", "sens"
        """
        column_names = ['Algorithm', 'Drug', "num_sensitive", "num_resistant",'AUC', "threshold", "spec", "sens"]
        results = pd.DataFrame(columns=column_names)

        for idx, drug in enumerate(original_drugs):
            if not drug in drugs:
                continue

            new_idx = drugs.index(drug)

            print("evaluating for the test set,", drug, idx, new_idx)

            # Calculate the threshold from the TRAINING data, not the test data
            val_threshold = float(drug_to_threshold[drug])

            non_missing_val = np.where(y_train[:, new_idx] != -1)[0]
            # If we don't have any phenotypes, we can't assess
            if len(non_missing_val)==0:
                results.loc[idx] = ['MD-CNN', drug, 0, 0, np.nan, val_threshold, np.nan, np.nan]
                continue

            # Input data must be indexed with new_idx, output data must be indexed with idx
            # (Model still predicts for all drugs)
            auc_y = np.reshape(y_train[non_missing_val, new_idx], (len(non_missing_val), 1)).astype(int)
            auc_preds = np.reshape(y_train_pred[non_missing_val, idx], (len(non_missing_val), 1))
            print(set(list(auc_y.reshape(1,-1)[0])), auc_preds)
            num_sensitive = np.sum(auc_y==1)
            num_resistant = np.sum(auc_y==0)

            # If we don't have at least 1 R and 1 S isolate we can't assess model
            if num_sensitive==0 or num_resistant==0:
                results.loc[idx] = ['MD-CNN', drug, num_sensitive, num_resistant, np.nan, val_threshold, np.nan, np.nan]
                continue

            # Compute the AUC
            val_auc = roc_auc_score(auc_y, auc_preds)

            # Binarize the predicted values
            binary_prediction = np.array(y_train_pred[non_missing_val, idx] > val_threshold).astype(int)

            # Remember that in RS encoding to numeric, resistant==0
            # Specificity = #TN / # Condition Negative
            val_spec = np.sum(np.logical_and(binary_prediction == 1, y_train[non_missing_val, new_idx] == 1)) / num_sensitive

            # Sensitivity = #TP / # Condition Positive, Here defining "positive" as resistant, or 0
            val_sens = np.sum(np.logical_and(binary_prediction == 0, y_train[non_missing_val, new_idx] == 0)) / num_resistant

            results.loc[idx] = ['MD-CNN', drug, num_sensitive, num_resistant, val_auc,  val_threshold, val_spec, val_sens]

        return results

    ### Prepare the test data - held out strains
    def compute_y_pred(df_geno_pheno_test):
        """
        Computes predicted phenotypes
        """

        # Get the numeric encoding for current subset of isolates
        df_geno_pheno_test = df_geno_pheno_test.fillna('-1')
        y_all_test, y_all_test_array = rs_encoding_to_numeric(df_geno_pheno_test, drugs)

        # Make sure that we have phenotype for at least one drug
        ind_with_phenotype_test = y_all_test.index[y_all_test.sum(axis=1) != -num_drugs] + int(df_geno_pheno_test.index[0])
        ind_with_phenotype_test_0index = y_all_test.index[y_all_test.sum(axis=1) != -num_drugs]

        # Get x indices for which we have phenotype
        X = X_sparse_test[ind_with_phenotype_test]
        print("the shape of X_test is {}".format(X.shape))

        y_test = y_all_test_array[ind_with_phenotype_test_0index]
        del y_all_test_array
        del y_all_test

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

    # Determine whether pickle already exists
    if os.path.isfile(kwargs["pkl_file"]):
        print("pickle file already exists, proceeding with modeling")
    else:
        print("creating genotype phenotype pickle")
        make_geno_pheno_pkl(**kwargs)

    df_geno_pheno = pd.read_pickle(kwargs["pkl_file"])
    y_all_train, y_array = rs_encoding_to_numeric(df_geno_pheno, drugs)


    if os.path.isfile(pkl_file_sparse):
        print("X input already exists, loading X")
        X_sparse = sparse.load_npz(pkl_file_sparse)

    else:
        print("creating X pickle")
        X_all = create_X(df_geno_pheno)
        X_sparse = sparse.COO(X_all)
        sparse.save_npz(pkl_file_sparse, X_sparse, compressed=False)

    # ### obtain isolates with at least 1 resistance status to length of drugs
    ind_with_phenotype = np.where(y_all_train.sum(axis=1) != -num_drugs)

    X = X_sparse[ind_with_phenotype]
    print("the shape of X is {}".format(X.shape))

    y = y_array[ind_with_phenotype]
    print("the shape of y is {}".format(y.shape))

    y_labels = np.array(df_geno_pheno.Isolate)[ind_with_phenotype]

    ### Train the model on the entire training set - no CV splits
    saved_model_path = kwargs['saved_model_path']

    if os.path.isdir(saved_model_path):
        model = models.load_model(saved_model_path, custom_objects={
            'masked_weighted_accuracy': masked_weighted_accuracy,
            "masked_multi_weighted_bce": masked_multi_weighted_bce
        })
    else:
        print("Did not find model", saved_model_path)
        model = myCNN()
        X_train = X.todense()
        print('fitting..')
        alpha_matrix = load_alpha_matrix(kwargs["alpha_file"], y, df_geno_pheno, **kwargs)
        history = model.fit_model(X_train, alpha_matrix)
        history.to_csv(output_path + "history.csv")
        model.save(saved_model_path)
    #
    # ## Get the thresholds for evaluation
    print("Predicting for training data...")
    y_train_pred = model.predict(X.todense())
    y_train = y_array[ind_with_phenotype]
    y_train[y_train=='I'] = -1
    np.save("predicted_y_cryptic.npy", y_train_pred)
    np.save("actual_y_cryptic.npy", y_train)
    np.save("labels_y_cryptic.npy", y_labels)

    # Select the prediction threshold for each drug based on TRAINING SET DATA
    threshold_data = pd.read_csv(kwargs["threshold_file"])
    drug_to_threshold = {x:y for x,y in zip(threshold_data.drug, threshold_data.threshold)}

    ## Compute AUC for training set data
    results = compute_drug_auc_table(y_train, y_train_pred, drug_to_threshold)
    results.to_csv(f"{output_path}_training_set_drug_auc.csv")

    prediction_df = pd.read_pickle("cryptic_geno_pheno_train_test.pkl")
    prediction_df = prediction_df[["Isolate"]]
    print(prediction_df.shape)
    print(y_train_pred.shape)
    prediction_df.loc[:,original_drugs] = y_train_pred

    prediction_df.to_csv(f"{output_path}_strain_auc.csv")

run()
