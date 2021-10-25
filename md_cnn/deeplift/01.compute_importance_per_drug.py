'''
Computes and saves deeplift importance scores for our method
Author: Anna G. Green

Based on Google Colab notebook DeepLIFT notebook genomics tutorial.ipynb
'''
from __future__ import print_function
import sparse
import os
import sys
import warnings
import h5py
print("h5py version:", h5py.__version__)
import json
import deeplift
warnings.filterwarnings("ignore")

import tensorflow as tf
print("Tensorflow version:", tf.__version__)
from tensorflow import keras
print("Keras version:", keras.__version__)
import numpy as np
print("Numpy version:", np.__version__)
import tensorflow.keras.backend as K
import pandas as pd
import deeplift.conversion.kerasapi_conversion as kc

from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from sklearn.metrics import roc_auc_score, average_precision_score

from tensorflow.keras.models import model_from_json
from deeplift.layers import NonlinearMxtsMode
from collections import OrderedDict
from deeplift.util import compile_func

####### Section 1: Read in the Data #########
#extract onehot data from h37rv_geno

def _get_shapes(df_geno):
		"""
		Finds the length of each gene in the input dataframe
		Parameters
		----------
		df_geno_pheno: pd.Dataframe

		Returns
		-------
		dict of str: int
			length of coordinates in each column
		"""
		shapes = {}
		for column in df_geno.columns:
			if "one_hot" in column:
				shapes[column] = df_geno.loc[df_geno.index[0],column].shape[0]

		return shapes

h37rv_geno = pd.read_pickle("../../input_data/h37rv_geno.pkl")

shapes = _get_shapes(h37rv_geno)
n_genes = len(shapes)
L_longest = max(list(shapes.values()))
print("found n genes", n_genes, "and longest gene", L_longest)

# Initialize X to hold H37rv sequence in one hot encoding
X = np.zeros((1, 4, L_longest, n_genes))

# for each gene locus
for gene_index, gene in enumerate(shapes.keys()):
  one_hot_gene = h37rv_geno.loc[:, gene][0]
  # remove column of gaps/'missing' nucleotides as per DeepLift implementation
  one_hot_gene = one_hot_gene[:, :4]
  # rearrange axes to fit X
  one_hot_gene_arr = np.moveaxis(one_hot_gene, source = [0], destination = [1])
  X[:, :, range(0, one_hot_gene.shape[0]), gene_index] = one_hot_gene_arr

X_h37rv = X

####### Section 2: Prepare the model and save in json format ########
#extract onehot data from h37rv_geno

def masked_multi_weighted_bce(alpha, y_pred):

	"""
	Calculates the masked weighted binary cross-entropy in multi-classification

	Parameters
	----------
	alpha: an element from the alpha matrix, a matrix of target y values weighted
		by proportion of strains with resistance data for any given drug
	y_pred: model-predicted y values

	Returns
	-------
	scalar value of the masked weighted BCE.
	"""
	y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
	y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
	mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
	num_not_missing = K.sum(mask, axis=-1)
	alpha = K.abs(alpha)
	bce = - alpha * y_true_ * K.log(y_pred) - (1.0 - alpha) * (1.0 - y_true_) * K.log(1.0 - y_pred)
	masked_bce = bce * mask
	return K.sum(masked_bce, axis=-1) / num_not_missing

def masked_weighted_accuracy(alpha, y_pred):

	"""
	Calculates the mased weighted accuracy of a model's predictions
	Parameters
	----------
	alpha: an element from the alpha matrix, a matrix of target y values weighted
		by proportion of strains with resistance data for any given drug
	y_pred: model-predicted y values

	Returns
	-------
	scalar value of the masked weighted accuracy.
	"""

	total = K.sum(K.cast(K.not_equal(alpha, 0.), K.floatx()))
	y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
	mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
	correct = K.sum(K.cast(K.equal(y_true_, K.round(y_pred)), K.floatx()) * mask)
	return correct / total

def convertKerasJSONtoDeepLIFT(kerasJSON_str):
    jsonData = json.loads(kerasJSON_str)
    layersData = jsonData["config"]["layers"]
    jsonData["config"] = layersData
    return json.dumps(jsonData)


#create model
deeplift_trialweights = "../model_training/saved_models/ccp_deepliftweights.h5"

deeplift_model_json = "CNN_model.json"

## Load our model from the json file
our_model = model_from_json(open(deeplift_model_json).read())
our_model.load_weights(deeplift_trialweights)

# Create the deeplift models
method_to_model = OrderedDict()

with h5py.File(deeplift_trialweights) as keras_model_weights :

    for method_name, nonlinear_mxts_mode in [
        #The genomics default = rescale on conv layers, revealcance on fully-connected
        ('rescale_conv_revealcancel_fc', NonlinearMxtsMode.DeepLIFT_GenomicsDefault)]:
        method_to_model[method_name] = kc.convert_model_from_saved_files(
            h5_file=deeplift_trialweights,
            json_file=deeplift_model_json,
            nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault
        )


###### Step 5: Computing importance scores #######
print("Compiling scoring functions")
method_to_scoring_func = OrderedDict()
for method, model in method_to_model.items():
    print("Compiling scoring function for: "+method)
    method_to_scoring_func[method] = model.get_target_contribs_func(find_scores_layer_idx=0,
                                                                    target_layer_idx=-2)


# Drug
drugs = ['RIFAMPICIN', 'ISONIAZID', 'PYRAZINAMIDE',
             'ETHAMBUTOL', 'STREPTOMYCIN', 'LEVOFLOXACIN',
             'CAPREOMYCIN', 'AMIKACIN', 'MOXIFLOXACIN',
             'OFLOXACIN', 'KANAMYCIN', 'ETHIONAMIDE',
             'CIPROFLOXACIN']

# Prepare the X data

X = sparse.load_npz("../model_training/multitask_X_sparse_train.npy.npz")
X = X.todense()
X = X[:, 0:4, :, :]
X = X[:, : ,: ,:]

print("The shape of the full X is", X.shape)

print("Locus lengths for full X")
for i in range(18):
    print(i, np.max(np.argwhere(X[1,:,:,i] != 0)))

from collections import OrderedDict

for method_name, score_func in method_to_scoring_func.items():
    print("on method",method_name)

    drug = sys.argv[1]
    task_idx = drugs.index(drug)
    print("Working on task", task_idx, drug)
    scores = np.array(score_func(
                task_idx=task_idx,
                input_data_list=[X],
                input_references_list=[X_h37rv],
                batch_size=200,
                progress_update=None))
    print(scores.shape)
    #The sum over the ACGT axis in the code below is important! Recall that DeepLIFT
    # assigns contributions based on difference-from-reference; if
    # a position is [1,0,0,0] (i.e. 'A') in the actual sequence and [0.3, 0.2, 0.2, 0.3]
    # in the reference, importance will be assigned to the difference (1-0.3)
    # in the 'A' channel, (0-0.2) in the 'C' channel,
    # (0-0.2) in the G channel, and (0-0.3) in the T channel. You want to take the importance
    # on all four channels and sum them up, so that at visualization-time you can project the
    # total importance over all four channels onto the base that is actually present (i.e. the 'A'). If you
    # don't do this, your visualization will look very confusing as multiple bases will be highlighted at
    # every position and you won't know which base is the one that is actually present in the sequence!
    scores = np.sum(scores, axis=1)

    scores_sparse = sparse.COO(scores)
    del scores
    sparse.save_npz(f"output/{drug}_scores_allstrains.npy", scores_sparse, compressed=True)
    #sparse.()
