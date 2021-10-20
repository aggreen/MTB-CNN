'''
Converts our tensorflow/keras model to a deeplift model
Authors: Anna G. Green
        Chang Ho Yoon

Note: This requires tensorflow v1! The CNN model must be saved in tf1

Based on Google Colab notebook DeepLIFT notebook genomics_tutorial.ipynb
'''
from __future__ import print_function
import sparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import h5py
import json
import deeplift
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import deeplift.conversion.kerasapi_conversion as kc

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.models import model_from_json
from deeplift.layers import NonlinearMxtsMode
from collections import OrderedDict
from deeplift.util import compile_func

print("Tensorflow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
print("Numpy version:", np.__version__)
print("h5py version:", h5py.__version__)

####### Section 1: Read in the reference input data and convert to one-hot #########

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

_, input_file = sys.argv
kwargs = yaml.safe_load(open(input_file, "r"))
h37rv_geno = pd.read_pickle("../../input_data/h37rv_geno.pkl")

columns_to_keep =  [x+"_one_hot" for x in kwargs['locus_list']]
print(list(h37rv_geno.columns))
h37rv_geno = h37rv_geno[columns_to_keep]

# Prepare the reference X Matrix
shapes = _get_shapes(h37rv_geno)
n_genes = len(shapes)
L_longest = max(list(shapes.values()))
print("found n genes", n_genes, "and longest gene", L_longest)

# Initialize X to hold H37rv sequence in one hot encoding
X = np.zeros((1, 5, L_longest, n_genes))

# for each gene locus
for gene_index, gene in enumerate(shapes.keys()):
  one_hot_gene = h37rv_geno.loc[:, gene][0]

  # rearrange axes to fit X
  one_hot_gene_arr = np.moveaxis(one_hot_gene, source = [0], destination = [1])
  X[:, :, range(0, one_hot_gene.shape[0]), gene_index] = one_hot_gene_arr

X_h37rv = X
print("Shape of X is", X.shape)
####### Section 2: Prepare the model and save in json format ########
# We must convert the keras model to a deeplift model
# Deeplift model saved as a json file to be loaded in the future

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
deeplift_trialweights = kwargs["tensorflow_model_weights_file"]

deeplift_model_json = kwargs["deeplift_model_file"]
print(deeplift_trialweights, deeplift_model_json)
#
filter_size = 12

# Define the model (must be the same architecture as used for training)
print("Model input shape", X.shape[1:])

model = models.Sequential()
model.add(layers.Conv2D(
    64, (5, filter_size),
    data_format='channels_last',
    activation='relu',
    input_shape = X.shape[1:]
))
model.add(layers.Conv2D(64, (1,12), activation='relu', name='conv1d'))
model.add(layers.MaxPooling2D((1,3), name='max_pooling1d'))
model.add(layers.Conv2D(32, (1,3), activation='relu', name='conv1d_1'))
model.add(layers.Conv2D(32, (1,3), activation='relu', name='conv1d_2'))
model.add(layers.MaxPooling2D((1,3), name='max_pooling1d_1'))
model.add(layers.Flatten(name='flatten'))
model.add(layers.Dense(256, activation='relu', name='d1'))
model.add(layers.Dense(256, activation='relu', name='d2'))
model.add(layers.Dense(1, activation=None, name='d4'))

print(model.summary())

opt = Adam(lr = np.exp(-1.0 * 9))

model.compile(optimizer=opt,
              loss=masked_multi_weighted_bce,
              metrics=[masked_weighted_accuracy])

#load weights
model.load_weights(deeplift_trialweights)

#save json
model_json = model.to_json()
with open(deeplift_model_json, "w") as json_file:
  json_file.write(model_json)

print("model saved")

###### Step 4: Read in Deeplift model and define method (rules) to use for assessment#####
## Load our model from the json file
our_model = model_from_json(open(deeplift_model_json).read())
our_model.load_weights(deeplift_trialweights)

print(our_model.summary())

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


###### Step 5: sanity check make sure that our predictions match with keras and deeplift#####
###make sure predictions are the same as the original model

model_to_test = method_to_model['rescale_conv_revealcancel_fc']

deeplift_prediction_func = compile_func([model_to_test.get_layers()[0].get_activation_vars()],
                                         model_to_test.get_layers()[-1].get_activation_vars())

original_model_predictions = our_model.predict([X_h37rv], batch_size=200)

converted_model_predictions = deeplift.util.run_function_in_batches(
                                input_data_list=[X_h37rv],
                                func=deeplift_prediction_func,
                                batch_size=200,
                                progress_update=None)

print(original_model_predictions)
print(converted_model_predictions)
print("maximum difference in predictions:",np.max(np.array(converted_model_predictions)-np.array(original_model_predictions)))
assert np.max(np.array(converted_model_predictions)-np.array(original_model_predictions)) < 10**-5
