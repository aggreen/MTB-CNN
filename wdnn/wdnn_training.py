import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import KFold, StratifiedKFold
from keras.layers import *
from keras.layers.convolutional import *
import keras.backend as K
from keras import regularizers
from keras.layers import merge
from keras.optimizers import Adam
from keras.models import Model
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from keras.utils import multi_gpu_model
import tensorflow as tf

### ------------------ WDNN Stuff ------------------ ###
##
##
##
##

def masked_multi_weighted_bce(alpha, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
    num_not_missing = K.sum(mask, axis=-1)
    print(num_not_missing)
    alpha = K.abs(alpha)
    bce = - alpha * y_true_ * K.log(y_pred) - (1.0 - alpha) * (1.0 - y_true_) * K.log(1.0 - y_pred)
    masked_bce = bce * mask
    return K.sum(masked_bce, axis=-1) / num_not_missing

def masked_weighted_accuracy(alpha, y_pred):
    total = K.sum(K.cast(K.not_equal(alpha, 0.), K.floatx()))
    y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
    correct = K.sum(K.cast(K.equal(y_true_, K.round(y_pred)), K.floatx()) * mask)
    return correct / total


'''
Wide and deep multi-task neural network.
'''
def get_wide_deep():
    input = Input(shape=(222,))
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    wide_deep = concatenate([input, x])
    preds = Dense(11, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-8))(wide_deep)
    model = Model(input=input, output=preds)
    opt = Adam(lr=np.exp(-1.0 * 9))
    model.compile(optimizer=opt,
                  loss=masked_multi_weighted_bce,
                  metrics=[masked_weighted_accuracy])
    return model


# Used to get "retroactive" threshold from each fold's validation set to (average and)
# apply to independent test set.
def get_threshold_val(y_true, y_pred):
    num_samples = y_pred.shape[0]
    fpr_ = []
    tpr_ = []
    thresholds = np.linspace(0,1,101)
    num_sensitive = np.sum(y_true)
    num_resistant = num_samples - num_sensitive
    for threshold in thresholds:
        fp_ = 0
        tp_ = 0
        for i in range(num_samples):
            if (y_pred[i] < threshold):
                if (y_true[i] == 1): fp_ += 1
                if (y_true[i] == 0): tp_ += 1
        fpr_.append(fp_ / float(num_sensitive))
        tpr_.append(tp_ / float(num_resistant))
    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)
    #valid_inds = np.where(fpr_ <= 0.1)
    valid_inds = np.arange(101)
    sens_spec_sum = (1 - fpr_) + tpr_
    best_sens_spec_sum = np.max(sens_spec_sum[valid_inds])
    best_inds = np.where(best_sens_spec_sum == sens_spec_sum[valid_inds])
    if best_inds[0].shape[0] == 1:
        best_sens_spec_ind = best_inds
    else:
        best_sens_spec_ind = np.array(np.squeeze(best_inds))[-1]
    return {'threshold':np.squeeze(thresholds[valid_inds][best_sens_spec_ind]),
            'spec':1 - fpr_[valid_inds][best_sens_spec_ind],
            'sens':tpr_[valid_inds][best_sens_spec_ind]}

### ------------------ START ANALYSIS ------------------ ###
##
##
##
##

# Get data from pickle
df_geno_pheno_wdnn = pd.read_pickle('data_wdnn/df_geno_pheno_wdnn_072919.pkl')

# Replace NAs with 1, per discussion about alternate allele frequencies
df_geno_pheno_wdnn.fillna(value=1, inplace=True)

# Replace all values greater than 1 with a 1
df_geno_pheno_wdnn.iloc[:,0:222] = df_geno_pheno_wdnn.iloc[:,0:222].mask(df_geno_pheno_wdnn.iloc[:,0:222] > 1, 1)

# Drugs
drugs = ['RIFAMPICIN', 'ISONIAZID', 'PYRAZINAMIDE',
			'ETHAMBUTOL', 'STREPTOMYCIN', 'CIPROFLOXACIN',
			'CAPREOMYCIN', 'AMIKACIN', 'MOXIFLOXACIN',
			'OFLOXACIN', 'KANAMYCIN']
num_drugs = 11

# Get features
X_wdnn_full = df_geno_pheno_wdnn.values[:,0:222].astype(np.float)

# Change phenotype data to 0s and 1s
drugs_list = df_geno_pheno_wdnn.columns.tolist()[-23:]
y_all_rs = df_geno_pheno_wdnn[drugs_list]
resistance_categories = {'R':0, 'S':1, -1:-1}
y_all = y_all_rs.copy(deep=True)
for key, val in resistance_categories.items():
	y_all[y_all_rs == key] = val

# Get phenotype data for WDNN
y_wdnn_full = y_all[drugs].values.astype(np.int)

# Get isolates that have at least 1 resistance status to the 11 drugs
ind_with_phenotype = np.where(y_wdnn_full.sum(axis=1) != -11)
X_wdnn = X_wdnn_full[ind_with_phenotype]
y_wdnn = y_wdnn_full[ind_with_phenotype]

# Get isolates strain IDs to use for other analysis
isolate_ids = df_geno_pheno_wdnn.iloc[ind_with_phenotype].index.values
np.savetxt("data_wdnn/isolate_IDs.csv", isolate_ids, fmt="%s", delimiter="",
	comments="#")

# Get alpha matrix
alphas = np.zeros(num_drugs, dtype=np.float)
alpha_matrix = np.zeros_like(y_wdnn, dtype=np.float)

for drug in range(num_drugs):
    resistant = len(np.squeeze(np.where(y_wdnn[:,drug] == 0.)))
    sensitive = len(np.squeeze(np.where(y_wdnn[:,drug] == 1.)))
    alphas[drug] = resistant / float(resistant + sensitive)
    alpha_matrix[:,drug][np.where(y_wdnn[:,drug] == 1.0)] = alphas[drug]
    alpha_matrix[:,drug][np.where(y_wdnn[:,drug] == 0.0)] = - alphas[drug]


column_names = ['Algorithm','Drug','AUC','AUC_PR',"threshold","spec","sens"]
results = pd.DataFrame(columns=column_names)
results_index = 0
cv_splits = 5

cross_val_split = KFold(n_splits=cv_splits, shuffle=True)
for train, val in cross_val_split.split(X_wdnn):
    X_train = X_wdnn[train]
    X_val = X_wdnn[val]
    y_train = y_wdnn[train]
    y_val = y_wdnn[val]
    #------- Train the wide and deep neural network ------#
    wdnn = get_wide_deep()
    wdnn.fit(X_train, alpha_matrix[train], epochs=100, validation_data=[X_val,alpha_matrix[val]])
    wdnn_probs = wdnn.predict(X_val)
    for i, drug in enumerate(drugs):
        non_missing_val = np.where(y_val[:,i] != -1)[0]
        auc_y = np.reshape(y_val[non_missing_val,i],(len(non_missing_val), 1))
        auc_preds = np.reshape(wdnn_probs[non_missing_val,i],(len(non_missing_val), 1))
        val_auc = roc_auc_score(auc_y, auc_preds)
        val_auc_pr = average_precision_score(1-y_val[non_missing_val,i], 1-wdnn_probs[non_missing_val,i])
        val__ = get_threshold_val(y_val[:,i][non_missing_val], wdnn_probs[:,i][non_missing_val])
        val_threshold = val__["threshold"]
        val_spec = val__['spec']
        val_sens = val__['sens']
        results.loc[results_index] = ['WDNN',drug,val_auc,val_auc_pr,val_threshold, val_spec, val_sens]
        #print (drug + '\t' + str(val_auc) + '\t' + str(val_auc_pr))
        results_index += 1


results.to_csv('results_wdnn/wdnn_threshold.csv',index=False)

### ------------------ ANALYSIS: BRANCH 1: GET THRESHOLDS ------------------ ###
##
##
##
##

# Get thresholds for each drug for Maha, fully trained model on all 10k isolates
# Thresholds determined by cross-validation on validation hold-out set
results["threshold"] = results["threshold"].astype(np.float64)
results["spec"] = results["spec"].astype(np.float64)
results["sens"] = results["sens"].astype(np.float64)
results.groupby("Drug").mean().to_csv("results_wdnn/wdnn_summary.csv")

### ------------------ TRAIN FULL MODEL ON ALL DATA ------------------ ###
##
##
##
##

wdnn = get_wide_deep()
wdnn.fit(X_wdnn, alpha_matrix, epochs=100)

# Save model
wdnn_model_json = wdnn.to_json()
with open("results_wdnn/wdnn_model.json", "w") as json_file:
    json_file.write(wdnn_model_json)

# Save weights
wdnn.save_weights("results_wdnn/wdnn_weights.h5")
