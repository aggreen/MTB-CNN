import sys
import os
import joblib
import pandas as pd
import numpy as np

from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score


def get_threshold_val(y_true, y_pred):
    """
    Compute the optimal threshold for prediction  based on the max sum of specificity and Sensitivity

    NB that we encoded R as 0, S as 1, so smaller predictions indicate higher chance of resistance

    We count falsely predicted resistance as a false positive, falsely predicted sensitivity as a false negative

    Parameters
    ----------
    y_true: np.array
        Actual labels for isolates
    y_pred: np.array
        Predicted labels for isolates

    Returns
    -------
    dict of str -> float with entries:
        sens: sensitivity at chosen threshold
        spec: specificity at chosen threshold
        threshold: chosen threshold value
    """

    # Compute number resistant and sensitive
    num_samples = y_pred.shape[0]
    num_sensitive = np.sum(y_true==1)
    num_resistant = np.sum(y_true==0)

    # Test thresholds from 0.01 to 0.99
    thresholds = np.linspace(0, 1, 101)

    fpr_ = []
    tpr_ = []

    for threshold in thresholds:

        fp_ = 0 # number of false positives
        tp_ = 0 # number of true positives

        for i in range(num_samples):
            # If y is predicted resistant
            if (y_pred[i] < threshold):
                # If actually sensitive, false positive
                if (y_true[i] == 1): fp_ += 1
                # If actually resistant, true positive
                if (y_true[i] == 0): tp_ += 1

        # Compute FPR and TPR
        fpr_.append(fp_ / float(num_sensitive))
        tpr_.append(tp_ / float(num_resistant))

    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)

    valid_inds = np.arange(101)
    # Sensitivity = TPR, Specificity = 1-FPR
    sens_spec_sum = (1 - fpr_) + tpr_

    # get index of highest sum(s) of sens and spec
    best_sens_spec_sum = np.max(sens_spec_sum[valid_inds])
    best_inds = np.where(best_sens_spec_sum == sens_spec_sum[valid_inds])

    # Determine if one or multiple best
    if best_inds[0].shape[0] == 1:
        best_sens_spec_ind = np.array(np.squeeze(best_inds))
    else:
        # If multiple best, take the last one (arbitrary)
        best_sens_spec_ind = np.array(np.squeeze(best_inds))[-1]

    return {'threshold': np.squeeze(thresholds[valid_inds][best_sens_spec_ind]),
            'spec': 1 - fpr_[valid_inds][best_sens_spec_ind],
            'sens': tpr_[valid_inds][best_sens_spec_ind]}

# Input argument is drug
drug = sys.argv[1]

# Read in the genotypes of interest
genotypes = pd.read_csv("site_indices.csv", index_col=0)
genotype_columns = [f"{x}_{y}" for x,y in zip(genotypes.locus, genotypes.sites)]

### Prepare the input data
train_df = pd.read_csv("combined_geno_pheno_df.csv", index_col=0)
test_df = train_df.query("category!='set1_original_10202'")
train_df = train_df.query("category=='set1_original_10202'")
for_fitting=train_df.dropna(subset=[drug])
X = for_fitting[genotype_columns]
Y = for_fitting[drug]
print(X.shape, Y.shape)

print(for_fitting.groupby(drug).size())

### Fit the GridSearchCV model to choose best C
parameters = {"C": [0.0001, 0.001, 0.01, 0.1, 1.]}
classifier = LogisticRegression(
    max_iter=1000,
    penalty='l2',
    class_weight="balanced"
)
clf = GridSearchCV(classifier, parameters)
print("fitting")
clf.fit(X, Y)

# Prepare and save output locations
output_dir = "20210830"
os.system("mkdir "+output_dir)
os.system(f"mkdir {output_dir}/{drug}")
joblib.dump(clf, f"{output_dir}/{drug}/GridSearchCV.model")


### Run cross validation using the best C, assess accuracy, sensitivity, specificity
kf = KFold(n_splits=5)
data = []
for train_index, test_index in kf.split(X.values):

    classifier = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000, **clf.best_params_)
    classifier.fit(X.values[train_index,:],Y.values[train_index])

    y_pred = classifier.predict_proba(X.values[test_index])[:,1]
    y_true = Y.values[test_index]

    cutoffs = get_threshold_val(y_true, y_pred)

    val_auc = roc_auc_score(y_true, y_pred)
    print(val_auc, cutoffs)

    data.append([drug, val_auc, cutoffs['spec'], cutoffs['sens'], cutoffs['threshold']])

df = pd.DataFrame(data, columns=["drug", "AUC", "spec", "sens", "threshold"])
df.to_csv(f"{output_dir}/{drug}/XVal_accuracy.csv")

### Fit LogisticRegression on best C, then assess on held-out set
print("chosen parameters", clf.best_params_)
classifer = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000, **clf.best_params_)
classifier.fit(X,Y)

joblib.dump(classifier, f"{output_dir}/{drug}/LogisticRegression_bestC.model")
test_df=test_df.dropna(subset=[drug])
print(test_df.shape)

X = test_df[genotype_columns]
Y = test_df[drug]

y_pred = classifier.predict_proba(X.values)[:,1]
y_true = Y.values

cutoffs = get_threshold_val(y_true, y_pred)

val_auc = roc_auc_score(y_true, y_pred)

# Save data
data=[drug, len(y_true), sum(y_true==1), sum(y_true==0), val_auc, cutoffs['spec'], cutoffs['sens'], cutoffs['threshold']]
df = pd.DataFrame([data], columns=["drug", "N", "N_S", "N_R", "AUC", "spec", "sens", "threshold"])
df.to_csv(f"{output_dir}/{drug}/test_set_accuracy.csv")
