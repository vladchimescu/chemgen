#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd
import os
import sys
import itertools
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import label_binarize

import chemgen_utils as utl
import predictor as prd

""" 
Find optimal hyperparameters for
One-vs-Rest RandomForest classifier
"""

# parameter grid for random forest classifier
# here 9 * 3 * 8 * 8 * 11 = 19008 combinations of hyperparameters
RF_grid = {"max_depth": list(range(2,10)) + [None],
           "n_estimators": [200, 500, 1000],
           "min_samples_split": range(2,10),
           "min_samples_leaf": range(2,10),
           "class_weight": [{0: 1, 1: i} for i in range(1,10)] + ['balanced', 'balanced_subsample']
           }

# XGB has 20 * 10 * 4 * 3 * 10 = 24000 combinations of hyperparameters
xgb_grid = {"learning_rate": np.logspace(-2,np.log10(0.9), 20),
            "colsample_bytree": np.linspace(0.1,1,10),
            "max_depth": range(4,11,2),
            "n_estimators": [200, 500, 1000],
            "scale_pos_weight": range(1,11)
            }
# function for generating train / validation splits
# by choosing randomly 15 drugs and withholding all pairwise combinations
def generate_train_val(drugs, combs, n_holdout=15):
    val_drugs = np.random.choice(drugs, size=n_holdout)
    
    combs_val = list(itertools.combinations(val_drugs, 2))
    combs_val = [sorted(i) for i in combs_val]
    combs_val = np.array([i[0]+"_"+i[1] for i in combs_val])
    combs_val = np.intersect1d(combs_val, combs)
    combs_train = np.setdiff1d(combs, combs_val)
    
    assert((combs_train.shape[0] + combs_val.shape[0]) == combs.shape[0])
    train = np.where(np.isin(combs, combs_train))[0]
    val = np.where(np.isin(combs, combs_val))[0]
    return (train, val)

outdir = "data/optimization/chemgen/"

if __name__ == "__main__":
     # Command line arguments
    # argv[1]: input file (chemical genetics data)
    # argv[2]: combination interaction type (input file)
    # argv[3]: classifier (RandomForest or XGBClassifier)
    # argv[4]: slice (1 - 1000) of the ParameterGrid
    # argv[5]: gene subset

    # argv = ['', '../data/chemgenetics/nichols_signed.csv',
    #          '../data/chemgenetics/nichols_y.csv',
    #         'randomforest', 1, 'data/interaction-genes-Ecoli']
    if sys.argv[3] == 'randomforest':
        pgrid = ParameterGrid(RF_grid)
    elif sys.argv[3] == 'xgboost':
        pgrid = ParameterGrid(xgb_grid)

    indices = np.array_split(np.arange(len(pgrid)), 1000)

    print("Total grid size: ", len(pgrid))
    print("Length of indices: ", len(indices))
    X_chemgen = pd.read_csv(sys.argv[1], index_col=0)
    targets = pd.read_csv(sys.argv[2])
    combs = targets['comb'].values
    y = targets['type'].values

    if len(sys.argv) > 5 and os.path.isfile(sys.argv[5]):
        gene_file = sys.argv[5]
        gene_subset = pd.read_csv(gene_file, header=None)[0].values
        X_chemgen = X_chemgen.iloc[:,np.where(np.isin(X_chemgen.columns, gene_subset))[0]]

    if (X_chemgen < 0).any().any():
        X_df = pd.DataFrame([utl.get_comb_feat_signed(X_chemgen, c) for c in combs])
    else:
        X_df = pd.DataFrame([utl.get_comb_feat(X_chemgen, c) for c in combs])
    X_onehot = pd.get_dummies(X_df.astype('category'))
    # at least 5 combinations with that variable set
    X_onehot = X_onehot.loc[:,(X_onehot.sum(axis=0) > 4)]
   
    strain = os.path.basename(sys.argv[2]).replace('_y.csv', '').title()
    outdir += sys.argv[3] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # one vs rest classification
    y[y=='none'] = 0
    y[y=='antagonism']=1
    y[y=='synergy']=2
    y=y.astype('int')
    y = label_binarize(y, classes=[0, 1, 2])
    # kf = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1401)
    # splits = list(kf.split(X=X_onehot.to_numpy(),y=y))

    np.random.seed(1601)
    # drugs in the chemical genetics dataset of E. coli
    drugs = np.unique(X_chemgen.index)
    # generate 20 CV folds by withholding 15 randomly chosen drugs
    splits = [generate_train_val(drugs, combs) for i in range(20)]
    
    metrics = [(prd.MultiClassPredictions(X=X_onehot.to_numpy(),
                                          y=y,
                                          combs=combs,
                                          clf=sys.argv[3], **pgrid[i]).
                crossval_ksplit(splits=splits).
                aggregate_precision())\
                   for i in indices[int(sys.argv[4])]]

    metrics_df = pd.DataFrame()
    for i, key in zip(range(len(metrics)), indices[int(sys.argv[4])]):
         df = metrics[i]
         df.loc[:,'index'] = key
         metrics_df = pd.concat([metrics_df, df], ignore_index=True)
    
    metrics_df.to_csv(outdir + strain + "-multiclass-" + str(sys.argv[4]) + ".tsv", sep='\t',
                      index=False)
   

