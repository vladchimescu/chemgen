#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd
import os
import sys

from sklearn.model_selection import ParameterGrid

import chemgen_utils as utl
import predictor as prd

""" 
Find optimal hyperparameters of the RandomForestClassifier and
XGBClassifier
"""

# first specify a parameter grid
# here 9 * 3 * 8 * 8 * 11 = 19008 combinations of hyperparameters
RF_grid = {"max_depth": list(range(2,10)) + [None],
           "n_estimators": [200, 500, 1000],
           "min_samples_split": range(2,10),
           "min_samples_leaf": range(2,10),
           "class_weight": [{0: 1, 1: i} for i in range(1,10)] + ['balanced', 'balanced_subsample']
           }
# XGB has 20 * 20 * 8 * 3 * 20 = 192000 combinations of hyperparameters
xgb_grid = {"learning_rate": np.linspace(0.1,0.9, 20),
            "colsample_bytree": np.linspace(0.1,0.9,20),
            "max_depth": range(2,10),
            "n_estimators": [200, 500, 1000],
            "scale_pos_weight": np.linspace(1,10,20)
            }
outdir = "data/optimization/chemgen/"
drugleg_fname = "data/chemicals/legend_gramnegpos.txt"

if __name__ == "__main__":
     # Command line arguments
    # argv[1]: input file (chemical genetics data)
    # argv[2]: combination interaction type (input file)
    # argv[3]: classifier (RandomForest or XGBClassifier)
    # argv[4]: slice (1 - 1000) of the ParameterGrid

    # argv = ['', '../data/chemgenetics/nichols_binarized.csv',
    #          '../data/chemgenetics/nichols_y.csv']

    if sys.argv[3] == 'randomforest':
        pgrid = ParameterGrid(RF_grid)
        indices = np.array_split(np.arange(len(pgrid)), 1000)
    else:
        pgrid = ParameterGrid(xgb_grid)
        indices = np.array_split(np.arange(len(pgrid)), 10000)

    print("Total grid size: ", len(pgrid))
    print("Length of indices: ", len(indices))
    X_chemgen = pd.read_csv(sys.argv[1], index_col=0)
    targets = pd.read_csv(sys.argv[2])
    combs = targets['comb'].values
    y = targets['type'].values

    X_df = pd.DataFrame([utl.get_comb_feat(X_chemgen, c) for c in combs])

    drugclasses = pd.read_csv(drugleg_fname, sep='\t')
    druglegend = drugclasses.loc[:,['Drug', 'Class']]

    comb_drugs = pd.DataFrame(np.array([utl.split_vec(i) for i in combs]),
                              columns=['d1', 'd2'])
    comb_drugs = utl.add_class(strain=comb_drugs,
                               druglegend=druglegend)
    # an array with all drug class labels
    class_arr = np.unique(np.union1d(pd.unique(comb_drugs.class1),
                                     pd.unique(comb_drugs.class2)))
    strain = os.path.basename(sys.argv[2]).replace('_y.csv', '').title()
    outdir += sys.argv[3] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # first antagonism vs none
    y_an, X_an, combs_an, leg_class = utl.subset_interactions(y=y,
                                                              X=X_df,
                                                              combs=combs,
                                                              comb_drugs=comb_drugs,
                                                              which=['none',
                                                                     'antagonism'])
    
    obj = prd.ObjectiveFun(X=X_an, y=y_an, combs=combs_an,
                           clf=sys.argv[3])
    metrics = [(obj.set_params(**pgrid[i]).
               aggregate_precision(class_arr=class_arr,
                                   leg_class=leg_class))\
                   for i in indices[int(sys.argv[4])]]

    metrics_df = pd.concat(metrics, ignore_index=True)
    metrics_df['index'] = indices[int(sys.argv[4])]
    metrics_df.to_csv(outdir + strain + "-AN-" + str(sys.argv[4]) + ".tsv", sep='\t',
                      index=False)

    # synergy vs none predictions
    y_sn, X_sn, combs_sn, leg_class = utl.subset_interactions(y=y,
                                                              X=X_df,
                                                              combs=combs,
                                                              comb_drugs=comb_drugs,
                                                              which=['none',
                                                                     'synergy'])

    obj = prd.ObjectiveFun(X=X_sn, y=y_sn, combs=combs_sn,
                           clf=sys.argv[3])
    metrics = [(obj.set_params(**pgrid[i]).
               aggregate_precision(class_arr=class_arr,
                                   leg_class=leg_class))\
                   for i in indices[int(sys.argv[4])]]

    metrics_df = pd.concat(metrics, ignore_index=True)
    metrics_df['index'] = indices[int(sys.argv[4])]
    metrics_df.to_csv(outdir + strain + "-SN-" + str(sys.argv[4]) + ".tsv", sep='\t',
                      index=False)
   

