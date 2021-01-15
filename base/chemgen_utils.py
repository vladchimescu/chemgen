#!/usr/bin/env python3
'''
Functions and classes for prediction of drug-drug
interactions based on chemical genetics
'''
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd


# function for splitting combinations into drug1 and drug2
split_vec = lambda x: x.split("_")

# for the binary feature matrix
def get_comb_feat(chemgen, c):
    comb_feat = chemgen.loc[split_vec(c),:]
    comb_sum = comb_feat.sum(axis=0)
    return comb_sum

def get_comb_feat_signed(chemgen, c):
    comb_feat = chemgen.loc[split_vec(c),:]
    comb_sum = comb_feat.sum(axis=0)
    comb_signed = comb_sum + 3*(np.abs(comb_feat.iloc[0,:] - comb_feat.iloc[1,:]) == 2)
    return comb_signed


def add_class(strain, druglegend):
    df1 = druglegend.rename(columns = {"Drug" : "d1",
                                       "Class": "class1"})
    strain = pd.merge(strain, df1, how='left', on = "d1")
    
    df2 = druglegend.rename(columns = {"Drug" : "d2",
                                       "Class": "class2"})
    strain = pd.merge(strain, df2, how='left', on = "d2")
    return strain


def split_drug_class(X, leg_class, cl):
    test_ind = np.where(np.logical_or(leg_class.class1 == cl,
                                      leg_class.class2 == cl))[0]
    train_ind = np.setdiff1d(np.arange(X.shape[0]), test_ind)
    return train_ind, test_ind

def subset_interactions(y, X, combs, comb_drugs, which):
    # which = ['none', 'antagonism']
    which_int = np.where(np.isin(y, which))[0]
    leg_class = comb_drugs.loc[which_int,['class1', 'class2']].reset_index(drop=True)
    combs_an = combs[which_int]
    X_an = X.iloc[which_int,:].reset_index(drop=True)
    y_an = y[which_int]
    y_an[y_an == 'none'] = False
    y_an[y_an == 'antagonism'] = True
    y_an = np.asarray(y_an, dtype = np.bool)
    X_an = X_an.to_numpy(copy=True)

    return y_an, X_an, combs_an, leg_class
