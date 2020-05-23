#!/usr/bin/env python3
'''
Functions and classes for prediction of drug-drug
interactions based on chemical genetics or 
chemical features
'''
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect, GetHashedTopologicalTorsionFingerprintAsBitVect
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP 


# function for splitting combinations into drug1 and drug2
split_vec = lambda x: x.split("_")

# for the binary feature matrix
def get_comb_feat(chemgen, c):
    comb_feat = chemgen.loc[split_vec(c),:]
    
    return comb_feat.sum(axis=0)


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

def get_smiles(fname):
    smiles_df = pd.read_csv(fname,
                        sep="\t", header=None,
                            names=['SMILES', 'drug'])
    smiles_df["mol"] = [Chem.MolFromSmiles(x) for x in smiles_df["SMILES"]]
    return smiles_df

def fp_to_pandas(fp, drug_names):
    fp_np = []
    for fp in fp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_np.append(arr)
    fp_df = pd.DataFrame(fp_np, index=drug_names)
    return fp_df

def get_fingerprints(smiles_df, r=2, length=512,
                     type_='morgan'):
    if type_ == 'morgan':
        fp = [AllChem.GetMorganFingerprintAsBitVect(m, r,
                                                    nBits = length)\
              for m in smiles_df['mol']]
    elif type_ == 'fcpf':
        fp = [AllChem.GetMorganFingerprintAsBitVect(m, r,
                                                    useFeatures=True,
                                                    nBits = length)\
              for m in smiles_df['mol']]
    elif type_ == 'atom pair':
        fp = [GetHashedAtomPairFingerprintAsBitVect(m,
                                                    nBits = length)\
              for m in smiles_df['mol']]
    elif type_ == 'avalon':
         fp = [GetAvalonFP(m, nBits = length) for m in smiles_df['mol']]
    elif type_ == 'torsion':
        fp = [GetHashedTopologicalTorsionFingerprintAsBitVect(m,
                                                         nBits = length)\
         for m in smiles_df['mol']]
    elif type_ == 'rdkit':
        fp = [RDKFingerprint(m, fpSize = length) for m in smiles_df['mol']]
    else:
        raise ValueError("Possible values: morgan, fcpf, atom pair, avalon, torision and rdkit")

    drug_names = smiles_df['drug'].values
    return fp_to_pandas(fp=fp, drug_names=drug_names)


def get_struct_feat(fp_select, c):
    fp_select = fp_select.iloc[:,np.where(fp_select.sum(axis=0).values > 0)[0]]
    comb_feat = fp_select.loc[split_vec(c),:]
    return comb_feat.sum(axis=0)
