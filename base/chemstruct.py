#!/usr/bin/env python3
'''
Utility functions for prediction of drug-drug
interactions based on chemical structures (ECPF4 fingerprints)
'''
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect, GetHashedTopologicalTorsionFingerprintAsBitVect
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP 


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
