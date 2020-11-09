#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import sys
import os
import matplotlib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score
from matplotlib.lines import Line2D

sys.path.append('.')
import base.chemgen_utils as utl
import MLmod.predictor as prd

figdir = 'figures/'
drugleg_fname = "data/chemicals/legend_gramnegpos.txt"
gene_subset = 'data/interaction-genes-Ecoli'

param_dict = {'n_estimators': 300,
 'min_samples_split': 6,
 'min_samples_leaf': 2,
 'max_depth': None,
 'class_weight': None}

drug_classes = ['DNA_gyrase', 'RNA_polymerase', 'aminoglycoside',
                'beta-lactam', 'chloramphenicol', 'folic_acid_biosynthesis',
               'human_drug', 'macrolide', 'multiple', 'other_DNA',
               'other_cell_wall', 'oxidative_stress', 'tRNA',
               'tetracycline']

if __name__ == "__main__":
    gene_subset = pd.read_csv(gene_subset, header=None)[0].values
    X_chemgen = pd.read_csv('data/chemgenetics/nichols_signed.csv',
                            index_col=0)
    X_chemgen = X_chemgen.iloc[:,np.where(np.isin(X_chemgen.columns, gene_subset))[0]]
    targets = pd.read_csv("data/chemgenetics/nichols_y.csv")
    combs = targets['comb'].values
    y = targets['type'].values

    X_df = pd.DataFrame([utl.get_comb_feat_signed(X_chemgen, c) for c in combs])
    X_onehot = pd.get_dummies(X_df.astype('category'))
    # at least 5 combinations with that variable set
    X_onehot = X_onehot.loc[:,(X_onehot.sum(axis=0) > 4)]

    # one vs rest classification
    y[y=='none'] = 0
    y[y=='antagonism']=1
    y[y=='synergy']=2

    y=y.astype('int')
    y = label_binarize(y, classes=[0, 1, 2])

    drugclasses = pd.read_csv(drugleg_fname, sep='\t')
    druglegend = drugclasses.loc[:,['Drug', 'Class']]

    comb_drugs = pd.DataFrame(np.array([utl.split_vec(i) for i in combs]),
                              columns=['d1', 'd2'])
    comb_drugs = utl.add_class(strain=comb_drugs,
                               druglegend=druglegend)
    # an array with all drug class labels
    class_arr = np.unique(np.union1d(pd.unique(comb_drugs.class1),
                                     pd.unique(comb_drugs.class2)))

    pr = prd.MultiClassPredictions(X=X_onehot.to_numpy(), y=y,
                                   combs=combs,
                                  **param_dict,
                                   clf='randomforest',
                                   top = 30)
    pr.crossval_drugclass(class_arr=class_arr, leg_class=comb_drugs)

    font = {'family': 'normal',
        'weight': 'normal',
        'size': 18}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6,6))
    for cl in drug_classes:
        plt.plot(pr.recall[cl][1], pr.precision[cl][1], lw=2, alpha=0.75,
             label='%s (AP = %0.2f)' % (cl, pr.avprec[cl][1]))
        plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    ax.yaxis.get_major_ticks()[1].label1.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend([], frameon=False)
    plt.tight_layout()
    plt.savefig(figdir + 'Ecoli-antagonism-vs-rest.pdf')


    figsize = (6, 6)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', frameon=False)
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.savefig(figdir + 'Ecoli-legend-antag.pdf')


    fig, ax = plt.subplots(figsize=(6,6))
    for cl in drug_classes:
        plt.plot(pr.recall[cl][2], pr.precision[cl][2], lw=2, alpha=0.75,
             label='%s (AP = %0.2f)' % (cl, pr.avprec[cl][2]))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    ax.yaxis.get_major_ticks()[1].label1.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend([], frameon=False)
    plt.tight_layout()
    plt.savefig(figdir + 'Ecoli-synergy-vs-rest.pdf')

    figsize = (6, 6)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', frameon=False)
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.savefig(figdir + 'Ecoli-legend-synergy.pdf')


    preds_ = (pd.concat(pr.predicted).
                  reset_index().
                  rename(columns={"level_0": "cvfold"}).
                  drop(columns=['level_1']))
    y_gt = targets.loc[:, ['comb', 'type']]
    y_gt['syn'] = 0
    y_gt['ant'] = 0
    y_gt.loc[y_gt.type == 1,'ant'] = 1
    y_gt.loc[y_gt.type == 2,'syn'] = 1

    pred_vs_true = pd.merge(preds_, y_gt, on='comb')
    pred_vs_true = pred_vs_true[np.isin(pred_vs_true.cvfold, drug_classes)]

    # A "micro-average": quantifying score on all classes jointly
    prec_micro, recall_micro, _ = precision_recall_curve(pred_vs_true['ant'].values,
                                                        pred_vs_true['prob_ant'].values)
    ap_micro = average_precision_score(pred_vs_true['ant'].values,
                                       pred_vs_true['prob_ant'].values,
                                       average="micro")

    fig, ax = plt.subplots(figsize=(6.5,6.5))
    for cl in drug_classes:
        plt.plot(pr.recall[cl][1], pr.precision[cl][1],
                 color='grey', lw=1.5, alpha=0.6)
        plt.xlim([-0.05, 1.05])
    plt.plot(recall_micro, prec_micro,
            label='Aggregated (AP = {0:0.2f})'
                   ''.format(ap_micro),
             color='#f54248', linestyle='-', linewidth=3)

    handles, labels = ax.get_legend_handles_labels()
    patch = Line2D([0], [0], color='grey', linewidth=2, linestyle="-",
                   label='Drug class withheld')
    handles.append(patch) 
    plt.ylim([-0.01, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
    ax.yaxis.get_major_ticks()[1].label1.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.title(title)
    plt.legend(handles=handles, loc='upper right', prop={"size": 13},
              frameon=False)
    plt.tight_layout()
    plt.savefig(figdir + 'Ecoli-antagonism-vs-rest-grey.pdf')

    # A "micro-average": quantifying score on all classes jointly
    prec_micro, recall_micro, _ = precision_recall_curve(pred_vs_true['syn'].values,
                                                        pred_vs_true['prob_syn'].values)
    ap_micro = average_precision_score(pred_vs_true['syn'].values,
                                       pred_vs_true['prob_syn'].values,
                                       average="micro")

    fig, ax = plt.subplots(figsize=(6.5,6.5))
    for cl in drug_classes:
        plt.plot(pr.recall[cl][2], pr.precision[cl][2],
                color='grey', lw=1.5, alpha=0.6)
        plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall_micro, prec_micro,
            label='Aggregated (AP = {0:0.2f})'
                   ''.format(ap_micro),
             color='#f54248', linestyle='-', linewidth=3)

    handles, labels = ax.get_legend_handles_labels()
    patch = Line2D([0], [0], color='grey', linewidth=2, linestyle="-",
                   label='Drug class withheld')
    handles.append(patch) 
    plt.ylim([-0.02, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    ax.yaxis.get_major_ticks()[1].label1.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.title(title)
    plt.legend(handles=handles, loc='upper right',
               prop={"size": 13},
                #bbox_to_anchor=(1.2,0.8),
              frameon=False)
    #plt.legend([], frameon=False)
    plt.tight_layout()
    plt.savefig(figdir + 'Ecoli-synergy-vs-rest-grey.pdf')

    fig, ax = plt.subplots(figsize=(7,7))
    for cl in drug_classes:
        plt.plot(pr.fpr[cl][1], pr.tpr[cl][1], lw=2, alpha=0.7,
                 label='{0} (area = {1:0.2f})'
                 ''.format(cl, pr.auc[cl][1]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.yaxis.get_major_ticks()[1].label1.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend([], frameon=False)
    plt.savefig(figdir + 'Ecoli-ROC-antagonism-vs-rest.pdf')

    figsize = (6, 6)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', frameon=False)
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.savefig(figdir + 'Ecoli-ROC-legend-antag.pdf')

    fig, ax = plt.subplots(figsize=(7,7))
    for cl in drug_classes:
        plt.plot(pr.fpr[cl][2], pr.tpr[cl][2], lw=2, alpha=0.7,
                 label='{0} (area = {1:0.2f})'
                 ''.format(cl, pr.auc[cl][2]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.yaxis.get_major_ticks()[1].label1.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend([], frameon=False)
    plt.savefig(figdir + 'Ecoli-ROC-synergy-vs-rest.pdf')

    figsize = (6, 6)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', frameon=False)
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.savefig(figdir + 'Ecoli-ROC-legend-syn.pdf')

    fpr_micro, tpr_micro, _ = roc_curve(pred_vs_true['ant'].values,
                                                    pred_vs_true['prob_ant'].values)
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    fig, ax = plt.subplots(figsize=(6.5,6.5))
    for cl in drug_classes:
        plt.plot(pr.fpr[cl][1], pr.tpr[cl][1],
                 color='grey', lw=1.5, alpha=0.6)
    plt.plot(fpr_micro, tpr_micro,
            label='Aggregated (AUCROC = {0:0.2f})'
                   ''.format(roc_auc_micro),
             color='#f54248', linestyle='-', linewidth=3)

    handles, labels = ax.get_legend_handles_labels()
    patch = Line2D([0], [0], color='grey', linewidth=2, linestyle="-",
                   label='Drug class withheld')
    handles.append(patch) 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(handles=handles, loc='lower right', prop={"size": 13},
              frameon=False)
    plt.tight_layout()
    plt.savefig(figdir + 'Ecoli-ROC-antagonism-vs-rest-grey.pdf')


    fpr_micro, tpr_micro, _ = roc_curve(pred_vs_true['syn'].values,
                                                    pred_vs_true['prob_syn'].values)
    roc_auc_micro = auc(fpr_micro, tpr_micro)


    fig, ax = plt.subplots(figsize=(6.5,6.5))
    for cl in drug_classes:
        plt.plot(pr.fpr[cl][2], pr.tpr[cl][2],
                 color='grey', lw=1.5, alpha=0.6)
    plt.plot(fpr_micro, tpr_micro,
            label='Aggregated (AUCROC = {0:0.2f})'
                   ''.format(roc_auc_micro),
             color='#f54248', linestyle='-', linewidth=3)

    handles, labels = ax.get_legend_handles_labels()
    patch = Line2D([0], [0], color='grey', linewidth=2, linestyle="-",
                   label='Drug class withheld')
    handles.append(patch) 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    #ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
    plt.legend(handles=handles, loc='lower right', prop={"size": 13},
              frameon=False)
    plt.tight_layout()
    plt.savefig(figdir + 'Ecoli-ROC-synergy-vs-rest-grey.pdf')
