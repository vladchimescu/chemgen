#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import sys

import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import csv
import datetime
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers
#from keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
from tensorflow.keras.callbacks import TensorBoard
#from keras.callbacks.callbacks import ModelCheckpoint
#import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.tree import export_graphviz

from sklearn.multiclass import OneVsRestClassifier
from matplotlib.backends.backend_pdf import PdfPages
from  itertools import cycle
from scipy.interpolate import make_interp_spline, BSpline

sys.path.append('..')
from chemgen_utils import split_drug_class

def export_rftrees(estimators, outdir, class_names, featname):
        if outdir is not None:
            tree_out = outdir + "/"
            if not os.path.exists(tree_out):
                os.makedirs(tree_out)

            for i, estimator in enumerate(estimators):
                export_graphviz(estimator,
                                out_file=tree_out +\
                                '-'.join(class_names) +\
                                "_"+ str(i) + '.dot', 
                                feature_names = featname,
                                class_names = class_names,
                                rounded = True, proportion = False, 
                                precision = 2, filled = True)


def export_xgb(estimators, outdir, cl, class_names, featname):
    if outdir is not None:
        # self.clf.get_booster().feature_names = list(featname)
        # estimators_ = (self.clf.get_booster().
        #                get_dump(with_stats=True, dump_format="dot"))

        tree_out = outdir + "/"
        if not os.path.exists(tree_out):
            os.makedirs(tree_out)

        for i in range(len(estimators)):
            estimator = estimators[i]
            fname = tree_out +\
                            '-'.join(class_names) +\
                            "_"+ str(i) + '.dot'
            file = open(fname, 'w')
            file.write(estimator)
            file.close()

class BasePredictions:
    def __init__(self, **kwargs):
        self.X = kwargs.get("X")
        self.y = kwargs.get("y")
        self.combs = kwargs.get("combs")
        self.clf = kwargs.get("clf")

        # arguments with default values
        self.top = kwargs.get("top", 20)
        self.mean_fpr = np.linspace(0, 1, 100)
        self.n_estimators = kwargs.get("n_estimators", 500)
        self.class_weight = kwargs.get("class_weight", "balanced")
        self.min_samples_split=kwargs.get("min_samples_split", 3)
        self.min_samples_leaf = kwargs.get("min_samples_leaf", 3)
        self.colsample_bytree = kwargs.get("colsample_bytree", 0.6)
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        self.random_state = kwargs.get("random_state", 125)
        self.max_depth = kwargs.get("max_depth", None)
        self.objective = kwargs.get("objective", 'binary:logistic')
        self.scale_pos_weight = kwargs.get("scale_pos_weight", 1)


        #for neural net
        self.layers = kwargs.get("layers", 3)
        self.dropout = kwargs.get("dropout", 0.2)
        self.nodes = kwargs.get("nodes", 32)
        self.epochs = kwargs.get("epochs", 200)
        self.steps = kwargs.get("steps", 32)
        self.learning_rate_deep = kwargs.get("learning_rate_deep", 0.001)
        #self.beta_1 = kwargs.get("beta_1", 0.9)
        #self.beta_2 = kwargs.get("beta_2", 0.999)

        self._set_classifier()

        self.predicted = dict()
        self.topfeat = dict()
        self.fpr = dict()
        self.tpr = dict()
        self.tprs = dict()
        self.auc = dict()
        self.precision = dict()
        self.recall = dict()
        self.avprec = dict()

    def _set_classifier(self):  
        if self.clf.lower() == "randomforest":
            self.clf = RandomForestClassifier(bootstrap=True,
                                              class_weight=self.class_weight,
                                              max_depth=self.max_depth,
                                              n_estimators=self.n_estimators,
                                              max_features='sqrt',
                                              min_samples_split=self.min_samples_split,
                                              min_samples_leaf=self.min_samples_leaf,
                                              random_state=self.random_state,
                                              n_jobs=-1)
        elif self.clf.lower() == "xgboost":
            if self.max_depth is None:
                self.max_depth = 5

            self.clf = xgb.XGBClassifier(learning_rate=self.learning_rate,
                    colsample_bytree=self.colsample_bytree,
                    random_state=self.random_state,
                    max_depth = self.max_depth,
                    n_estimators = self.n_estimators,
                    scale_pos_weight= self.scale_pos_weight,
                    objective=self.objective,
                    n_jobs= -1)


        elif self.clf.lower() == "neural_network":
            def Neural_network(dropout=0.2, nodes=32, layers=3, learning_rate_deep=0.001):
                clf = Sequential()
                for i in range(self.layers):
                    #With BN
                    #clf.add(Dense(self.nodes, activation="linear", use_bias="False"))
                    #clf.add(BatchNormalization())
                    #clf.add(Activation("relu"))
                    #clf.add(Dropout(self.dropout))
                    #Without BN
                    clf.add(Dense(self.nodes, activation="relu"))
                    clf.add(Dropout(self.dropout))
                clf.add(Dense(3, activation="softmax"))
                
                opt = keras.optimizers.Adam(learning_rate=self.learning_rate_deep)#, beta_1=self.beta_1, beta_2=self.beta_2)
                clf.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer = opt,  metrics = ["accuracy"]) #other option loss = "binary_crossentropy"
                return clf
            self.clf = KerasClassifier(Neural_network, epochs = self.epochs, steps_per_epoch=self.steps, dropout = self.dropout, nodes = self.nodes, layers = self.layers, learning_rate_deep = self.learning_rate_deep) #class_weight=self.class_weight

        else:
            raise ValueError("only randomforest, xgboost and neural_network are supported")

        

class InteractionPredictions(BasePredictions):
    '''Binary interaction predictions
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _crossval_iter(self, train, test, cl):
        X_test = self.X[test]
        y_test = self.y[test]
        combs_test = self.combs[test]
        probas_ = self.clf.fit(self.X[train], self.y[train]).predict_proba(X_test)

        pred_df = pd.DataFrame({'comb': combs_test,
                                'score': probas_[:,1]})
    
        try:

            importances = self.clf.feature_importances_
            self.topfeat[cl] = pd.DataFrame({'feat': np.argsort(importances)[::-1][:self.top],
                      'importance': np.sort(importances)[::-1][:self.top]})

        except:
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])

            if not np.any(np.isnan(tpr)):
                self.fpr[cl] = fpr
                self.tpr[cl] = tpr

                tprs = interp(self.mean_fpr, fpr, tpr)
                tprs[0] = 0.0
                self.tprs[cl] = tprs
                self.auc[cl] = auc(fpr, tpr)

            precision, recall, _ = precision_recall_curve(y_test, probas_[:,1])
            average_precision = average_precision_score(y_test, probas_[:,1])

            if not np.any(np.isnan(recall)):
                self.precision[cl] = precision
                self.recall[cl] = recall
                self.avprec[cl] = average_precision
                self.predicted[cl] = pred_df

    def crossval_drugclass(self, class_arr, leg_class,
                           class_names, featname, treedir):
        for cl in class_arr:
            train, test = split_drug_class(X=self.X,
                                           leg_class=leg_class, cl=cl)

            print("Test set size in %s: %d" % (cl, test.shape[0]))
            # run a single iteration of cross-validation with the
            # current train / test split and CV fold `cl`
            self._crossval_iter(train, test, cl)
        return self

    def crossval_ksplit(self, splits):
        for cl, split in enumerate(splits):
            train, test = split
            print("Validation set size in CV fold %s: %d" % (cl+1, test.shape[0]))
            # run a single iteration of cross-validation with the
            # current train / test split and CV fold `cl`
            self._crossval_iter(train, test, cl)

        return self
                   
class MultiClassPredictions(InteractionPredictions):
    """
    Class for making and storing OneVsRest predictions
    of synergies and antagonisms
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.class_names = ['none', 'antag', 'syn']
        self.one_vs_rest()

    def one_vs_rest(self):
        self.clf = OneVsRestClassifier(self.clf)

    def set_params(self, **kwargs):
        # for random forest
        self.clf.set_params(**kwargs)
        return self

    def _crossval_iter(self, train, test, cl):
        X_test = self.X[test]
        y_test = self.y[test]
        combs_test = self.combs[test]

        probas_ = self.clf.fit(self.X[train], self.y[train]).predict_proba(X_test)
        # predictions
        pred_dict ={'comb': combs_test}
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        precision = dict()
        recall = dict()
        average_precision = dict()

        n_classes = len(self.class_names)
        for i in range(n_classes):
            pred_dict['score_'+str(self.class_names[i])] = probas_[:,i]
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probas_[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                                probas_[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], probas_[:, i])

        self.fpr[cl] = fpr
        self.tpr[cl] = tpr
        self.auc[cl] = roc_auc
        self.precision[cl] = precision
        self.recall[cl] = recall
        self.avprec[cl] = average_precision
        self.predicted[cl] = pd.DataFrame(pred_dict)

        try:
            importances = [self.clf.estimators_[i].feature_importances_ for i in range(n_classes)]
        
            imp_list = [pd.DataFrame({'feat': np.argsort(imp)[::-1][:self.top],
                                      'importance': np.sort(imp)[::-1][:self.top],
                                      'type': i})\
                        for imp, i in zip(importances, self.class_names)]
            imp_df = pd.concat(imp_list, ignore_index=True)
            self.topfeat[cl] = imp_df
        except:
            pass

    def aggregate_precision(self):
        index = ['AP_' + lab for lab in self.class_names]
        ap_df = (pd.concat({k: pd.DataFrame(v.values(),
                                            index=index).T \
                   for k,v in self.avprec.items()}).
         reset_index().rename(columns={"level_0": "cvfold"}).
         drop(columns=["level_1"]))
        return ap_df

    def aggregate_auc(self):
        index = ['AUCROC_' + lab for lab in self.class_names]
        auc_df = (pd.concat({k: pd.DataFrame(v.values(),
                                             index=index).T \
                   for k,v in self.auc.items()}).
         reset_index().rename(columns={"level_0": "cvfold"}).
         drop(columns=["level_1"]))
        
        return auc_df

        
    def plot_ROC(self, figdir=None, fname=None,
                 title='One-vs-Rest ROC curves', sz=10):
        class_names = ['none', 'antagonism', 'synergy']
        colors = cycle(['#808080','#FFCC33', '#009999'])
        n_classes = 3
        if figdir is not None and fname is not None:
            with PdfPages(figdir + fname + '.pdf') as pdf:
                for cl in list(self.auc.keys()):
                    plt.figure(figsize=(sz,sz))
                    for i, color in zip(range(n_classes), colors):
                        plt.plot(self.fpr[cl][i], self.tpr[cl][i], color=color, lw=2,
                                 label='ROC curve of class {0} (area = {1:0.2f})'
                                 ''.format(class_names[i], self.auc[cl][i]))

                    plt.plot([0, 1], [0, 1], 'k--', lw=2)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(title + cl)
                    plt.legend(loc="lower right")
                    pdf.savefig()
                    plt.close()   

    def plot_precision(self, figdir=None, fname=None,
                       title='One-vs-Rest Precision-Recall', sz=10):
        class_names = ['none', 'antagonism', 'synergy']
        colors = cycle(['#808080','#FFCC33', '#009999'])
        n_classes = 3
        if figdir is not None and fname is not None:
            with PdfPages(figdir + fname + '.pdf') as pdf:
                for cl in list(self.avprec.keys()):
                    plt.figure(figsize=(sz,sz))
                    f_scores = np.linspace(0.2, 0.8, num=4)
        
                    for f_score in f_scores:
                        x = np.linspace(0.01, 1)
                        y_ = f_score * x / (2 * x - f_score)
                        plt.plot(x[y_ >= 0], y_[y_ >= 0], color='gray', alpha=0.2,
                                 label='iso-F1 curves')
                        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y_[45] + 0.02))
                    for i, color in zip(range(n_classes), colors):
                        plt.plot(self.recall[cl][i], self.precision[cl][i], color=color, lw=2,
                                 label='Precision-recall of class {0} (area = {1:0.2f})'
                                 ''.format(class_names[i], self.avprec[cl][i]))

                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(title + cl)
                    plt.legend(loc="lower right")
                    pdf.savefig()
                    plt.close()

    def save_metrics(self, outdir=None, fname=None):
        auc_df = self.aggregate_auc()
        ap_df = self.aggregate_precision()
        metrics = pd.merge(auc_df, ap_df, on='cvfold', how='inner')
        
        if outdir is not None and fname is not None:
            metrics.to_csv(outdir + fname + '.tsv', sep="\t",
                           index=False)