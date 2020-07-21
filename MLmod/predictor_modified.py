#!/usr/bin/env python3

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
import sys
import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

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
        self.epochs = kwargs.get("epochs", 300)
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
                    clf.add(Dense(self.nodes, activation="relu"))
                    clf.add(Dropout(self.dropout))
                clf.add(Dense(3, activation="softmax"))
                
                # Approach 1
                # clf.add(Dense(32, activation="relu"))#, input_dim=1))
                # clf.add(Dropout(0.2))
                # clf.add(Dense(32, activation="relu"))
                # clf.add(Dropout(0.2))
                # clf.add(Dense(32, activation="relu"))
                # clf.add(Dropout(0.2))
                # clf.add(Dense(32, activation="relu"))
                # clf.add(Dense(3, activation="softmax"))
                
                #1 Approach 2
                # clf.add(Dense(32, activation="relu"))
                # clf.add(Dropout(0.5))
                # clf.add(Dense(3, activation="softmax"))             
                
                opt = keras.optimizers.Adam(learning_rate=self.learning_rate_deep)#, beta_1=self.beta_1, beta_2=self.beta_2)
                clf.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer = opt,  metrics = ["accuracy"]) #other option loss = "binary_crossentropy"
                return clf
            self.clf = KerasClassifier(Neural_network, epochs = self.epochs, steps_per_epoch=self.steps,
                    class_weight=self.class_weight, dropout = self.dropout, nodes = self.nodes, layers = self.layers, learning_rate_deep = self.learning_rate_deep)
            
        else:
            raise ValueError("only randomforest, xgboost and neural_network are supported")

        

class InteractionPredictions(BasePredictions):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def crossval_drugclass(self, class_arr, leg_class,
                           class_names, featname, treedir):
        for cl in class_arr:
            train, test = split_drug_class(X=self.X,
                                           leg_class=leg_class, cl=cl)

            X_test = self.X[test]
            y_test = self.y[test]
            combs_test = self.combs[test]

            print("Test set size in %s: %d" % (cl, X_test.shape[0]))
            
            self.clf.fit(self.X[train], self.y[train])
            probas_ = self.clf.predict_proba(X_test)
            
            #probas_ = self.clf.fit(self.X[train], self.y[train]).predict_proba(X_test)

            pred_df = pd.DataFrame({'comb': combs_test,
                                     'prob': probas_[:,1]})
            
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
                    # change here to calling an internal function
                    # that plots trees depending on whether it's a
                    # random forest or XGBoost
                    if hasattr(self.clf, 'estimators_'):
                        self._export_rftrees(outdir=treedir,
                                             cl=cl,
                                             featname=featname,
                                             class_names=class_names)
                    if hasattr(self.clf, 'get_booster'):
                        self._export_xgb(outdir=treedir,
                                         cl=cl,
                                         featname=featname,
                                         class_names=class_names)

    def plot_ROC(self, figdir=None, fname=None, title='ROC curves',
                 sz=13):
        font = {'family': 'normal',
                'weight': 'normal',
                'size': 18}
        mpl.rc('font', **font)
        plt.figure(figsize=(sz, sz))
        for cl in list(self.auc.keys()):
            fpr = self.fpr[cl]
            tpr = self.tpr[cl]
            roc_auc = self.auc[cl]
            
            plt.plot(fpr, tpr, lw=2, alpha=0.5,
                 label='ROC %s (AUC = %0.2f)' % (cl, roc_auc))


        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

        tprs = list(self.tprs.values())
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(list(self.auc.values()))
        plt.plot(self.mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right", prop={"size": 13})
        plt.tight_layout()
        if figdir is not None and fname is not None:
            plt.savefig(figdir + fname + ".pdf")

    def plot_precision(self, figdir=None, fname=None,
                       title='Precision-recall', sz=13):
        font = {'family': 'normal',
                'weight': 'normal',
                'size': 18}
        mpl.rc('font', **font)
        plt.figure(figsize=(sz, sz))
        for cl in list(self.auc.keys()):
             plt.plot(self.recall[cl], self.precision[cl], lw=2, marker='o', alpha=0.5,
                     label='%s (AP = %0.2f)' % (cl, self.avprec[cl]))
             plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="upper right", prop={"size": 13})
        plt.tight_layout()
        
        if figdir is not None and fname is not None:
            plt.savefig(figdir + fname + ".pdf")

    def save_predictions(self, outdir=None, fname=None):
        preds_ = (pd.concat(self.predicted).
                  reset_index().
                  rename(columns={"level_0": "cvfold"}).
                  drop(columns=['level_1']))
        if outdir is not None and fname is not None:
            preds_.to_csv(outdir + fname + '.tsv', sep="\t",
                          index=False)
        #return preds_


    def save_topfeat(self, outdir=None, fname=None, featname=None):
        topvars = (pd.concat(self.topfeat).
                   reset_index().
                   rename(columns={"level_0": "cvfold"}).
                   drop(columns=['level_1']))
        if featname is not None:
            topvars = (topvars.assign(feature=featname[topvars.feat]).
                       drop(columns=['feat']))

        if outdir is not None and fname is not None:
            topvars.to_csv(outdir + fname + '.tsv', sep="\t",
                          index=False)

    def save_metrics(self, outdir=None, fname=None):
        auc_df = pd.DataFrame(self.auc.values(),
                     columns = ['AUCROC'],
                     index = self.auc.keys())

        ap_df =  pd.DataFrame(self.avprec.values(),
                     columns = ['AP'],
                     index = self.avprec.keys())

        metrics = (pd.concat([auc_df, ap_df], axis=1).reset_index(level=0).
                   rename(columns={"index": "cvfold"}))
        if outdir is not None and fname is not None:
            metrics.to_csv(outdir + fname + '.tsv', sep="\t",
                           index=False)

    def _export_rftrees(self, outdir, cl, class_names, featname):
        if outdir is not None:
            tree_out = outdir + cl + "/"

            if not os.path.exists(tree_out):
                os.makedirs(tree_out)

            for i in range(len(self.clf.estimators_)):
                estimator = self.clf.estimators_[i]
                # Export as dot file
                export_graphviz(estimator,
                                out_file=tree_out +\
                                '-'.join(class_names) +\
                                "_"+ str(i) + '.dot', 
                                feature_names = featname,
                                class_names = class_names,
                                rounded = True, proportion = False, 
                                precision = 2, filled = True)


    def _export_xgb(self, outdir, cl, class_names, featname):
        if outdir is not None:
            self.clf.get_booster().feature_names = list(featname)
            estimators_ = (self.clf.get_booster().
                           get_dump(with_stats=True, dump_format="dot"))

            tree_out = outdir + cl + "/"

            if not os.path.exists(tree_out):
                os.makedirs(tree_out)

            for i in range(len(estimators_)):
                estimator = estimators_[i]
                fname = tree_out +\
                                '-'.join(class_names) +\
                                "_"+ str(i) + '.dot'
                file = open(fname, 'w')
                file.write(estimator)
                file.close()
           
class OOBPredictions(BasePredictions):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.error_rate = dict()

    def _set_classifier(self):
        self.clf = RandomForestClassifier(warm_start=True, oob_score=True,
                                              max_depth=self.max_depth,
                                              max_features='sqrt',
                                              min_samples_split=self.min_samples_split,
                                              min_samples_leaf=self.min_samples_leaf,
                                              random_state=self.random_state)

    def oob_by_drugclass(self, class_arr, leg_class):
        for cl in class_arr:
            train, test = split_drug_class(X=self.X,
                                           leg_class=leg_class, cl=cl)
            self.error_rate[cl] = []
            # reset the classifier 
            self._set_classifier()
            
            for i in range(10, 1005, 5):
                self.clf.set_params(n_estimators=i)
                self.clf.fit(self.X[train], self.y[train])
                #oob_error = 1 - self.clf.oob_score_
                y_pred = np.argmax(self.clf.oob_decision_function_, axis=1)
                oob_error = 1 - precision_score(self.y[train], y_pred)
                
                self.error_rate[cl].append((i, oob_error))

    def plot_error_rate(self, figdir, fname, sz=13):
        valrange = np.array([v for k,v in self.error_rate.items()])
        #ymax = max(0.3, np.max(valrange[:,:,1]))
        #ymin = min(0.05, np.min(valrange[:,:,1]))
        font = {'family': 'normal',
                'weight': 'normal',
                'size': 18}
        mpl.rc('font', **font)
        plt.figure(figsize=(sz,sz))
        for label, err in self.error_rate.items():
            x = np.array(err)[:,0]
            y = np.array(err)[:,1]
            y[y > 1] = 1
            spl = make_interp_spline(x, y, k=3)
            xnew = np.linspace(x.min(), x.max(), 1000)
            plt.plot(xnew, spl(xnew), label=label)
        #plt.ylim(ymin, ymax)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate: 1 - precision")
        plt.legend(loc="upper right", prop={'size': 13})
        plt.tight_layout()
        if figdir is not None and fname is not None:
            plt.savefig(figdir + fname + ".pdf")

    def log_error_rate(self, outdir=None, fname=None):        
         errors_ = (pd.concat(dict((k,
                                   pd.DataFrame(v, columns=['n', 'oob']))\
                                  for k,v in self.error_rate.items())).
                    reset_index().
                    rename(columns={"level_0": "cvfold"}).
                    drop(columns=['level_1']))
        
         if outdir is not None and fname is not None:
            errors_.to_csv(outdir + fname + '.tsv', sep="\t",
                          index=False)
        
class ObjectiveFun(BasePredictions):
    """
    ObjectiveFun class stores the data for 
    loss function evaluation and computes the objective
    for a given set of hyperparameters
    This is a subclass of BasePredictions class
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.cvlen = dict()

    def set_params(self, **kwargs):
        # for random forest
        self.clf.set_params(**kwargs)
        return self

    """
    return weighted average precision (AP) score
    aggregated by drug class
    """
    def aggregate_precision(self, class_arr, leg_class):
        for cl in class_arr:
            train, test = split_drug_class(X=self.X,
                                           leg_class=leg_class, cl=cl)

            X_test = self.X[test]
            y_test = self.y[test]
            combs_test = self.combs[test]
            
            self.clf.fit(self.X[train], self.y[train])
            probas_ = self.clf.predict_proba(X_test)
            #probas_ = self.clf.fit(self.X[train], self.y[train]).predict_proba(X_test)

            precision, recall, _ = precision_recall_curve(y_test, probas_[:,1])
            average_precision = average_precision_score(y_test, probas_[:,1])

            if not np.any(np.isnan(recall)):
                self.precision[cl] = precision
                self.recall[cl] = recall
                self.avprec[cl] = average_precision
                # number of combnations in each cross-validation fold
                # self.cvlen[cl] = probas_.shape[0]

        # cross validation in each fold
        ap_df =  pd.DataFrame(self.avprec.values(),
                     columns = ['AP'],
                     index = self.avprec.keys())
        return ap_df.T


class MultiClassPredictions(InteractionPredictions):
    """
    Class for making and storing OneVsRest predictions
    of synergies and antagonisms
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.one_vs_rest()

    def one_vs_rest(self):
        self.clf = OneVsRestClassifier(self.clf)

    def crossval_drugclass(self, class_arr, leg_class):
        for cl in class_arr:
            train, test = split_drug_class(X=self.X,
                                           leg_class=leg_class, cl=cl)
            X_test = self.X[test]
            y_test = self.y[test]
            combs_test = self.combs[test]
            
            print("Test set size in %s: %d" % (cl, X_test.shape[0]))
            
            probas_ = self.clf.fit(self.X[train], self.y[train]).predict_proba(X_test)
            # predictions
            pred_df = pd.DataFrame({'comb': combs_test,
                                     'prob_none': probas_[:,0],
                                     'prob_ant': probas_[:,1],
                                    'prob_syn': probas_[:,2]})

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            precision = dict()
            recall = dict()
            average_precision = dict()

            for i in range(3):
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
            self.predicted[cl] = pred_df
            
            try:
                importances = [self.clf.estimators_[i].feature_importances_ for i in range(3)]
                class_names = ['none', 'antagonism', 'synergy']
                imp_list = [pd.DataFrame({'feat': np.argsort(imp)[::-1][:self.top],
                                          'importance': np.sort(imp)[::-1][:self.top],
                                          'type': i})\
                            for imp, i in zip(importances, class_names)]
                imp_df = pd.concat(imp_list, ignore_index=True)
                self.topfeat[cl] = imp_df
            except:
                pass
                

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
        auc_df = (pd.concat({k: pd.DataFrame(v.values(),
                                   index=['AUCROC_none',
                                          'AUCROC_antag',
                                          'AUCROC_syn']).T \
                   for k,v in self.auc.items()}).
         reset_index().rename(columns={"level_0": "cvfold"}).
         drop(columns=["level_1"]))

        ap_df = (pd.concat({k: pd.DataFrame(v.values(),
                                   index=['AP_none',
                                          'AP_antag',
                                          'AP_syn']).T \
                   for k,v in self.avprec.items()}).
         reset_index().rename(columns={"level_0": "cvfold"}).
         drop(columns=["level_1"]))

        metrics = pd.merge(auc_df, ap_df, on='cvfold', how='inner')
        
        if outdir is not None and fname is not None:
            metrics.to_csv(outdir + fname + '.tsv', sep="\t",
                           index=False)

    def _export_rftrees(self):
        pass

    def _export_xgb(self):
        pass


class MultiObjective(ObjectiveFun):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_precision(self, class_arr, leg_class):
        # for each function call initialize a OvR classifier
        clf = OneVsRestClassifier(self.clf)
        
        for cl in class_arr:
            train, test = split_drug_class(X=self.X,
                                           leg_class=leg_class, cl=cl)

            X_test = self.X[test]
            y_test = self.y[test]
            combs_test = self.combs[test]
            
            self.clf.fit(self.X[train], self.y[train])
            probas_ = self.clf.predict_proba(X_test)
            
            #probas_ = clf.fit(self.X[train], self.y[train]).predict_proba(X_test)
            
            precision = dict()
            recall = dict()
            average_precision = dict()

            for i in range(3):
                precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                                    probas_[:, i])
                average_precision[i] = average_precision_score(y_test[:, i], probas_[:, i])
           
            self.precision[cl] = precision
            self.recall[cl] = recall
            self.avprec[cl] = average_precision

        ap_df = (pd.concat({k: pd.DataFrame(v.values(),
                                   index=['AP_none',
                                          'AP_antag',
                                          'AP_syn']).T \
                   for k,v in self.avprec.items()}).
         reset_index().rename(columns={"level_0": "cvfold"}).
         drop(columns=["level_1"]))
        return ap_df
