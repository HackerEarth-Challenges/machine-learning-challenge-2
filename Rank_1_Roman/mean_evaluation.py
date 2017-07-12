
# coding: utf-8

# In[17]:

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score



# In[18]:

class roman_mean:
    def __init__(self, directory, data, target, n_folds_gen, n_folds_sub, seed, sub_seed, ltr,
                 extra_train = None, extra_target = None):
        self.directory = directory
        self.n_folds_gen = n_folds_gen
        self.n_folds_sub = n_folds_sub
        self.seed = seed
        self.sub_seed = sub_seed
        self.ltr = ltr
        self.data = data
        self.target = target
        self.extra_train = extra_train
        self.extra_target = extra_target
    
    def save_in_file(self, data):
        for x in data.columns.values:
            directory = self.directory + '\\features\\' + x
            print(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)
            else:
                print(x + ' already save.')
                continue
            data.loc[:, x].to_csv(directory + '\\' + x + '.csv', index = None, header = True)
    
    #mean_eval + mean_start + cols_mean Computation mean values by target with double cross_validation
    def mean_eval(self, pred, alpha, train_fold, test_fold, target, col_name):
        if type(self.extra_train) == type(None):
            cur_train = train_fold
            cur_target = target
        else:
            cur_train = pd.concat([self.extra_train.loc[:, col_name], train_fold], axis = 0)
            cur_train.index = range(len(cur_train))
            cur_target = pd.concat([self.extra_target, target], axis = 0)
            cur_target.index = range(len(cur_target))
        grouped = cur_target.groupby(cur_train)
        grouped_mean = grouped.mean().to_dict()
        grouped_count = grouped.count().to_dict()
        glob_mean = cur_target.mean()
        pred[list(test_fold.index)] = [(grouped_mean[x] * grouped_count[x] + glob_mean * alpha) / (grouped_count[x] + alpha) 
                                 if x in grouped_mean else glob_mean for x in test_fold]
  
    def mean_start(self, col):
        kf_gen = StratifiedKFold(n_splits=self.n_folds_gen, random_state=self.seed, shuffle=True)
        kf_sub = StratifiedKFold(n_splits=self.n_folds_sub, random_state=self.sub_seed, shuffle=True)
        alpha = 20
        directory = self.directory + '\\features\\' + col.name + '_mean'
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            print(col.name + ' already exist.')
            return
        for i, (train_index, test_index) in enumerate(kf_gen.split(col[:self.ltr], self.target)):
            pred = pd.Series([-1] * len(col))
            sub_col = col[train_index]
            sub_target = self.target[train_index]
            for j, (sub_train_index, sub_test_index) in enumerate(kf_sub.split(sub_col, sub_target)):
                self.mean_eval(pred, alpha,
                          sub_col.iloc[sub_train_index], 
                          sub_col.iloc[sub_test_index], 
                          sub_target.iloc[sub_train_index], col.name)
            self.mean_eval(pred, alpha,
                      col[train_index], 
                      col[test_index], 
                      self.target[train_index], col.name)
            self.mean_eval(pred, alpha, 
                      col[train_index], 
                      col[self.ltr:], 
                      self.target[train_index], col.name)
            pred.name = col.name + '_mean'
            pred.to_csv(self.directory + '\\features\\' + col.name + '_mean' + '\\' + str(i)
                        + '.csv', index = None, header = True)
    
    def cols_mean(self, cols):
        for col in cols:
            print(col)
            self.mean_start(self.data.loc[:, col])

    #Computation factor machine with double cross_validation
    
    #Computation logistic regression with double cross_validaton
    def cols_LR(self, feature_list):
        
        kf_gen = StratifiedKFold(n_splits=self.n_folds_gen, random_state=self.seed, shuffle=True)
        kf_sub = StratifiedKFold(n_splits=self.n_folds_sub, random_state=self.sub_seed, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf_gen.split(self.data[:self.ltr], self.target)):
            print(i)
            sp_data = pd.DataFrame()
            features_directory = self.directory + '\\features'
            col_i = 1
            for feature in feature_list:
                print(str(col_i) + '/' + str(len(feature_list)))
                col_i += 1
                cur_feat_directory = features_directory + '\\' + feature
                if len(os.listdir(cur_feat_directory)) > 1:
                    feature_col = pd.read_csv(cur_feat_directory + '\\' + str(i) + '.csv')
                else:
                    feature_col = pd.read_csv(cur_feat_directory + '\\' + feature + '.csv')
                sp_data = pd.concat([sp_data, feature_col], axis = 1)
                del feature_col
            pred = pd.Series([-1] * len(self.data))
            clf = LogisticRegression(C = 20, n_jobs = -1)
            pred[train_index] = cross_val_predict(clf, sp_data.loc[train_index, :], self.target[train_index], cv = kf_sub, 
                                                  method = 'predict_proba', n_jobs = -1)[:, 1]
            print('OK')
            clf.fit(sp_data.loc[train_index, :], self.target[train_index])
            pred[test_index] = clf.predict_proba(sp_data.loc[test_index, :])[:, 1]
            print(roc_auc_score(self.target[test_index], pred[test_index]))
            pred[self.ltr:] = clf.predict_proba(sp_data.loc[self.ltr:, :])[:, 1]
            pred.name = 'LR_true'
            directory = self.directory + '\\features\\LR_true2\\' 
            if not os.path.exists(directory):
                    os.makedirs(directory)
            pred.to_csv(directory + str(i) + '.csv', index = None, header = True)

    #Computation xgboost predict with double cross_validation         
    def cols_XGB(self, feature_list, dic_par_list, num_round):
        features_directory = self.directory + '//features'

        kf_gen = StratifiedKFold(n_splits=self.n_folds_gen, random_state=self.seed, shuffle=True)
        kf_sub = StratifiedKFold(n_splits=self.n_folds_sub, random_state=self.sub_seed, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf_gen.split(self.data[:self.ltr], self.target)):
            print('Calculate ' + str(i + 1) + '/' + str(self.n_folds_gen))
            data = pd.DataFrame()
            col_i = 1
            for feature in feature_list:
                print(str(col_i) + '/' + str(len(feature_list)))
                col_i += 1
                cur_feat_directory = features_directory + '//' + feature
                if len(os.listdir(cur_feat_directory)) > 1:
                    feature_col = pd.read_csv(cur_feat_directory + '//' + str(i) + '.csv')
                else:
                    feature_col = pd.read_csv(cur_feat_directory + '//' + feature + '.csv')
                data = pd.concat([data, feature_col], axis = 1)
                #print(feature_col.columns)
            del feature_col
            print(i)
            for k, dic_par in enumerate(dic_par_list):
                pred = pd.Series([-1] * len(self.data))
                for j, (sub_train_index, sub_test_index) in enumerate(kf_sub.split(self.data.loc[train_index, :], self.target[train_index])):
                    print(i, k, j)
                    xgall = xgb.DMatrix(data.loc[train_index[sub_train_index], :], self.target[train_index[sub_train_index]])
                    xgeval = xgb.DMatrix(data.loc[train_index[sub_test_index], :], self.target[train_index[sub_test_index]])
                    bst = xgb.train(dic_par, xgall, maximize=True, early_stopping_rounds=20,
                                num_boost_round=num_round, evals=[(xgall, 'train'), (xgeval, 'test')], verbose_eval=False)            
                    pred[train_index[sub_test_index]] = bst.predict(xgeval)
                    del xgall, xgeval, bst
                xgall = xgb.DMatrix(data.loc[train_index, :], self.target[train_index])
                xg_cvtest = xgb.DMatrix(data.loc[test_index, :], self.target[test_index])
                xg_test = xgb.DMatrix(data.loc[self.ltr:, :])
                bst = xgb.train(dic_par, xgall, maximize=True, early_stopping_rounds=20,
                num_boost_round=num_round, evals=[(xgall, 'train'), (xg_cvtest, 'test')], verbose_eval=False) 
                pred[test_index] = bst.predict(xg_cvtest)
                pred[self.ltr:] = bst.predict(xg_test)
                print(bst.best_score)
                print(roc_auc_score(self.target[test_index], pred[test_index]))
                name = 'XGB' + str(k + 3)
                pred.name = name
                directory = self.directory + '//features//' + name + '//' 
                if not os.path.exists(directory):
                        os.makedirs(directory)
                pred.to_csv(directory + str(i) + '.csv', index = None, header = True)
                del xgall, xg_cvtest, xg_test, bst, pred

    #Computation lightGBM with double cross_validaton
                
    #Computation SVD recommends with double cross_validation            
    #Computation xgboost predict    
    def predict(self, dic_par, feature_list, num_round, save = False, fscore = False):
        pred = pd.Series([-1] * len(self.data))
        kf = StratifiedKFold(n_splits=self.n_folds_gen, random_state=self.seed, shuffle=True)
        kf_split = kf.split(self.data.loc[:self.ltr-1, :], self.target)
        features_directory = self.directory + '\\features'
        pred_test = pd.DataFrame()
        score_list = []
        tree_limit = []
        pred_directory = self.directory + '\\models'
        model_number = 100    
        model_directory = pred_directory + '\\model_' + str(model_number)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        for i, (train_index, test_index) in enumerate(kf_split):
            print('Calculate ' + str(i + 1) + '/' + str(self.n_folds_gen))
            data = pd.DataFrame()
            col_i = 1
            for feature in feature_list:
                print(str(col_i) + '/' + str(len(feature_list)))
                col_i += 1
                cur_feat_directory = features_directory + '\\' + feature
                if len(os.listdir(cur_feat_directory)) > 1:
                    feature_col = pd.read_csv(cur_feat_directory + '\\' + str(i) + '.csv')
                else:
                    feature_col = pd.read_csv(cur_feat_directory + '\\' + feature + '.csv')
                data = pd.concat([data, feature_col], axis = 1)
            xgall = xgb.DMatrix(data.loc[train_index, :], self.target[train_index])
            xgeval = xgb.DMatrix(data.loc[test_index, :], self.target[test_index])
            bst = xgb.train(dic_par, xgall, maximize=False, early_stopping_rounds=30,
                            num_boost_round=num_round, evals=[(xgall, 'train'), (xgeval, 'test')], verbose_eval=50)
            del xgall, xgeval
            xg_cvtest = xgb.DMatrix(data.loc[test_index, :])
            xg_test = xgb.DMatrix(data.loc[self.ltr:, :])
            del data
            pred[test_index] = bst.predict(xg_cvtest, ntree_limit=bst.best_ntree_limit)
            score_list += [bst.best_score]
            if fscore == True:
                return bst.get_fscore()
            print(bst.best_score)
            print(bst.best_ntree_limit)
            tree_limit += [bst.best_ntree_limit]
            cur_pred = pd.DataFrame(bst.predict(xg_test, ntree_limit=bst.best_ntree_limit))
            pred_test = pd.concat([pred_test, cur_pred], axis = 1)
            pred.to_csv(model_directory + '\\predict' + str(i) + '.csv')
            pred_test.to_csv(model_directory + '\\pred_test' + str(i) + '.csv')
        pred[self.ltr:] = np.array(pred_test.mean(axis = 1))
        del xg_cvtest, xg_test, bst
        if save == True:
            pred_directory = self.directory + '\\models'
            if not os.path.exists(pred_directory):
                os.makedirs(pred_directory)
            model_number = len(os.listdir(pred_directory)) + 1
            model_directory = pred_directory + '\\model_' + str(model_number)
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            f = open(model_directory + '\\info.txt', 'w')
            f.write('Model_' + str(model_number) + ' info:\n')
            for i, (x, y) in enumerate(zip(score_list, tree_limit)):
                f.write('Fold ' + str(i + 1) + ': Score: ' + str(x) + ' Tree_number: ' + str(y) + '\n')
            f.write('Model score:' + str(1 - np.mean(score_list)))
            f.close()
            pred.to_csv(model_directory + '\\predict.csv')
            del pred
     
    def predictSparse(self, dic_par, sparse, feature_list, num_round, save = False, fscore = False, score = False):
        pred = pd.Series([-1] * len(self.data))
        kf = StratifiedKFold(n_splits=self.n_folds_gen, random_state=self.seed, shuffle=True)
        kf_split = kf.split(self.data.loc[:self.ltr-1, :], self.target)
        features_directory = self.directory + '\\features'
        pred_test = pd.DataFrame()
        score_list = []
        tree_limit = []
        pred_directory = self.directory + '\\models'
        model_number = 100    
        model_directory = pred_directory + '\\model_' + str(model_number)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        for i, (train_index, test_index) in enumerate(kf_split):
            if score == False:
                print('Calculate ' + str(i + 1) + '/' + str(self.n_folds_gen))
            data = pd.DataFrame()
            col_i = 1
            for feature in feature_list:
                if score == False:
                    print(str(col_i) + '/' + str(len(feature_list)))
                col_i += 1
                cur_feat_directory = features_directory + '\\' + feature
                if len(os.listdir(cur_feat_directory)) > 1:
                    feature_col = pd.read_csv(cur_feat_directory + '\\' + str(i) + '.csv')
                else:
                    feature_col = pd.read_csv(cur_feat_directory + '\\' + feature + '.csv')
                data = pd.concat([data, feature_col], axis = 1)
            data = hstack([data, sparse]).tocsr()
            xgall = xgb.DMatrix(data[train_index], self.target[train_index])
            xgeval = xgb.DMatrix(data[test_index], self.target[test_index])
            bst = xgb.train(dic_par, xgall, maximize=False, early_stopping_rounds=50,
                            num_boost_round=num_round, evals=[(xgall, 'train'), (xgeval, 'test')], verbose_eval=20)
            del xgall, xgeval
            xg_cvtest = xgb.DMatrix(data[test_index])
            xg_test = xgb.DMatrix(data[self.ltr:])
            del data
            pred[test_index] = bst.predict(xg_cvtest, ntree_limit=bst.best_ntree_limit)
            score_list += [bst.best_score]
            if fscore == True:
                return bst.get_fscore()
            print(bst.best_score)
            print(bst.best_ntree_limit)
            tree_limit += [bst.best_ntree_limit]
            cur_pred = pd.DataFrame(bst.predict(xg_test, ntree_limit=bst.best_ntree_limit))
            pred_test = pd.concat([pred_test, cur_pred], axis = 1)
            pred.to_csv(model_directory + '\\predict' + str(i) + '.csv')
            pred_test.to_csv(model_directory + '\\pred_test' + str(i) + '.csv')
        pred[self.ltr:] = np.array(pred_test.mean(axis = 1))
        if score == True:
            return accuracy_score(self.target, round(pred[:self.ltr]).astype(int))
        del xg_cvtest, xg_test, bst
        if save == True:
            pred_directory = self.directory + '\\models'
            if not os.path.exists(pred_directory):
                os.makedirs(pred_directory)
            model_number = len(os.listdir(pred_directory)) + 1
            model_directory = pred_directory + '\\model_' + str(model_number)
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            f = open(model_directory + '\\info.txt', 'w')
            f.write('Model_' + str(model_number) + ' info:\n')
            for i, (x, y) in enumerate(zip(score_list, tree_limit)):
                f.write('Fold ' + str(i + 1) + ': Score: ' + str(x) + ' Tree_number: ' + str(y) + '\n')
            f.write('Model score:' + str(1 - np.mean(score_list)))
            f.close()
            pred.to_csv(model_directory + '\\predict.csv')
            del pred

