# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:18:37 2019

@author: Renhb
"""
import pickle
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

###############################################################################
###############################################################################
###############################################################################
#bitter_boruta
bitter_x = pd.read_csv('./data/model_data/bitter_del_NAN.csv')
bitter_y = pd.read_csv('./data/model_data/bitter_all.csv')
bitter_x = bitter_x.drop(['Unnamed: 0'], axis=1)
bitter_y = bitter_y.drop(['Unnamed: 0'], axis=1)
bitter = {'data':bitter_x.values, 'target':bitter_y.bitter.values}
#bitter = pd.DataFrame(bitter)
X_train,X_test,y_train, y_test = train_test_split(bitter['data'], bitter['target'], 
                                                  test_size = .25, stratify=bitter['target'],
                                                  random_state=0)
forest = RandomForestClassifier(n_estimators = 100, n_jobs=-1, class_weight='balanced', max_depth =5, random_state = 0)
#define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
#find all relevant features -5 features should be selected
feat_selector.fit(X_train, y_train)
#check selected features - first 5 features are selected
feat_selector.support_
#del 5 features of X_train(colnum)
#check ranking of features
feat_selector.ranking_
#bitter_feature = pd.read_csv('D:/jupyter notebook/model_bitter_sweet/bitter/all/bitter_del_feature.csv')
bitter_select_feature = {'bitter_feature':bitter_x.columns.values, 
                         'feature_importance':feat_selector.support_}
bitter_select_feature = pd.DataFrame(bitter_select_feature)
bitter_boruta = bitter_select_feature[bitter_select_feature.feature_importance == True]
bitter_rf_feature = list(bitter_boruta.bitter_feature.values)
with open('./data/model_data/bitter_rf_feature.p', 'wb') as file:
    pickle.dump(bitter_rf_feature, file)
#bitter_boruta.to_csv('bitter_rf_feature.csv')
###############################################################################
###############################################################################
###############################################################################






###############################################################################
###############################################################################
###############################################################################
#sweet boruta
sweet_x = pd.read_csv('./data/model_data/sweet_del_NAN.csv')
sweet_y = pd.read_csv('./data/model_data/sweet_all.csv')
sweet_x = sweet_x.drop(['Unnamed: 0'], axis=1)
sweet_y = sweet_y.drop(['Unnamed: 0'], axis=1)
sweet = {'data':sweet_x.values, 'target':sweet_y.sweet.values}
#sweet = pd.Dataframe(sweet)
X_train,X_test,y_train, y_test = train_test_split(sweet['data'], sweet['target'],
                                                  test_size = .25, stratify = sweet['target'],
                                                  random_state=0)
forest = RandomForestClassifier(n_estimators = 100, n_jobs=-1, class_weight='balanced', max_depth =5, random_state = 0)
#define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
#find all relevant features -5 features should be selected
feat_selector.fit(X_train, y_train)
#check selected features - first 5 features are selected
feat_selector.support_
#del 5 features of X_train(colnum)
#check ranking of features
feat_selector.ranking_
#sweet_feature = pd.read_csv('D:/jupyter notebook/model_bitter_sweet/sweet/all/sweet_del_feature.csv')
sweet_select_feature = {'sweet_feature':sweet_x.columns.values,
                        'feature_importance':feat_selector.support_}
sweet_select_feature = pd.DataFrame(sweet_select_feature)
sweet_boruta = sweet_select_feature[sweet_select_feature.feature_importance == True]
sweet_rf_feature = list(sweet_boruta.sweet_feature.values)
with open('./data/model_data/sweet_rf_feature.p', 'wb') as file:
    pickle.dump(sweet_rf_feature, file)
#sweet_boruta.to_csv('D:/jupyter notebook/model_bitter_sweet/sweet/all/sweet_rf_feature.csv')
###############################################################################
###############################################################################
###############################################################################