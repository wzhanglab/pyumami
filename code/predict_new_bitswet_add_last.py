# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:24:38 2021

@author: Renhb
"""

#苦味模型预测甜味分子，甜味模型预测苦味分子，取并集
import pickle
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import mglearn
#import pickle
#from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
#from sklearn.metrics import classification_report
#from boruta import BorutaPy
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import roc_curve, auc

with open('./data/model_data/bitter_mlp_model.p', 'rb') as file:
    bitter_mlp_model = pickle.load(file)
with open('./data/model_data/bitter_rf_model.p', 'rb') as file:
    bitter_rf_model = pickle.load(file)
with open('./data/model_data/bitter_rf_feature.p', 'rb') as file:
    bitter_rf_feature = bitter_rf_feature = pickle.load(file)
with open('./data/model_data/bitter_scaler.p', 'rb') as file:
    bitter_scaler = pickle.load(file)

with open('./data/model_data/sweet_mlp_model.p', 'rb') as file:
    sweet_mlp_model = pickle.load(file)
with open('./data/model_data/sweet_rf_model.p', 'rb') as file:
    sweet_rf_model = pickle.load(file)
with open('./data/model_data/sweet_rf_feature.p', 'rb') as file:
    sweet_rf_feature = pickle.load(file)
with open('./data/model_data/sweet_scaler.p', 'rb') as file:
    sweet_scaler = pickle.load(file)
    
bitswet_data = pd.read_csv('./data/data_calibration/calibration/last/bitswet_descriptors.csv',engine='python')
bitswet_data = bitswet_data.drop(['Unnamed: 0'], axis=1)
bitswet_all = pd.read_csv('./data/data_calibration/calibration/last/bitswet_all.csv',engine='python')
bitswet_all = bitswet_all.drop(['Unnamed: 0'], axis=1)
#kinds=1
#bitswet_data_bit = bitswet_data[0:969]
#bitswet_all_bit = bitswet_all[0:969]
#kinds=2
#bitswet_data_sweet = bitswet_data[969:]
#bitswet_all_sweet = bitswet_all[969:]


bit_data = bitswet_data[sweet_rf_feature]
bit_data = bit_data.apply(pd.to_numeric, errors = 'coerce')
null_values = bit_data.isnull().sum(axis=1)
null_rows = bit_data.loc[null_values > 0]
# Drop null rows
bit_data = bit_data.loc[null_values == 0]
#Drop rawdata including null_rows
bit_all = bitswet_all.drop(index=null_rows._stat_axis.values.tolist())
bit_data = sweet_scaler.transform(bit_data)

###swet_data
swet_data = bitswet_data[bitter_rf_feature]
swet_data = swet_data.apply(pd.to_numeric, errors = 'coerce')
null_values = swet_data.isnull().sum(axis=1)
null_rows = swet_data.loc[null_values > 0]
# Drop null rows
swet_data = swet_data.loc[null_values == 0]
#Drop rawdata including null_rows
swet_all = bitswet_all.drop(index=null_rows._stat_axis.values.tolist())
swet_data = bitter_scaler.transform(swet_data)







###predict
bitswet_pre_pred = bitter_mlp_model.predict(np.array(swet_data))
bitswet_pre_pred = pd.DataFrame(bitswet_pre_pred)
bitswet_pre_pred_prob = bitter_mlp_model.predict_proba(np.array(swet_data))
bitswet_pre_pred_prob = pd.DataFrame(bitswet_pre_pred_prob)
a = list(bitswet_pre_pred_prob.loc[:,1])
b = list(bitswet_pre_pred_prob.loc[:,0])
bitswet_pre_predict_bitter = {'smiles':swet_all.bitswet_smiles.values, 
               'cid':swet_all.cid.values,
               'bitter':np.array(list(bitswet_pre_pred.iloc[:,0])), 
               'bitter_prob':np.array(a), 'non_bitter_prob':np.array(b)}
bitswet_pre_predict_bitter = pd.DataFrame(bitswet_pre_predict_bitter)
print(bitswet_pre_predict_bitter)
bitswet_pre_mlp_bitter_prob = bitswet_pre_predict_bitter.astype({'bitter_prob':'float'})
bitswet_bitter_prob = bitswet_pre_predict_bitter[(bitswet_pre_predict_bitter["bitter_prob"] >= 0.50)]

bitswet_pre_pred_s = sweet_mlp_model.predict(np.array(bit_data))
bitswet_pre_pred_s = pd.DataFrame(bitswet_pre_pred_s)
bitswet_pre_pred_prob_s = sweet_mlp_model.predict_proba(np.array(bit_data))
bitswet_pre_pred_prob_s = pd.DataFrame(bitswet_pre_pred_prob_s)
a = list(bitswet_pre_pred_prob_s.loc[:,1])
b = list(bitswet_pre_pred_prob_s.loc[:,0])
bitswet_pre_predict_sweet = {'smiles':bit_all.bitswet_smiles.values, 
               'cid':bit_all.cid.values,
               'sweet':np.array(list(bitswet_pre_pred_s.iloc[:,0])), 
               'sweet_prob':np.array(a), 'non_sweet_prob':np.array(b)}
bitswet_pre_predict_sweet = pd.DataFrame(bitswet_pre_predict_sweet)
print(bitswet_pre_predict_sweet)
bitswet_pre_mlp_sweet_prob = bitswet_pre_predict_sweet.astype({'sweet_prob':'float'})
bitswet_sweet_prob = bitswet_pre_predict_sweet[(bitswet_pre_predict_sweet["sweet_prob"] >= 0.50)]
bitswet_result = pd.concat([bitswet_pre_predict_bitter,
                            bitswet_pre_predict_sweet.loc[:,['sweet','sweet_prob','non_sweet_prob']]],
                           axis=1,ignore_index=False)

bitswet = bitswet_result.astype({'bitter_prob':'float','sweet_prob':'float'})

bitswet_prob = bitswet[(bitswet['bitter_prob'] >= 0.50)&(bitswet['sweet_prob'] >= 0.50)]
bitswet_prob.to_csv('./data/data_calibration/calibration/last/bitswet_mlp_prob.csv')
bitswet.to_csv('./data/data_calibration/calibration/last/bitswet_mlp_prob_all(0-1).csv')