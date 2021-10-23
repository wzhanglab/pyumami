# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:28:43 2019

@author: Renhb
"""

#import numpy as np
import pandas as pd
data = pd.read_csv('./data/model_data/sweet_descriptors.csv')
data1 = pd.read_csv('./data/model_data/bitter_descriptors.csv')
data = data.drop(['Unnamed: 0'], axis=1)
data1 = data1.drop(['Unnamed: 0'], axis=1)
#sweet del NAN
data_1 = data.apply(pd.to_numeric, errors = 'coerce')
data_1.to_csv('./data/model_data/sweet_NAN.csv')
data_1.isnull().sum()
sweet_del_NAN = data_1.dropna(axis=1)
sweet_del_NAN.to_csv('./data/model_data/sweet_del_NAN.csv')


#bitter del NAN
data_11 = data1.apply(pd.to_numeric, errors = 'coerce')
data_11.to_csv('./data/model_data/bitter_NAN.csv')
data_11.isnull().sum()
bitter_del_NAN = data_11.dropna(axis=1)
bitter_del_NAN.to_csv('./data/model_data/bitter_del_NAN.csv')