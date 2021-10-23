# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:38:34 2019

@author: Renhb
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
#import mglearn
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from boruta import BorutaPy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
#######################################################bitter


#data_prepare
bitter_x = pd.read_csv('./data/model_data/bitter_del_NAN.csv')
bitter_y = pd.read_csv('./data/model_data/bitter_all.csv')
bitter_x = bitter_x.drop(['Unnamed: 0'], axis=1)
bitter_y = bitter_y.drop(['Unnamed: 0'], axis=1)


with open('./data/model_data/bitter_rf_feature.p', 'rb') as file:
    bitter_rf_feature = pickle.load(file)
bitter_feature = bitter_rf_feature#.tolist() 
#pre-processing
bitter = bitter_x[bitter_feature]
bitter = {'data':bitter, 'target':bitter_y.bitter.values}
X_train,X_test,y_train, y_test = train_test_split(bitter['data'], bitter['target'], 
                                                  stratify=bitter['target'], 
                                                  test_size = .25, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#model
mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(100, 100, 200), 
                    random_state = 0, alpha=0.2).fit(X_train, y_train)
print(classification_report(y_test, mlp.predict(X_test)))
print('Accuracy on test set: {:.3f}'.format(mlp.score(X_test, y_test)))
rf= RandomForestClassifier(n_estimators = 50, n_jobs=-1, 
                                class_weight='balanced', 
                                max_depth =5, random_state = 0).fit(X_train, y_train)
#model_result

#print(classification_report(y_test, rf.predict(X_test)))
#print('Accuracy on test set: {:.3f}'.format(rf.score(X_test, y_test)))

with open('./data/model_data/bitter_mlp_model.p', 'wb') as file:
    pickle.dump(mlp, file)
with open('./data/model_data/bitter_rf_model.p', 'wb') as file:
    pickle.dump(rf, file)

with open('./data/model_data/bitter_scaler.p', 'wb') as file:
    pickle.dump(scaler, file)


######################################################
y_score = mlp.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1]);
roc_auc = auc(fpr, tpr)
#right_index = (tpr + (1 - fpr) - 1)
#yuzhi = max(right_index)
#index = right_index.index(max(right_index))
#tpr_val = tpr(index)
#fpr_val = fpr(index)
## 绘制roc曲线图
 
fig,ax = plt.subplots(figsize=(7,5.5));
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %1f)' % roc_auc);
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('ROC Curve');
plt.legend(loc="lower right");
plt.show()
print(confusion_matrix(y_test, mlp.predict(X_test), labels=[0, 1]))
fig.savefig("./mlp_b.pdf",dpi=600,format="pdf")


    
    
    
    
    
    
    
######################################################sweet
#data_prepare
sweet_x = pd.read_csv('./data/model_data/sweet_del_NAN.csv')
sweet_y = pd.read_csv('./data/model_data/sweet_all.csv')
sweet_x = sweet_x.drop(['Unnamed: 0'], axis=1)
sweet_y = sweet_y.drop(['Unnamed: 0'], axis=1)
with open('./data/model_data/sweet_rf_feature.p', 'rb') as file:
    sweet_rf_feature = pickle.load(file)
sweet_feature = sweet_rf_feature#.tolist() 
sweet = sweet_x[sweet_feature]
#sweet_feature = sweet_x.columns.values#.tolist() 
#pre-processing

sweet = {'data':sweet, 'target':sweet_y.sweet.values}
X_train,X_test,y_train, y_test = train_test_split(sweet['data'], sweet['target'], 
                                                  stratify=sweet['target'],
                                                  test_size = .25, random_state=2)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#model
mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(10,10,10), 
                    random_state = 0, alpha=0.2).fit(X_train, y_train)
print(classification_report(y_test, mlp.predict(X_test)))
print('Accuracy on test set: {:.3f}'.format(mlp.score(X_test, y_test)))
print(confusion_matrix(y_test, mlp.predict(X_test), labels=[0, 1]))
rf = RandomForestClassifier(n_estimators = 50, n_jobs=-1, 
                                class_weight='balanced', 
                                max_depth =5, random_state = 0).fit(X_train, y_train)
#model_result
#print(classification_report(y_test, rf.predict(X_test)))
#print('Accuracy on test set: {:.3f}'.format(rf.score(X_test, y_test)))

with open('./data/model_data/sweet_mlp_model.p', 'wb') as file:
    pickle.dump(mlp, file)
with open('./data/model_data/sweet_rf_model.p', 'wb') as file:
    pickle.dump(rf, file)

with open('./data/model_data/sweet_scaler.p', 'wb') as file:
    pickle.dump(scaler, file)










## 绘制roc曲线图
#y_score = mlp.predict_proba(X_test)
#fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1]);
#roc_auc = auc(fpr, tpr)
#plt.subplots(figsize=(7,5.5));
#plt.plot(fpr, tpr, color='darkorange',
         #lw=2, label='ROC curve (area = %1f)' % roc_auc);
#plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--');
#plt.xlim([0.0, 1.0]);
#plt.ylim([0.0, 1.05]);
#plt.xlabel('False Positive Rate');
#plt.ylabel('True Positive Rate');
#plt.title('ROC Curve');
#plt.legend(loc="lower right");
#plt.show()
#from sklearn.metrics import roc_curve, auc
y_score = mlp.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1]);
roc_auc = auc(fpr, tpr)
#right_index = (tpr + (1 - fpr) - 1)
#yuzhi = max(right_index)
#index = right_index.index(max(right_index))
#tpr_val = tpr(index)
#fpr_val = fpr(index)
## 绘制roc曲线图
fig,ax = plt.subplots(figsize=(7,5.5));#plt.subplots(figsize=(7,5.5));
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %1f)' % roc_auc);
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('ROC Curve');
plt.legend(loc="lower right");
plt.show()
fig.savefig("./mlp_s.pdf",dpi=600,format="pdf")
class Model():
    def __init__(self, bitter_model_path, bitter_features_path, sweet_model_path, sweet_features_path):
        self.bitter_model = pickle.load(open(bitter_model_path, 'rb'))
        self.bitter_features = pickle.load(open(bitter_features_path, 'rb'))
        self.sweet_model = pickle.load(open(sweet_model_path, 'rb'))
        self.sweet_features = pickle.load(open(sweet_features_path, 'rb'))

    def predict_bitter(self, data):
        # Subset to relevant columns
        d = data[self.bitter_features]

        # Predict
        bitter_prob = self.bitter_model.predict_proba(d)
        bitter_taste = self.bitter_model.predict(d)
        
        return bitter_prob, bitter_taste
        
    def predict_sweet(self, data):
        # Subset to relevant columns
        d = data[self.sweet_features]

        # Predict
        sweet_prob = self.sweet_model.predict_proba(d)
        sweet_taste = self.sweet_model.predict(d)

        return sweet_prob, sweet_taste
        
    def predict(self, data):
        bitter_prob, bitter_taste = self.predict_bitter(data)
        sweet_prob, sweet_taste = self.predict_sweet(data)
        
        return bitter_taste, bitter_prob, sweet_taste, sweet_prob