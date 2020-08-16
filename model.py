# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd # to read our data set
import numpy as np # to perform numeric operation
import matplotlib.pyplot as plt # for Visuvalise
import seaborn as sns #Visuvalise
import statistics


df=pd.read_csv("C:\\Users\\sivak\\Desktop\\project\\data.csv", encoding='latin1')

df.drop(df.columns[[0,3,5,6,7,8,9,11,12,13,20,23,31,32]],axis=1, inplace=True)
df.drop(df.columns[[18]],axis=1, inplace=True)
df['loan_amnt '] = df['loan_amnt '].astype(float)
def remove_months(string):
  return str(string.strip(' months'))
df['terms'] = df['terms'].apply(remove_months)
df['terms'] = df['terms'].astype(float)

df.annual_inc =df.annual_inc.fillna(df.annual_inc.mean())
df.delinq_2yrs  =df.delinq_2yrs.fillna(df.delinq_2yrs.mean())
df.inq_last_6mths  =df.inq_last_6mths.fillna(df.inq_last_6mths.mean())
df.mths_since_last_delinq =df.mths_since_last_delinq.fillna(df.mths_since_last_delinq.mean())
df.mths_since_last_record =df.mths_since_last_record.fillna(df.mths_since_last_record.mean())
df.numb_credit  =df.numb_credit.fillna(df.numb_credit.mean())
df.total_credits   =df.total_credits.fillna(df.total_credits.mean())
df.collections_12_mths_ex_med     =df.collections_12_mths_ex_med.fillna(df.collections_12_mths_ex_med.mean())
df.mths_since_last_major_derog     =df.mths_since_last_major_derog.fillna(df.mths_since_last_major_derog.mean())
df.tot_colle_amt      =df.tot_colle_amt.fillna(df.tot_colle_amt.mean())
df.tot_curr_bal      =df.tot_curr_bal.fillna(df.tot_curr_bal.mean())
df.acc_now_delinq       =df.acc_now_delinq .fillna(df.acc_now_delinq .mean())

df.isnull().sum()

df_new=df.iloc[:,[10,0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20]]
X = df_new.iloc[:, 1:]
y = df_new.iloc[:, 0]

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#def random_forest_classifier(features, target):
   #clf = RandomForestClassifier()
    #clf.fit(features, target)
   # return clf
clf = RandomForestRegressor(n_estimators = 40,bootstrap=True,verbose=3,max_features="auto",oob_score=True, max_depth=10, random_state = 80)
# Train the model on training data
clf.fit(X, y)
pred=clf.predict(pd.DataFrame(X))

from sklearn import metrics
print(' RMSE', np.sqrt(metrics.mean_squared_error(y, pred)))

accuracy = clf.score(X, y)
print(accuracy)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.30, random_state=80)

#def random_forest_classifier(features, target):
   #clf = RandomForestClassifier()
    #clf.fit(features, target)
   # return clf
clf = RandomForestRegressor(n_estimators = 40,bootstrap=True,verbose=3,max_features="auto",oob_score=True, max_depth=10, random_state = 80)
# Train the model on training data
clf.fit(X_train, y_train)
pred_train=clf.predict(pd.DataFrame(X_train))

from sklearn import metrics
print(' RMSE_Train', np.sqrt(metrics.mean_squared_error(y_train, pred_train)))

accuracy_train = clf.score(X_train, y_train)

print(accuracy_train)

pred_test=clf.predict(pd.DataFrame(X_test))

from sklearn import metrics
print(' RMSE_Test', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))

accuracy_test = clf.score(X_test, y_test)
print(accuracy_test)
from sklearn.metrics import r2_score
from rfpimp import permutation_importances

def r2(clf, X, y):
    return r2_score(y, clf.predict(X))

perm_imp_rfpimp = permutation_importances(clf, X, y, r2)

perm_imp_rfpimp

import pickle
# Saving model to disk
pickle.dump(clf, open('md.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('md.pkl','rb'))
