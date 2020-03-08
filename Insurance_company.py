# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:09:01 2020

@author: Aqsa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from scipy import stats
from statsmodels.stats import weightstats as stests
from sklearn.model_selection import train_test_split
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import svm
from sklearn.metrics import r2_score

#Read data sets
HastingDirect = pd.read_csv('C:/Users/Aqsa/Desktop/Job Hunt/Presentation-Hastings/DS_Assessment.csv')

#summary of numeric and descriptive data
HastingDirect.shape
HastingDirect.describe()
HastingDirect.describe(include=['O'])
HastingDirect.info()
HastingDirect.isnull().sum()
HastingDirect.head()

#Summary of data using descriptive data mean
HastingDirect.Sale.value_counts()
HastingDirect.groupby('Sale').mean()
HastingDirect.groupby('Marital_Status').mean()
HastingDirect.groupby('Payment_Type').mean()

#Histogram of indivual feature
#histogram
HastingDirect.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
plt.show()

#relationship between Variables
sns.pairplot(HastingDirect)

#Sales plot
sns.countplot(x='Sale', data = HastingDirect)

#Sales vs independent variables boxplots
i = HastingDirect[['Age' , 'Price', 'Tax' , 'Veh_Mileage' , 'Credit_Score' , 'License_Length' , 'Sale' ]]
for k in i.columns:
    sns.boxplot( x='Sale',y=k , data = i)
    plt.show()
    
#Sales vs Barplots of numeric variables with bins 
j = HastingDirect[['Age' , 'Price', 'Tax' , 'Veh_Mileage' , 'Credit_Score' , 'License_Length' , 'Sale']]
for l in j.columns:
    j[l] = pd.qcut(j[l], 4)
    sns.barplot( x='Sale',y=l , data = j)
    plt.show()

#Number of people who bought policy vs who didn't
bought_policy = len(HastingDirect[HastingDirect['Sale'] == 1])
didnot_bought_policy = len(HastingDirect[HastingDirect['Sale'] == 0])
print('The number of Customers who bought the policy :  %i (%.1f%%)'% (bought_policy, (bought_policy)/len(HastingDirect)*100 ))
print('The number of Customers who did not buy the policy: %i (%.1f%%)'% (didnot_bought_policy,  (didnot_bought_policy)/len(HastingDirect)*100 ) )

#Marital_Status vs Sales
HastingDirect.Marital_Status.value_counts()
HastingDirect.groupby('Marital_Status')['Sale'].apply(lambda x: (x == 1).sum()).reset_index(name = 'Sale')
HastingDirect[['Marital_Status', 'Sale']].groupby(['Marital_Status'], as_index=False).mean()
HastingDirect['Did not buy'] = 1- HastingDirect['Sale']
HastingDirect.groupby('Marital_Status').agg('mean')[['Sale', 'Did not buy']].plot(kind='bar', figsize=(15, 7),
                                                          stacked=True, color=['g', 'r']);
sns.barplot( x='Marital_Status',y='Sale', data = HastingDirect)
sns.barplot( x='Marital_Status',y='Price', data = HastingDirect)                     

#Payment Type vs Sales
HastingDirect.Payment_Type.value_counts()
print(' who bought the policy:', HastingDirect.groupby('Payment_Type').Sale.value_counts())
HastingDirect[['Payment_Type', 'Sale']].groupby(['Payment_Type'], as_index=False).mean()   
HastingDirect.groupby('Payment_Type').agg('mean')[['Sale', 'Did not buy']].plot(kind='bar', figsize=(15, 7),
                                                          stacked=True, colors=['g', 'r']); 
sns.barplot( x='Payment_Type',y='Sale', data = HastingDirect)
sns.barplot( x='Payment_Type',y='Price', data = HastingDirect)
HastingDirect.drop('Did not buy', axis = 1, inplace = True)

#Marital vs Sales vs Payment_Type
sns.factorplot('Marital_Status', 'Sale', hue='Payment_Type',kind = 'bar', size=4, aspect=2, data=HastingDirect)
tab = pd.crosstab(HastingDirect['Payment_Type'], HastingDirect['Marital_Status'])
print (tab)
tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Payment Type')
plt.ylabel('Percentage')

#Comparison of Price effect on Sale
figure = plt.figure(figsize=(25, 7))
plt.hist([HastingDirect[HastingDirect['Sale'] == 1]['Price'], HastingDirect[HastingDirect['Sale'] == 0]['Price']], 
         stacked=False, color = ['g','r'],
         bins = 50, label = ['Bought_Policy','Didnt buy policy'])
plt.xlabel('Price')
plt.ylabel('Sale Policies')
plt.legend();

#Vehicle Registration Date vs Sales
plt.figure(figsize=(20, 7))
sns.set(style="whitegrid")
ax = sns.barplot(x="Veh_Reg_Year", y="Sale", data=HastingDirect)


#Heatmap Correlation                      
plt.figure(figsize=(25,11))
ax = sns.heatmap(HastingDirect.corr(), vmax=0.8, square=True, annot_kws={'size':10}, annot=True)
ax.get_ylim()
(5.5, 0.5)
ax.set_ylim(9.0, 0)                      
#########Data cleaning###########

#Missing data percentages
total = HastingDirect.isnull().sum().sort_values(ascending=False)
percent = (HastingDirect.isnull().sum()/HastingDirect.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data.head()

#Replace msissing values of Payment type with the mode of the column
Missing_Payment_Type= HastingDirect['Payment_Type'].value_counts().index[0]
HastingDirect.loc[:,'Payment_Type'] = HastingDirect['Payment_Type'].fillna(Missing_Payment_Type)

#Filling missing value of Marital_Status with mode of column
Missing_Marital_Status= HastingDirect['Marital_Status'].value_counts().index[0]
HastingDirect.loc[:,'Marital_Status'] = HastingDirect['Marital_Status'].fillna(Missing_Marital_Status)

#Filling missing value of Date with mode of column
Missing_Date= HastingDirect['Date'].value_counts().index[0]
HastingDirect.loc[:,'Date'] = HastingDirect['Date'].fillna(Missing_Date)

#Replacing rest of continous variables missing values with mean of the column 
HastingDirect.median()
HastingDirect.fillna(HastingDirect.median(), inplace = True)


#######Feature Engineering##########
###Converting catagorical data to numeric#####
#Make Sperate columns of Marital Status as M,S,D using one hot encoding 
Marital_dummies = pd.get_dummies(HastingDirect.Marital_Status)
HastingDirect = pd.concat([HastingDirect,Marital_dummies],axis = 1)

#Make Sperate columns of Payment Type as Installments and Cash using one hot encoding 
Payment_dummies = pd.get_dummies(HastingDirect.Payment_Type)
HastingDirect = pd.concat([HastingDirect,Payment_dummies],axis = 1)

#Splitting Date to Year,Month and Day
HastingDirect.loc[:,'Date'] = pd.to_datetime(HastingDirect.Date)
HastingDirect['Year'] = HastingDirect['Date'].dt.year
HastingDirect['Month'] = HastingDirect['Date'].dt.month
HastingDirect['Day'] = HastingDirect['Date'].dt.day


##Creating Age bins
HastingDirect.loc[ HastingDirect['Age'] <= 17, 'Age'] = 0
HastingDirect.loc[(HastingDirect['Age'] > 17) & (HastingDirect['Age'] <= 30), 'Age'] = 1
HastingDirect.loc[(HastingDirect['Age'] > 30) & (HastingDirect['Age'] <= 43), 'Age'] = 2
HastingDirect.loc[(HastingDirect['Age'] > 43) & (HastingDirect['Age'] <= 55), 'Age'] = 3
HastingDirect.loc[(HastingDirect['Age'] > 55) & (HastingDirect['Age'] <= 68), 'Age'] = 4
HastingDirect.loc[(HastingDirect['Age'] > 68) & (HastingDirect['Age'] <= 80), 'Age'] = 5
HastingDirect.loc[(HastingDirect['Age'] > 80) & (HastingDirect['Age'] <= 93), 'Age'] = 6
HastingDirect.loc[ HastingDirect['Age'] > 93, 'Age'] = 7

#Drop Unwanted catagorical data
HastingDirect.drop(['Marital_Status'], axis=1, inplace = True)
HastingDirect.drop(['Payment_Type'], axis=1, inplace = True)
HastingDirect.drop(['Date'], axis=1, inplace = True)

#Split data in in independant variables and target variable

X = HastingDirect.drop('Sale',axis = 1)
Y = HastingDirect['Sale']

#Cross Validation
skfold = StratifiedKFold(n_splits=10, random_state=100)
model_skfold = LogisticRegression()
results_skfold = model_selection.cross_val_score(model_skfold, X, Y, cv=skfold)
print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))

## Replacing outliers with 5th and 95th quantile values
X.clip(lower=X.quantile(0.05), upper=X.quantile(0.95), axis = 1, inplace = True)


#Split data in 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

##Standardize data
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std  = std_scale.transform(X_test)


####Running Models####
#Logistic Regression
clf = LogisticRegression()
trained_model = clf.fit(X_train_std, y_train)
trained_model.fit(X_train_std, y_train)
y_pred_log_reg = trained_model.predict(X_test_std)
acc_log_reg = round( trained_model.score(X_train_std, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent')

Trained = round(accuracy_score(y_train, trained_model.predict(X_train_std))*100,2)
Test =  round(accuracy_score(y_test, y_pred_log_reg)*100,2)
print ("Accuracy of Train Data: %i %% \n"%Trained)
print ("Accuracy of Test Data: %i %% \n" %Test)

print(r2_score(trained_model.predict(X_train_std), y_train))
report = classification_report(y_test, y_pred_log_reg)
print(report)

#Random Forest
clf = RandomForestClassifier(n_estimators=100,
                                 class_weight="balanced",
                                 criterion='gini',
                                 bootstrap=True,
                                 max_features=0.7,
                                 min_samples_split=3,
                                 min_samples_leaf=5,
                                 max_depth=100,
                                 n_jobs=1)
clf.fit(X_train_std, y_train)
y_pred_random_forest_training_set = clf.predict(X_test_std)
acc_random_forest = round(clf.score(X_train_std, y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_random_forest)

Trained = round(accuracy_score(y_train, trained_model.predict(X_train_std))*100,2)
Test =  round(accuracy_score(y_test, y_pred_random_forest_training_set)*100,2)
print ("Accuracy of Train Data: %i %% \n"%Trained)
print ("Accuracy of Test Data: %i %% \n" %Test)


####Model Evaluation######
#Confusion Matrix

class_names = ['Buy', 'Did not Buy']
cnf_matrix = confusion_matrix(y_test, y_pred_log_reg)
np.set_printoptions(precision=2)

conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred_log_reg)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)

#Precision, Recall and F1 Score
print(classification_report(y_test, y_pred_log_reg))

#ROC Curve
logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test_std))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test_std)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()




