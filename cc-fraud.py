#Improting Librarires

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

import os
import warnings
warnings.filterwarnings('ignore')

# Getting current working directory

os.getcwd()

#Datset is already present in current directory so no need to change directory

#---

#Loading the Data

data = pd.read_csv("creditcard.csv")

#Exploring the data

data

#This data consists of 284807 records (rows)

data.head(6)

#Checking if dataset has any null or 'Nan' values

data.isnull().values.any()

# Getting statistical description of data

data.describe()

#Getting data info

data.info()

#Setting X and y variables for deploying the Machine Learning CLassification Algorithms

X = data.iloc[ : , 1:30].values
y = data.iloc[ : , -1].values

y

#Checking if both variables imported correct number of records

X.shape

y.shape

#Visually analyzing the number of Fraud Transactions vs. Normal Transactions

plt.rcParams['figure.figsize'] = (15,6)

data_class = pd.value_counts(data['Class'], sort = True)
data_class.plot(kind = 'bar', rot = 1)
plt.xticks(range(2))
plt.xlabel("Class", size = 18, color = "Red", fontname = "Verdana")
plt.ylabel("Number of Records", size = 18, color = "Red", fontname = "Verdana")
plt.show()

#Getting number of Normal Transactions

len(data[data.Class == 0])

#Getting counts of Fraud transactions

len(data[data.Class == 1])

#Checking if no records are lost // Must macth with len(X)

len(data[data.Class == 0]) + len(data[data.Class == 1])

#---

#Finding coorelation between the attributes (columns) by plotting the Seaborn Heatmap

plt.rcParams['figure.figsize'] = (20,15)

correl = data.corr()
g=sns.heatmap(correl, square = True, cmap = 'YlOrRd', vmax = 0.9)
plt.show()

#Thus, none of the atrributes are correalted to each other excpet with self.

#---

#Creating train and test set to deploy Machine Learning Algorithms

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Standardization of X_train and X_test to avoid abnormal behavior of model

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)

X_train

#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train, y_train)

#Predicting results based on classifier being fitted above

y_pred_dt = classifier_dt.predict(X_test)
y_pred_dt

#To check the performance of algorithm, we will use Confusion Matrix
#It is a matrix having True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN) values as elements

#Building Confusion Matrix for Decision Trees model

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test, y_pred_dt)
cm1

#Model Performance Evaluation for Support Vector Machine

print("Decision Trees Model Performance Evaluation Meausres \n")

accuracy_dt = accuracy_score(y_test,y_pred_dt) * 100
print("Accuracy (in percent)   :", np.round(accuracy_dt, 2))

error_dt = 100 - accuracy_dt
print('Error Rate (in percent) :', np.round(error_dt, 2))

precision_dt = precision_score(y_test,y_pred_dt) * 100
print("Precision (in percent)  :", np.round(precision_dt, 2))

recall_dt = recall_score(y_test, y_pred_dt) * 100
print("Recall (in percent)     :", np.round(recall_dt, 2))

#---

#Predicting the X_test value with another Classification Algorithm Support Vector Machine
#This is done to check if SVM produces better Evaluations reuslts or not

#Support Vector Machine

#Fitting the Support Vector Machine classifier on X_train and y_train 

from sklearn.svm import SVC

svc_classifier = SVC(kernel = "rbf", random_state = 0)
svc_classifier.fit(X_train, y_train)

#Predicting results based on classifier being fitted above

y_pred_svm = svc_classifier.predict(X_test)
y_pred_svm

#Building Confusion Matrix for Support Vector Machine model

cm2 = confusion_matrix(y_test, y_pred_svm)
cm2

#Model Performance Evaluation for Support Vector Machine

print("Support Vector Machine Model Performance Evaluation Meausres \n")

accuracy_svm = accuracy_score(y_test,y_pred_svm) * 100
print("Accuracy (in percent)   :", np.round(accuracy_svm, 2))

error_svm = 100 - accuracy_svm
print('Error Rate (in percent) :', np.round(error_svm, 2))

precision_svm = precision_score(y_test,y_pred_svm) * 100
print("Precision (in percent)  :", np.round(precision_svm, 2))

recall_svm = recall_score(y_test, y_pred_svm) * 100
print("Recall (in percent)     :", np.round(recall_svm, 2))

#---

#Thus, based on above 2 evaluations of both models, both perfomed equally well with Decision Tree having slightly higher performance.
