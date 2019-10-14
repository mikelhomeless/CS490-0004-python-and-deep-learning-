# 5)
#   a)
# Soln:
# Dataset: Car Evaluation Data Set from UC Irvine Machine Learning Repository
#
# Dataset Link: http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
#
# Features:
#
# buying - vhigh, high, med, low
#
# maint - vhigh, high, med, low
#
# doors - 2, 3, 4, 5more
#
# persons - 2, 4, more
#
# lug_boot - small, med, big
#
# safety - low, med, high
#
#
# Target variable:
#
# class values - unacc, acc, good, vgood

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('use_inf_as_na', True)
dataset = pd.read_csv('car.csv', header=None)
dataset.columns = ["buying", "maint", "doors", "persons", "lugBoot", "safety", "classValue"]
# replace anything that is inf to nan
dataset.replace([np.inf, -np.inf], np.nan)
# remove all na
dataset = dataset.dropna()

# change non numerical data into numerical
dataset['buying'] = dataset['buying'].map({'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}).astype(int)
dataset['maint'] = dataset['maint'].map({'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}).astype(int)
dataset['doors'] = dataset['doors'].map({'2': 0, '3': 1, '4': 2, '5more': 3}).astype(int)
dataset['persons'] = dataset['persons'].map({'2': 0, '4': 1, 'more': 2}).astype(int)
dataset['lugBoot'] = dataset['lugBoot'].map({'small': 0, 'med': 1, 'big': 2}).astype(int)
dataset['safety'] = dataset['safety'].map({'low': 0, 'med': 1, 'high': 2}).astype(int)
dataset['classValue'] = dataset['classValue'].map({'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}).astype(int)


# identify correlation of features by doing pivots
for feature in dataset.columns[:-1]:
    yeet = dataset[[feature, 'classValue']].groupby(feature, as_index=False).mean().sort_values(by='classValue', ascending=False)
    print(yeet, end='\n\n\n\n')

# step through each of the features and visualize them against the end state
for feature in dataset.columns[:-1]:
    g = sns.FacetGrid(dataset, col='classValue')
    g.map(plt.hist, feature)
    plt.show()


from sklearn.model_selection import train_test_split
x_train = dataset.drop('classValue', axis=1)
y_train = dataset['classValue']
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4)
# Naive Bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print("GNB classification report:\n", classification_report(y_test, y_pred))

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print("SVM classification report:\n", classification_report(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print("KNN classification report:\n", classification_report(y_test, y_pred))

# it is clear that svm and knn both work better than bayes