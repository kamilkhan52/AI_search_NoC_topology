import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import networkx as nx
import numpy as np

# load dataset
wine = pd.read_csv('winequality-red.csv', sep=';')

X = wine.iloc[:, 1: 11]
Y = wine.iloc[:, 11]

x_train, x_test, y_train, y_test = train_test_split(X, Y)

clf = svm.SVR(gamma='auto')
clf.fit(X, Y)

read_data = pd.read_csv('export.csv', sep=',')
print(len(read_data))
features = read_data.iloc[:, 0: 4160]
label = read_data.iloc[:, 4160]

x_train, x_test, y_train, y_test = train_test_split(features, label)

clf = svm.SVR(gamma='auto')
clf.fit(features, label)

print(clf.score(x_test, y_test))
print(clf.predict(x_test.iloc[10:11, :]))
print(y_test.iloc[11])
