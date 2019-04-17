import pandas as pd
import seaborn as sns
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

# load dataset
wine = pd.read_csv('winequality-red.csv', sep=';')

X = wine.iloc[:, 1: 11]
Y = wine.iloc[:, 11]

x_train, x_test, y_train, y_test = train_test_split(X, Y)

clf = svm.SVR(gamma='auto')
clf.fit(X, Y)

print(clf.score(x_test, y_test))

grid = nx.grid_2d_graph(8, 8)
pos = dict(zip(grid, grid))
nx.set_edge_attributes(grid, 1, 'length')
nx.draw(grid, pos)
nx.draw_networkx_labels(grid, pos, font_size=6)
# plt.pause(.00005)
# plt.show()

nx.write_graphml(grid, "test.gml")
adj_mat = nx.adjacency_matrix(grid)
print(grid.edges)
print(adj_mat)
