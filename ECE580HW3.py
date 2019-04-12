import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

traffic_data = np.loadtxt('traffic_uniform.csv', dtype=float, delimiter=',')

task_placement = np.arange(64)

grid = nx.grid_2d_graph(8, 8)
pos = dict(zip(grid, grid))

nx.set_edge_attributes(grid, 1, 'length')

grid.remove_edge((0, 0), (0, 1))
grid.add_edge((0, 0), (6, 7))

link_placement = nx.adjacency_matrix(grid).todense()

nx.draw(grid, pos, with_labels=False)
print(nx.nodes(grid))
plt.show()

print(link_placement)
print(task_placement)

