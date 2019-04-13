import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math as math


def get_initial_topology():
    global task_placement
    global grid
    global pos
    global link_placement
    global traffic_data_dict_square

    task_placement = np.arange(64)
    grid = nx.grid_2d_graph(8, 8)
    pos = dict(zip(grid, grid))
    nx.set_edge_attributes(grid, 1, 'length')
    link_placement = nx.adjacency_matrix(grid).todense()
    # format traffic data into a dictionary
    traffic_data_dict = {node: 0 for node in grid}
    print("traffic_data_dict=" + str(len(traffic_data_dict)) + str(traffic_data_dict))
    for key in traffic_data_dict:
        for key2 in traffic_data_dict:
            traffic_data_dict_square[key + key2] = traffic_data[key[0]][key2[1]]
    print("traffic_data_dict_square=" + str(len(traffic_data_dict_square)) + str(traffic_data_dict_square))


def draw_topology(grid, pos):
    nx.draw(grid, pos, with_labels=False)
    print(nx.nodes(grid))
    plt.show(block=False)
    plt.show()


def get_obj_function(grid):
    global objective
    # O = sum of all (m.h + l)f
    m = 3
    global traffic_data
    h = nx.single_source_dijkstra_path(grid, (0, 0))
    h = {key: len(value) - 1 for key, value in h.items()}
    l: dict = nx.single_source_dijkstra_path_length(grid, (0, 0))

    print("h: " + str(h))
    # create m*h dict
    mh = h
    mh.update((x, y * m) for x, y in h.items())
    print("mh: " + str(mh))
    for source_node in grid:
        for destination_node in grid:
            h = nx.single_source_dijkstra_path(grid, source_node)
            h = {key: len(value) - 1 for key, value in h.items()}
            l: dict = nx.single_source_dijkstra_path_length(grid, source_node)
            # print("h=" + str(h))
            # print("l=" + str(l))
            # print("traffic_data=" + str(traffic_data))
            objective[str(source_node) + str(destination_node)] = (mh[destination_node] + l[destination_node]) * \
                                                                  traffic_data_dict_square[
                                                                      source_node + destination_node]

    print("objective: " + str(objective))


# read traffic data
traffic_data = np.loadtxt('traffic_uniform.csv', dtype=float, delimiter=',')
traffic_data_dict_square = {}
# get initial mesh topology
get_initial_topology()

#grid.remove_edge((0, 0), (0, 1))
edge_add_x = (0, 0)
edge_add_y = (2, 2)
weight = math.ceil(distance.euclidean(edge_add_x, edge_add_y))
print(weight)
#grid.add_edge(edge_add_x, edge_add_y, weight=weight)

# draw topology
draw_topology(grid, pos)

print(link_placement)
print(task_placement)

objective = {}
get_obj_function(grid)
