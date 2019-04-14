import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math as math
import random


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
    # print("traffic_data_dict=" + str(len(traffic_data_dict)) + str(traffic_data_dict))
    for source in traffic_data_dict:
        for dest in traffic_data_dict:
            traffic_data_dict_square[source + dest] = traffic_data[8 * source[1] + source[0]][8 * dest[1] + dest[0]]
    # print("traffic_data_dict_square=" + str(len(traffic_data_dict_square)) + str(traffic_data_dict_square))


def draw_topology(grid, pos):
    nx.draw(grid, pos, with_labels=False)
    print(nx.nodes(grid))
    plt.show(block=False)
    plt.show()


def get_obj_function(grid):
    global objective_dict
    objective_dict = {}
    global objective
    # O = sum of all (m.h + l)f
    m = 3
    global traffic_data
    h = nx.single_source_dijkstra_path(grid, (0, 0))
    h = {key: len(value) - 1 for key, value in h.items()}
    l: dict = nx.single_source_dijkstra_path_length(grid, (0, 0))

    for source_node in grid:
        # print(source_node)
        h = nx.single_source_dijkstra_path(grid, source_node)
        # print("shortest paths: " + str(h))
        h = {key: len(value) - 1 for key, value in h.items()}
        # print("total hops: " + str(h))
        # create m*h dict
        mh = h
        mh.update((x, y * m) for x, y in h.items())
        # print("mh: " + str(mh))

        l: dict = nx.single_source_dijkstra_path_length(grid, source_node)
        # print("l : " + str(l))

        for destination_node in grid:
            objective_dict[source_node + destination_node] = (mh[destination_node] + l[destination_node]) * \
                                                             traffic_data_dict_square[
                                                                 source_node + destination_node]
            # print(
            #     "objective_dict calculated from values: source_node" + str(source_node) + "| destination_node: " + str(
            #         destination_node) + "| mh :" + str(mh[destination_node]) + " | l[destination_node]): " + str(
            #         l[destination_node]) + "|  traffic_data_dict_square[source_node + destination_node]: " + str(
            #         traffic_data_dict_square[source_node + destination_node]))

    # print("objective_dict: " + str(objective_dict))
    objective = sum(objective_dict.values())

    print("objective: " + str(round(objective)))


def add_edge(grid, add_x, add_y):
    # grid.remove_edge((0, 0), (0, 1))
    # add_x = (3, 7)
    # add_y = (0, 1)
    weight = math.ceil(distance.euclidean(add_x, add_y))
    grid.add_edge(add_x, add_y, weight=weight)


def remove_edge(grid, rem_x, rem_y):
    grid.remove_edge(rem_x, rem_y)


def swap_edges(grid, edge1, edge2):
    pass


def swap_app(grid, app1, app2):
    pass


def perturb(grid):
    # pick edge to swap

    # swap_edges()
    edges_one_d = []
    not_edges_one_d = []
    edge2_core1 = []
    edge2_core2 = []


    for edge in grid.edges:
        edges_one_d.append(str(edge))
    for not_edge in nx.non_edges(grid):
        not_edges_one_d.append(str(not_edge))

    print("list of edges: " + str(edges_one_d))
    print("list of not edges: " + str(not_edges_one_d))

    edge1 = np.random.choice(edges_one_d)

    # need to pick only edges within a certain edge length
    edge2 = np.random.choice(not_edges_one_d)
    for i in edge2[2:6].split(","):
        edge2_core1.append(int(i))
    for i in edge2[10:14].split(","):
        edge2_core2.append(int(i))

    print("core1: " + str(edge2_core1))
    print("core2: " + str(edge2_core2))
    weight = math.ceil(distance.euclidean(edge2_core1, edge2_core2))  # length of the proposed link

    while weight > 4:
        print("weight is: " + str(weight))
        edge2_core1.clear()
        edge2_core2.clear()
        edge2 = np.random.choice(not_edges_one_d)
        for i in edge2[2:6].split(","):
            edge2_core1.append(int(i))
        for i in edge2[10:14].split(","):
            edge2_core2.append(int(i))

        print("Last link was too long. New core1: " + str(edge2_core1))
        print("New core2: " + str(edge2_core2))
        weight = math.ceil(distance.euclidean(edge2_core1, edge2_core2))  # length of the proposed link

    print("edges to swap (not exactly swap) are: " + str(edge1) + " and " + str(edge2))

    # pick app to swap

    app1 = int(random.random() * 64)
    app2 = int(random.random() * 64)
    print("apps to swap are: " + str(app1) + " and " + str(app2))
    # swap_app()
    # draw_topology()


def iterate(n):
    global temperature
    for i in range(1, n):
        get_obj_function(grid)
        print("returned objective: " + str(round(objective)))

        if firstTime:
            previous_objective = objective

        if objective < previous_objective:
            # keep change
            previous_objective = objective
        else:
            probability = math.exp(- abs((objective - previous_objective)) / temperature)
            print("probability: " + str(probability))
            # accept change with probability
            if random.random() < probability:
                # keep change
                previous_objective = objective

    # perturb
    if temperature > temperature_threshold:
        temperature = temperature * alpha
        print("temperature = " + str(round(temperature, 2)))
        perturb(grid)
        iterate(n)


# start here
firstTime = True
temperature = 500
num_iterations = 3
temperature_threshold = 0.1
alpha = 0.90  # temperature decay
# read traffic data
traffic_data = np.loadtxt('traffic_uniform.csv', dtype=float, delimiter=',')
traffic_data_dict_square = {}
# get initial mesh topology
if firstTime:
    get_initial_topology()

# draw topology
# draw_topology(grid, pos)

iterate(num_iterations)
