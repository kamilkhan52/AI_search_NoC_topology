import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math as math
import random
from matplotlib.pyplot import plot, ion, show


def get_initial_topology():
    global grid
    global pos
    global traffic_data_dict_square
    global app_mapping
    grid = nx.grid_2d_graph(8, 8)
    pos = dict(zip(grid, grid))
    nx.set_edge_attributes(grid, 1, 'length')
    # print(app_mapping)
    # format traffic data into a dictionary
    traffic_data_dict = {node: 0 for node in grid}
    for source in traffic_data_dict:
        for dest in traffic_data_dict:
            traffic_data_dict_square[source + dest] = traffic_data[8 * source[1] + source[0]][8 * dest[1] + dest[0]]
    # print("traffic_data_dict_square=" + str(len(traffic_data_dict_square)) + str(traffic_data_dict_square))


def draw_topology():
    global pos
    plt.figure(1)
    plt.clf()
    pos = dict(zip(grid, grid))
    nx.draw(grid, pos)
    nx.draw_networkx_labels(grid, pos, font_size=6)
    plt.pause(.00005)
    plt.show(block=False)


def get_obj_function():
    global objective_dict
    objective_dict = {}
    global objective
    global traffic_data_suggested
    # Objective = sum of all (m.h + l)f
    m = 3

    # update traffic data for new mapping
    traffic_data_dict = {node: 0 for node in new_grid}
    for source in traffic_data_dict:
        for dest in traffic_data_dict:
            traffic_data_dict_square[source + dest] = traffic_data_suggested[8 * source[1] + source[0]][
                8 * dest[1] + dest[0]]

    for source_node in new_grid:
        h = nx.single_source_dijkstra_path(new_grid, source_node)
        # print("shortest paths: " + str(h))
        h = {key: len(value) - 1 for key, value in h.items()}
        # print("total hops: " + str(h))
        # create m*h dict
        mh = h
        mh.update((x, y * m) for x, y in h.items())

        l: dict = nx.single_source_dijkstra_path_length(new_grid, source_node)

        for destination_node in new_grid:
            try:
                objective_dict[source_node + destination_node] = (mh[destination_node] + l[destination_node]) * \
                                                                 traffic_data_dict_square[
                                                                     source_node + destination_node]
            # print(
            #     "objective_dict calculated from values: source_node" + str(source_node) + "| destination_node: " + str(
            #         destination_node) + "| mh :" + str(mh[destination_node]) + " | l[destination_node]): " + str(
            #         l[destination_node]) + "|  traffic_data_dict_square[source_node + destination_node]: " + str(
            #         traffic_data_dict_square[source_node + destination_node]))

            # print("objective_dict: " + str(objective_dict))
            except Exception:
                objective_dict[source_node + destination_node] = 9999999

    objective = sum(objective_dict.values())
    objective_dict.clear()
    l.clear()
    mh.clear()
    h.clear()

    # print("objective: " + str(round(objective)))


def add_edge(add_x, add_y):
    weight = math.ceil(distance.euclidean(add_x, add_y))
    new_grid.add_edge(add_x, add_y, weight=weight)


def remove_edge(rem_x, rem_y):
    new_grid.remove_edge(rem_x, rem_y)


def swap_app(app1, app2):
    # swap the column of app1 with app2, and the row of app1 and app2 in traffic data

    global app_mapping_suggested
    global app_mapping
    global traffic_data
    global traffic_data_suggested
    app_mapping_suggested = app_mapping

    app_mapping_suggested[app1] = app2
    app_mapping_suggested[app2] = app1
    # print("app_mapping_suggested: " + str(app_mapping_suggested))

    traffic_data_suggested = traffic_data[:, app_mapping_suggested]
    traffic_data_suggested[[app1, app2]] = traffic_data_suggested[[app2, app1]]


def perturb():
    # ready the variables
    edges_one_d = []
    not_edges_one_d = []
    edge2_core1 = []
    edge2_core2 = []
    edge1_core1 = []
    edge1_core2 = []
    global new_grid
    new_grid = grid.copy()

    # covert existing edges to 1-dimensional array
    for edge in new_grid.edges:
        edges_one_d.append(str(edge))
    for not_edge in nx.non_edges(new_grid):
        not_edges_one_d.append(str(not_edge))

    # pick an edge from existing edges to remove
    edge1 = np.random.choice(edges_one_d)
    for i in edge1[2:6].split(","):
        edge1_core1.append(int(i))
    for i in edge1[10:14].split(","):
        edge1_core2.append(int(i))

    total_edges_for_core2 = len(new_grid.edges(tuple(edge1_core2)))
    total_edges_for_core1 = len(new_grid.edges(tuple(edge1_core1)))

    # make sure the edge selected is not the only one for a core
    while total_edges_for_core1 <= 1 or total_edges_for_core2 <= 1:
        edge1_core1.clear()
        edge1_core2.clear()
        edge1 = np.random.choice(edges_one_d)
        for i in edge1[2:6].split(","):
            edge1_core1.append(int(i))
        for i in edge1[10:14].split(","):
            edge1_core2.append(int(i))

        # print("edges to remove : " + str(edge1))
        # print("case detected: total_edges_for_core2: " + str(
        #     total_edges_for_core2) + "case detected: total_edges_for_core1: " + str(total_edges_for_core1))
        total_edges_for_core2 = len(new_grid.edges(tuple(edge1_core2)))
        total_edges_for_core1 = len(new_grid.edges(tuple(edge1_core1)))

    # covert non-existing edges to 1-dimensional array
    edge2 = np.random.choice(not_edges_one_d)
    for i in edge2[2:6].split(","):
        edge2_core1.append(int(i))
    for i in edge2[10:14].split(","):
        edge2_core2.append(int(i))

    # print("core1: " + str(edge2_core1))
    # print("core2: " + str(edge2_core2))
    weight = math.ceil(distance.euclidean(edge2_core1, edge2_core2))  # length of the proposed link

    while weight > 4:
        # stay till a link length of <4 is found
        edge2_core1.clear()
        edge2_core2.clear()
        edge2 = np.random.choice(not_edges_one_d)
        for i in edge2[2:6].split(","):
            edge2_core1.append(int(i))
        for i in edge2[10:14].split(","):
            edge2_core2.append(int(i))

        weight = math.ceil(distance.euclidean(edge2_core1, edge2_core2))  # length of the proposed link

    # print("edges to remove : " + str(edge1) + " and add: " + str(edge2))
    remove_edge(tuple(edge1_core2), tuple(edge1_core1))
    add_edge(tuple(edge2_core2), tuple(edge2_core1))
    # clear link removal data
    edge1_core1.clear()
    edge1_core2.clear()
    not_edges_one_d.clear()
    edges_one_d.clear()
    edge2_core1.clear()
    edge2_core2.clear()

    # pick app to swap

    app1 = int(random.random() * 64)
    app2 = int(random.random() * 64)
    # print("apps to swap are: " + str(app1) + " and " + str(app2))

    swap_app(app1, app2)


def iterate(n):
    global temperature
    global new_grid
    global grid
    global app_mapping_suggested
    global app_mapping
    global firstTime
    global previous_objective
    global traffic_data

    for i in range(1, n):
        get_obj_function()
        # print("returned objective: " + str(round(objective)))

        if firstTime:
            previous_objective = objective
            firstTime = False

        if objective <= previous_objective:
            # keep change, grid=new_grid
            print("↓Objective: " + str(round(objective, 2)))
            previous_objective = objective
            grid = new_grid.copy()
            app_mapping = app_mapping_suggested
        else:
            probability = math.exp(- abs((objective - previous_objective)) / temperature)
            #
            # print("probability: " + str(probability))
            # accept change with probability
            if random.random() < probability:
                # keep change, grid=new_grid
                print("↓Objective: " + str(round(objective, 2)))
                previous_objective = objective
                grid = new_grid.copy()
                app_mapping = app_mapping_suggested.copy()
                traffic_data = traffic_data_suggested.copy()
                app_mapping = app_mapping_suggested
            else:
                new_grid = grid.copy()
                print("-Objective: " + str(round(objective, 2)))

        draw_topology()
        perturb()

    if temperature > temperature_threshold:
        temperature = temperature * alpha
        print("temperature = " + str(round(temperature, 2)))
        iterate(n)


# start here
firstTime = True
temperature = 100
num_iterations = 20
temperature_threshold = 0.1
alpha = 0.70  # temperature decay
# read traffic data
traffic_data = np.loadtxt('traffic_complement.csv', dtype=float, delimiter=',')
traffic_data_dict_square = {}
app_mapping = np.arange(0, 64)
app_mapping_suggested = app_mapping
previous_objective = 0

traffic_data_suggested = traffic_data

# get initial mesh topology
if firstTime:
    get_initial_topology()
    new_grid = grid.copy()

iterate(num_iterations)

# final solution
plt.clf()
pos = dict(zip(grid, grid))
nx.draw(grid, pos, font_size=6)
plt.show()
print("Final traffic weighted hop count = " + str(previous_objective))
print("Final app mapping = " + str(app_mapping))
