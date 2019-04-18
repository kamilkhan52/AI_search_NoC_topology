import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math as math
import random
import pandas as pd
from matplotlib.pyplot import plot, ion, show
from sklearn.model_selection import train_test_split
from sklearn import svm


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


def reset_temperature():
    global temperature
    temperature = 5


def draw_topology(grid):
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
    temp_app1_to_app2 = traffic_data_suggested[app1][app1].copy()
    temp_app2_to_app1 = traffic_data_suggested[app2][app2].copy()
    traffic_data_suggested[app2][app1] = temp_app2_to_app1.copy()
    traffic_data_suggested[app1][app2] = temp_app1_to_app2.copy()
    temp_app1_to_app2 = 0
    temp_app2_to_app1 = 0


def perturb(total_perturbations):
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
    for t in range(0, total_perturbations):
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
    global sub_dataset
    global traffic_data

    for i in range(1, n):
        get_obj_function()
        # print("returned objective: " + str(round(objective)))

        if firstTime:
            previous_objective = objective

        if objective <= previous_objective:
            # keep change, grid=new_grid
            print("Keeping objective of " + str(objective) + " over previous objective of: " + str(previous_objective))
            previous_objective = objective
            grid = new_grid.copy()
            app_mapping = app_mapping_suggested.copy()
            traffic_data = traffic_data_suggested.copy()
            adj_mat = np.ndarray.flatten(np.triu(nx.to_scipy_sparse_matrix(grid).todense()))
            data_sample = np.append(adj_mat, app_mapping)
            data_sample = np.append(data_sample, int(objective))
            if firstTime:
                sub_dataset = data_sample.copy()
                firstTime = False
            else:
                sub_dataset = np.vstack([sub_dataset, data_sample])
                # print(sub_dataset)

        else:
            new_grid = grid.copy()
            print(
                "Rejecting objective of " + str(objective) + " over previous objective of: " + str(
                    previous_objective))

        # draw_topology()
        perturb(1)

    if temperature > temperature_threshold:
        temperature = temperature * alpha
        print("temperature = " + str(round(temperature, 2)))
        iterate(n)


# start here
firstTime = True
firstSTAGE = True
reset_temperature()
num_iterations = 5
num_iterations_stage = 5
temperature_threshold = 0.5
alpha = 0.90  # temperature decay
# read traffic data
traffic_data = np.loadtxt('traffic_uniform.csv', dtype=float, delimiter=',')
traffic_data_dict_square = {}
app_mapping = np.arange(0, 64)
app_mapping_suggested = app_mapping
previous_objective = 0
previous_prediction = 0

# np.set_printoptions(threshold=np.inf)
sub_dataset = []
dataset = []
traffic_data_suggested = traffic_data

# get initial mesh topology
if firstTime:
    get_initial_topology()
    new_grid = grid.copy()

iterate(num_iterations)

# final solution
plt.clf()
pos = dict(zip(grid, grid))
nx.draw(grid, pos, with_labels=True)
# plt.show()
print("Final traffic weighted hop count = " + str(previous_objective))
print("Final app mapping = " + str(app_mapping))

# STAGE
# reset temperature and append dataset with sub_dataset
if firstSTAGE:
    sub_dataset[:, 4160] = objective
    dataset = sub_dataset.copy()
    firstSTAGE = False

# reiterate for stage
for i in range(1, num_iterations_stage):
    sub_dataset[:, 4160] = objective
    reset_temperature()
    dataset = np.vstack([dataset, sub_dataset])  # add new sub_dataset to global dataset
    get_initial_topology()
    new_grid = grid.copy()
    objective = 0
    firstTime = True
    iterate(num_iterations)

np.savetxt("export.csv", dataset, delimiter=",")
read_data = pd.read_csv('export.csv', sep=',')

features = read_data.iloc[:, 0: 4160]
label = read_data.iloc[:, 4160]

x_train, x_test, y_train, y_test = train_test_split(features, label)

clf = svm.SVR(gamma='auto')
clf.fit(features, label)

print(clf.score(x_test, y_test))
print(clf.predict(x_test.iloc[10:11, :]))
print(y_test.iloc[11])

get_initial_topology()
# perturb and predict
for s in range(1, num_iterations * 10):

    perturb(200)
    # convert grid to list
    adj_mat = np.ndarray.flatten(np.triu(nx.to_scipy_sparse_matrix(new_grid).todense()))
    draw_topology(new_grid)
    data_sample = np.append(adj_mat, app_mapping_suggested)
    # data_sample = np.append(data_sample, int(objective))

    # predict objective function
    # features_grid = data_sample[:4160]
    data_sample = data_sample.reshape(1, -1)
    prediction = clf.predict(data_sample)
    print("Prediction for current grid: " + str(prediction))
    if previous_prediction == 0:
        previous_prediction = prediction
        grid = new_grid.copy()

    if prediction < previous_prediction:
        # accept perturbation
        grid = new_grid.copy()
        app_mapping = app_mapping_suggested.copy()
        traffic_data = traffic_data_suggested.copy()
        previous_prediction = prediction
    else:
        print("Prediction rejected: " + str(prediction) + " because Previous prediction = " + str(previous_prediction))

    # restart Greedy search with new starting point

print("Final Prediction: " + str(prediction))
