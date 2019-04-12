import csv
import numpy as np

traffic_data = np.loadtxt('traffic_uniform.csv', dtype=float, delimiter=',')

task_placement = np.arange(64)
print(len(traffic_data))