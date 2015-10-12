__author__ = 'anilpa'

# ONLY FOR 1D INPUT

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import csv
with open('/Users/anilpa/Desktop/GitHub/OnlineRegression/data/input_test/ocl_groupby_on_device_2.csv', 'rb') as csvfile:
    opdata = csv.reader(csvfile, delimiter='\t')

    input1 = []
    target = []

    for row in opdata:
        input1.append(float(row[0]))
        target.append(float(row[-1]))


    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel('Input1')
    ax1.set_ylabel('Target')

    # ax1.scatter(input1, target, s=10, c='r', label='target')

    ax1.scatter(input1, target, s=10,  c='r', label='target')
    ax1.grid(True)

    plt.show()