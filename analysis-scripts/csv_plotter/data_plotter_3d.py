__author__ = 'anilpa'

# ONLY FOR 2D INPUT

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import csv
with open('/Users/anilpa/Desktop/GitHub/OnlineRegression/data/input_test/ocl_leftfetchjoin_on_device_1.csv', 'rb') as csvfile:
    opdata = csv.reader(csvfile, delimiter='\t')

    input1 = []
    input2 = []
    target = []

    for row in opdata:
        input1.append(float(row[0]))
        input2.append(float(row[1]))
        target.append(float(row[5]))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(input1, input2, target, c='m', marker='^')

    ax.set_xlabel('Input1 Size')
    ax.set_ylabel('Input2 Size')
    ax.set_zlabel('Runtime')

    plt.show()