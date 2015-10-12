__author__ = 'anilpa'

# ONLY FOR 2D INPUT

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib

concept_drift = -1

import csv
with open('/Users/anilpa/Desktop/github/OnlineRegression/data/plotdata_test/KernelRegressionWS64_on_ocl_leftfetchjoin_on_device_1_prediction_plot_data.txt', 'rb') as csvfile:
    opdata = csv.reader(csvfile, delimiter='\t')

    input1 = []
    input2 = []
    predicted = []
    lower = []
    upper = []
    target = []

    ctr = 0;

    for row in opdata:

        if ctr == concept_drift:
            break

        input1.append(float(row[0]))
        input2.append(float(row[1]))
        lower.append(float(row[2]))
        predicted.append(float(row[3]))
        upper.append(float(row[4]))
        target.append(float(row[5]))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')

    ax.scatter(input1, input2, predicted, c='m', marker='*')

    ax.set_xlabel('Input1')
    ax.set_ylabel('Input2')
    ax.set_zlabel('Predicted')

    # fig1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='m', marker='*')
    # ax.legend([fig1_proxy], ['prediction'], loc='upper left', numpoints = 1)

    ax2.scatter(input1, input2, target, c='m', marker='*', label='test2')

    # fig2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='m', marker='*')
    # ax2.legend([fig2_proxy], ['target'], loc='upper left', numpoints = 1)

    z_min, z_max = ax2.get_zlim();
    ax.set_zlim(z_min, z_max);

    ax2.set_xlabel('Input1')
    ax2.set_ylabel('Input2')
    ax2.set_zlabel('Target')


    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')

    ctr = 0
    burn_in_time = 96
    stream_size = len(input1)
    random_points = 25

    points = np.random.randint(burn_in_time+100, stream_size, random_points)

    for point in points:

        x_single = [input1[point]]
        x_double = [input1[point], input1[point]]

        y_single = [input2[point]]
        y_double = [input2[point], input2[point]]

        ax3.plot(x_single,y_single,[predicted[point]],'or',)
        ax3.plot(x_double,y_double,[lower[point], upper[point]],'-b')
        ax3.plot(x_single,y_single,[target[point]],'*m')


    g1 = matplotlib.lines.Line2D([0],[0], linestyle="none", c='r', marker='o')
    g2 = matplotlib.lines.Line2D([0],[0], linestyle="none", c='b', marker='_')
    g3 = matplotlib.lines.Line2D([0],[0], linestyle="none", c='m', marker='*')

    ax3.legend([g1,g2,g3], ['prediction', 'prediction interval', 'target'], loc='upper left', numpoints = 1)

    ax3.set_xlabel('Input1 Size')
    ax3.set_ylabel('Input2 Size')
    ax3.set_zlabel('Runtime')

    plt.show()