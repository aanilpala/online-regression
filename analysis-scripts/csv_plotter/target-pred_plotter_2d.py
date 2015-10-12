__author__ = 'anilpa'

inp = []
pred = []
target = []
lower = []
upper = []

con_drift = -1

import csv
with open('/Users/anilpa/Desktop/GitHub/OnlineRegression/data/plotdata_test/KernelRegressionWS64_on_ocl_groupby_on_device_2_prediction_plot_data.txt', 'rb') as csvfile:
    opdata = csv.reader(csvfile, delimiter='\t')

    ctr = 0

    for row in opdata:

        if(ctr == con_drift):
            break

        inp.append(float(row[0]))
        lower.append(float(row[1]))
        pred.append(float(row[2]))
        upper.append(float(row[3]))
        target.append(float(row[-2]))

        ctr = ctr + 1


import numpy as np

burn_in_time = 50
stream_size = len(inp)
random_points = 50
points = np.random.randint(burn_in_time, stream_size, random_points)

import matplotlib.pyplot as plt
import matplotlib

full_fig = plt.figure()
sampled_fig = plt.figure()

ax1 = full_fig.add_subplot(111)
ax2 = sampled_fig.add_subplot(111)

# ax1.scatter(inp, target, s=10, c='r', label='target')
ax1.scatter(inp, pred, s=10, c='b', label='')

ax1.set_xlabel('Input1')
ax1.set_ylabel('Predicted')
ax1.grid(True)
# ax1.set_ylim(-10000000,50000000)

ax1.legend(loc='upper left', numpoints = 1);

ax2.set_xlabel('Input1')
ax2.set_ylabel('Target')
ax2.grid(True)

for point in points:
    domain = [inp[point]]
    doubled_dom = ([inp[point], inp[point]])

    p = pred[point]
    lb = lower[point]
    ub = upper[point]
    t = target[point]

    ax2.plot(domain,pred[point],'or')
    ax2.plot(doubled_dom,[lower[point], upper[point]],'-b')
    ax2.plot(domain,[target[point]],'*m')

    print lb, p, ub, t


g1 = matplotlib.lines.Line2D([0],[0], linestyle="none", c='r', marker='o')
g2 = matplotlib.lines.Line2D([0],[0], linestyle="none", c='b', marker='_')
g3 = matplotlib.lines.Line2D([0],[0], linestyle="none", c='m', marker='*')

ax2.legend([g1,g2,g3], ['prediction', 'prediction interval', 'target'], loc='upper left', numpoints = 1)

plt.show()