__author__ = 'anilpa'

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

noise_var = 2.5
input_var = 10
discontinuity = False

def func(x1):
    target = 2.4*x1 # linear
    # target = 2.4*(x1**2.0) # quad
    # target = 2.4*log2(x1) # log
    return target

def disc_func(x1):
    target = 0
    if x1 < input_var/2.0:
        target = 2.4*x1
    else:
        target = 2.1*(x1**2.0)

    noise = 2*(0.5 - np.random.random())*noise_var
    target = target + noise
    return target

inp1 = []
targets = []
size = 2500;

import csv
with open('/Users/anilpa/Desktop/GitHub/OnlineRegression/data/input/SYNTH_1_10_1_1_2_1' + size.__str__() + '.csv', 'w') as csvfile:
    opdata = csv.writer(csvfile, delimiter='\t')

    for i in range(0, size):
        val1 = np.random.random()*input_var
        if discontinuity:
            y = disc_func(val1)
        else:
            y = func(val1)
        opdata.writerow([val1, "|" + y.__str__()])
        inp1.append(val1)
        targets.append(y)

# plot the graph of the generated data set
import itertools

for (x, y) in itertools.izip(inp1, targets):
    print str(x) + ' ' + str(y)

plt.scatter(inp1, targets)


# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(inp1, inp2, targets, c='m', marker='^')
#
# ax.set_xlabel('Input1 Size')
# ax.set_ylabel('Input2 Size')
# ax.set_zlabel('Runtime')
#
# ax.set_zlim(0,3000);
#
plt.show()