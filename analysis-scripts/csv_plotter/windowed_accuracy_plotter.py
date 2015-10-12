from matplotlib.mlab import prctile

__author__ = 'anilpa'

accuracy_log_path = '/Users/anilpa/Desktop/github/OnlineRegression/data/plotdata_compare/'

acc_data_array = []
regressor_name = []
# input_width = 1;

import csv
import os

# data collected

filelist = os.listdir(accuracy_log_path)[1:]

# filelist = sorted(filelist, key=lambda key : int(key.split('_')[0].split('WS')[1]), reverse=True)

for filename in filelist:

    if filename[0] == '.':
        continue

    print(accuracy_log_path + filename)
    regressor_name.append(filename.split('_')[0])

    with open(accuracy_log_path + filename) as csvfile:
        accuracy_data = csv.reader(csvfile, delimiter='\t')

        cur_accuracy_data = []

        for row in accuracy_data:
            cur_accuracy_data.append(row[-1])
            # print(row[input_width + 3])

        acc_data_array.append(cur_accuracy_data)

regressor_count = len(regressor_name)
item_count = len(acc_data_array[0])

## processing data (into sub-interval accuracy data) and (optional) smoothing
import math as math
import collections


domain = range(0, item_count, 1)
processed_acc_arr = []
max_val = 0
window_size = 96;

for each_reg in acc_data_array:
    cur_processed_acc_arr = []
    window = collections.deque(maxlen=window_size)
    window_mse = 0;
    window_ctr = 0

    for dp in each_reg:
        value = float(dp)
        if window_ctr < window_size:
            window.append(value)
            window_mse = window_mse + value*value
            window_ctr = window_ctr + 1;

            if(window_mse < 0):
                window_mse = 0

            cur_val = math.sqrt(window_mse) / window_ctr

            # if cur_val > max_val:
            #     max_val = cur_val

            cur_processed_acc_arr.append(cur_val)
        else:
            dropped = window.popleft()
            window.append(value)
            window_mse = window_mse - dropped*dropped
            window_mse = window_mse + value*value

            if(window_mse < 0):
                window_mse = 0

            cur_val = math.sqrt(window_mse) / window_ctr
            if cur_val > max_val:
                max_val = cur_val

            cur_processed_acc_arr.append(cur_val)

    processed_acc_arr.append(cur_processed_acc_arr)


from scipy.interpolate import spline

# plotting
import matplotlib.pyplot as plt
import numpy as np

colormap = plt.cm.gist_ncar
colors=[colormap(i) for i in np.linspace(0, 0.9, regressor_count)]

ax = []
ln = []
lns = []

# fig = plt.subplots()[0]
ax.append(plt.subplots()[1])

ln.append(ax[0].plot(domain, processed_acc_arr[0], 'k', label=regressor_name[0]))
ax[0].set_xlabel('stream items')
# Make the y-axis label and tick labels match the line color.
ax[0].set_ylabel('sliding window rmse', color='k')
# ax[0].set_ylim(-10,max_val)
ax[0].set_ylim(0,1.5*max_val)
for tl in ax[0].get_yticklabels():
    tl.set_color('k')
lns += ln[0]

for ctr in range (1,len(acc_data_array), 1):
    ax.append(ax[0].twinx())
    ln.append(ax[ctr].plot(domain, processed_acc_arr[ctr], c=colors[ctr], label=regressor_name[ctr]))
    ax[ctr].get_yaxis().set_ticks([])
    # ax[ctr].set_ylim(-10,max_val)
    ax[ctr].set_ylim(0,1.5*max_val)
    lns += ln[ctr]

# ax.append(ax[0].twinx())
#
# ln.append(ax[1].plot(domain, acc_data_array[1], 'r', label=regressor_name[1]))
# ax[1].get_yaxis().set_ticks([])
# # ax[1].set_ylabel('learning rate', color='r')
# # for tl in ax[1].get_yticklabels():
# #     tl.set_color('r')
#
# ax.append(ax[0].twinx())
#
# ln.append(ax[2].plot(domain, acc_data_array[2], 'g', label=regressor_name[2]))
# ax[2].get_yaxis().set_ticks([])
#
#
# lns = ln[0]+ln[1]+ln[2]

labs = [l.get_label() for l in lns]
ax[0].legend(lns, labs, loc=0)

plt.show(1)

