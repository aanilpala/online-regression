__author__ = 'anilpa'

from pylab import *

noise_vars = [0,1,3,5]
input_scales = [10,50,100]
sizes = [2000] # always even!
discontinuity = False
stationary = True

if discontinuity:
    func_types = [[1,2],[1,3],[1,4],[2,3],[2,4]]
else:
    func_types = [[1,1],[2,2],[3,3],[4,4]]

coeffs = [0,0,0,0,0,0,0,0]

def draw_coeffs():
    for i in range(0,len(coeffs)):
        coeffs[i] = np.random.random()*10

def func(x1, x2, func_type, noise_var):
    target = 0
    if func_type == 1:
        target = coeffs[0]*x1 + coeffs[1]*x2 # linear
    elif func_type == 2:
        sum = coeffs[2]*x1 + coeffs[3]*x2
        target = sum*log2(sum) # n*logn
    elif func_type == 3:
        target = coeffs[4]*(x1**2.0) + coeffs[5]*(x2**2.0) # quad
    elif func_type == 4:
        sum = coeffs[6]*x1 + coeffs[7]*x2
        target = sum**2.0 # quad_sum

    noise = np.random.randn()*noise_var
    target = target + noise
    return target

import csv

for cur_inp_var in input_scales:
    for cur_noise_var in noise_vars:
        for size in sizes:
            for cur_func_types in func_types:

                draw_coeffs()

                name = 'SYNTH'
                if discontinuity:
                    name += '_D_'
                else:
                    name += '_ND_'

                if stationary:
                    name += 'NCD_'
                else:
                    name += 'CD_'
                name += str(size) + '_2_' + str(cur_inp_var) + '_' + str(cur_noise_var) + '_' + str(cur_func_types[0]) + str(cur_func_types[1])
                with open('/Users/anilpa/Desktop/GitHub/OnlineRegression/data/input/' + name + '.csv', 'w') as csvfile:
                    opdata = csv.writer(csvfile, delimiter='\t')

                    # data generation
                    for i in range(0, size):
                        if not stationary and 2*i == size:
                            draw_coeffs() # concept drift!

                        target = 0
                        inp1 = np.random.random()*cur_inp_var
                        inp2 = np.random.random()*cur_inp_var

                        if inp1+inp2 < cur_inp_var:
                            target = func(inp1, inp2, cur_func_types[0], cur_noise_var)
                        else:
                            target = func(inp1, inp2, cur_func_types[1], cur_noise_var)

                        opdata.writerow([inp1, inp2, "|" + str(target)])
                print(name)