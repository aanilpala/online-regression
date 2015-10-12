__author__ = 'anilpa'

from pylab import *

noise_vars = [0,1,3,5]
input_scales = [10,50,100]
sizes = [2000] # always even!
discontinuity = True
stationary = False

func_types = []

if discontinuity:
    func_types = [[1,2],[1,3],[2,3]]
else:
    func_types = [[1,1],[2,2],[3,3]]

coeffs = [0,0,0]

def draw_coeffs():
    for i in range(0,len(coeffs)):
        coeffs[i] = np.random.random()*10
        # print(coeffs[i])

def func(x1, func_type, noise_var):
    target = 0
    if func_type == 1:
        target = coeffs[0]*x1 # linear
    elif func_type == 2:
        target = coeffs[1]*x1*log2(x1) # n*logn
    elif func_type == 3:
        target = coeffs[2]*(x1**2.0) # quad

    noise = np.random.randn()
    target = target + noise
    return target


import csv



for cur_inp_scale in input_scales:
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
                name += str(size) + '_1_' + str(cur_inp_scale) + '_' + str(cur_noise_var) + '_' + str(cur_func_types[0]) + str(cur_func_types[1])
                with open('/Users/anilpa/Desktop/GitHub/OnlineRegression/data/input/' + name + '.csv', 'w') as csvfile:
                    opdata = csv.writer(csvfile, delimiter='\t')

                    # data generation
                    for i in range(0, size):
                        if not stationary and 2*i == size:
                            draw_coeffs() # concept drift!

                        target = 0
                        inp = np.random.random()*cur_inp_scale

                        if inp < cur_inp_scale*0.5:
                            target = func(inp, cur_func_types[0], cur_noise_var)
                        else:
                            target = func(inp, cur_func_types[1], cur_noise_var)

                        opdata.writerow([inp, "|" + str(target)])
                        # inp1.append(val1)
                        # targets.append(y)
                    # plt.scatter(inp1, targets)
                    # plt.show()
                    # raw_input("Press Enter to continue...")
                print(name)


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