import numpy as np

from gpcam.gp_optimizer import GPOptimizer

import matplotlib.pyplot as plt

from numpy.random import default_rng


a = np.load("us_topo.npy")
print(np.shape(a))

rng = default_rng()   # Construct a new Generator with the default BitGenerator (PCG64)
"""
PCG-64 has a period of 2^128 and supports advancing an arbitrary number of steps as well as 2^127 streams.
"""
ind = rng.choice(len(a)-1, size=3000, replace=False)   # Replace means the value won't be selected multiple times.
print(np.shape(ind))
ind = np.array([15760])    # 15760 out of 25000
points = a[ind, 0:2]    # First and second column
print(np.shape(points))
print('This is point:', points)
values = a[ind, 2:3]    # Third column
print(np.shape(values))
print('This is value:', values)
# print("x_min ", np.min(points[:,0])," x_max ",np.max(points[:,0]))
#
# print("y_min ", np.min(points[:,1])," y_max ",np.max(points[:,1]))
#
# print("length of data set: ", len(points))



index_set_bounds = np.array([[0,99],[0,248]])

hyperparameter_bounds = np.array([[0.001,1e9],[1,1000],[1,1000]])

hps_guess = np.array([4.71907062e+06, 4.07439017e+02, 3.59068120e+02])



###################################################################################
"""
help(GPOptimizer)
GPOptimizer(input_space_dimension, output_space_dimension, output_number, index_set_bounds)
"""
gp = GPOptimizer(2, 1, 1, index_set_bounds)

gp.tell(points, values)  # (Points in the data, Values measured at the associated points)

gp.init_gp(hps_guess)   # Initialize the GP, 1d numpy array containing the initial guesses for the hyperparemeters

# gp.train_gp(hyperparameter_bounds,likelihood_optimization_pop_size = 20,
#
#                   likelihood_optimization_tolerance = 1e-6, likelihood_optimization_max_iter = 2)



# x_pred = np.empty((10000,2))
#
# counter = 0
#
# x = np.linspace(0,99,100)
#
# y = np.linspace(0,248,100)
#
#
#
# for i in x:
#
#  for j in y:
#
#    x_pred[counter] = np.array([i,j])
#
#    counter += 1
#
#
#
# res1 = gp.gp.posterior_mean(x_pred)
#
# res2 = gp.gp.posterior_covariance(x_pred)
#
# #res3 = gp.gp.shannon_information_gain(x_pred)
#
# X,Y = np.meshgrid(x,y)
#
#
#
# PM = np.reshape(res1["f(x)"],(100,100))
#
# PV = np.reshape(res2["v(x)"],(100,100))
#
# plt.figure(figsize= (10,10))
#
# plt.pcolormesh(X,Y,PM)
#
# plt.figure(figsize= (10,10))
#
# plt.pcolormesh(X,Y,PV)
#
# plt.show()
# point = np.array([])
# value = np.array([])


def upper_confidence_bounds(x,obj):

    a = 3.0  # 3.0 for 95 percent confidence interval

    mean = obj.posterior_mean(x)["f(x)"]    # Return an array
    cov = obj.posterior_covariance(x)["v(x)"]

    print((mean, cov))
    scalar = mean[-1] + a * cov[-1]
    # print(scalar)   # We need a scalar
    return scalar.item()
    # return np.asscalar(mean + a * cov)

"""
Position (numpy array):            last measured point, default = None
n (int):                           how many new measurements are requested, default = 1
objective_function:                default = None, means that the class objective function will be used
 """

for i in range(3):
    next = gp.ask(position=None, n=1, objective_function="covariance", optimization_bounds=None,

                 optimization_method="global", optimization_pop_size=50, optimization_max_iter=20,

                 optimization_tol=10e-6, dask_client=False)   # objective_function = "covariance"
    print('--------------------------------------------------------------------------The next is')
    print(next)
    next_f = next['f(x)'][0]
    print('--------------------------------------------------------------------------Points and values')
    x = next['x'][0][0]
    y = next['x'][0][1]
    # TODO: Assign the next roi(ground truth)
    for index in a:
        print('index:', index)
        if np.array([x, y]) in index[0:2]:
            print('Yes')
            next_roi = index[2:3]
        else:
            print('Error')
    points = np.append(points, next['x'])
    values = np.append(values, next_roi)
    print('points:', points)
    print('values:', values)
    print(next_f)
    gp.tell(points, values)
print('--------------------------------------')
points = points.reshape(-1, 2)
print(points)
values = values.reshape(-1, 1)
print(values)
ans = upper_confidence_bounds(values, gp.gp)
print(ans)