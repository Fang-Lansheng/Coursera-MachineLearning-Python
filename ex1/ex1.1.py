# -*- coding: utf-8 -*-
# @Time    : 2020/3/2 16:26
# @Author  : Thistledown
# @Email   : 120768091@qq.com
# @File    : ex1.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

""" Part 1: Basic Function """
def computeCost(X, y, theta):
    '''
    Compute cost for linear regression
    :param X: input variables
    :param y: output variables
    :param theta: parameters
    :return: Cost function
    '''
    m = len(y)
    J = 0
    for xi, yi in zip(X, y):
        J = J + np.square(float(np.dot(xi, theta)) - yi)
    return J / (2 * m)

def gradientDescent(X, y, theta, alpha, num_iters):
    '''
    updates theta by taking num_iters gradient steps with learning rate alpha
    :param X: input variables
    :param y: output variables
    :param theta: parameters
    :param alpha: learning rate
    :param num_iters: times of iteration
    :return: [theta, J_history]
    '''
    # Initialize some useful values
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        sum0, sum1 = 0, 0
        for j in range(m):
            sum0 = sum0 + (np.dot(X[j], theta) - y[j]) * X[j][0]
            sum1 = sum1 + (np.dot(X[j], theta) - y[j]) * X[j][1]
        theta[0] = theta[0] - alpha * sum0 / m
        theta[1] = theta[1] - alpha * sum1 / m

        # Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history


""" Part 2: Plotting """
print('Plotting Data ...')
data = np.loadtxt("ex1data1.txt", dtype=float, delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)      # number of training examples

# Plot Data
fig0, ax0 = plt.subplots()
ax0.scatter(X, y, marker='x', label='Training data')
ax0.set_xlabel('Population of City in 10,000s')
ax0.set_ylabel('Profit in $10,000s')

""" Part 3: Cost and Gradient Descent """
X = np.c_[np.ones((m, 1)), data[:,0]]   # Add a column of ones to X
theta = np.zeros((2, 1))                # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ... \n')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0 ; 0], cost computed = ', J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, np.array([[-1], [2]]))
print('With theta = [-1 ; 2], cost computed = ', J)
print('Expected cost value (approx) 54.24\n')

print('Running Gradient Descent ...\n')
# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: θ0 =', theta[0], ', θ1 =', theta[1])
print('Expected theta values (approx): -3.6303  1.6664\n')

# Plot the linear fit
ax0.plot(X[:, 1], np.dot(X, theta), color='r', label='Linear regression')
ax0.legend(loc='lower right')
ax0.set_title('Linear regression with one variable')

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
predict2 = np.dot([1, 7], theta)
print('For population = 35,000, we predict a profit of ', predict1 * 10000)
print('For population = 70,000, we predict a profit of ', predict2 * 10000)

""" Part 4: Visualizing J(theta_0, theta_1) """
print('\nVisualizing J(theta_0, theta_1) ... \n')

# Plot the reducing of cost function during iteration
fig1, ax1 = plt.subplots()
ax1.plot(np.arange(iterations), J_history, 'b')
ax1.set_xlabel('Iterations')
ax1.set_ylabel(r'$J(\theta_0, \theta_1)$')
ax1.set_title(r'Reducing of $J(\theta_0, \theta_1)$ during iteration')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# Initialize J_vals to a matrix of 0's
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = [theta0_vals[i], theta1_vals[j]]
        J_vals[i][j] = computeCost(X, y, t)

x_contour, y_contour = theta0_vals, theta1_vals
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

# Produce surface and contour plots of J(θ)
fig2 = plt.figure()
ax2 = Axes3D(fig2)
ax2.plot_surface(theta0_vals, theta1_vals, J_vals.T, rstride=1, cstride=1, cmap=cm.rainbow)
ax2.set_xlabel(r'$\theta_0$')
ax2.set_ylabel(r'$\theta_1$')
ax2.set_title('Surface')

fig3, ax3 = plt.subplots()
CS = ax3.contour(theta0_vals, theta1_vals, J_vals.T, levels=50)
ax3.plot(theta[0], theta[1], 'bx')
ax3.set_xlabel(r'$\theta_0$')
ax3.set_ylabel(r'$\theta_1$')
ax3.set_title('Contour')

plt.show()

