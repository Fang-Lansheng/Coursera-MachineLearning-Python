# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 21:27
# @Author  : Thistledown
# @Email   : 120768091@qq.com
# @File    : ex2.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

""" Part 0: Basic Function """
def computeCost(X, y, theta):
    '''
    Compute cost for linear regression
    :param X: input variables
    :param y: output variables
    :param theta: parameters
    :return: Cost function
    '''
    m = len(y)
    J = np.sum(np.square(X.dot(theta) - y)) / (2 * m)
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
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
    m, n = len(y), len(theta)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        temp = np.dot((np.dot(X, theta) - y.reshape(m, 1)).T, X)
        theta = theta - alpha * temp.T / m
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history

def featureNormalize(X):
    '''
    Normalize the features in X
    :param X: features
    :return: X_norm: normalized X; mean: mean value; sigma: standard deviation
    '''
    mean = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X - mean) / sigma

    return X_norm, mean, sigma

def normalEqn(X, y):
    '''
    Conputes the closed-form solution to linear regression
    :param X: input variables
    :param y: output variables
    :return: parameters
    '''
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta

""" Part 1: Feature Normalization """
# Load data
data = np.loadtxt('ex1data2.txt', dtype=float, delimiter=',')
X = data[:, 0:2]
y = data[:, 2:3]
m = len(y)

# Scale features and set them to zero mean
X_norm, mean, sigma = featureNormalize(X)
y_norm, _, _ = featureNormalize(y)

# Add intercept term to X
X_norm = np.c_[np.ones([m, 1]), X_norm]

""" Part 2: Gradient Descent """
print('Running gradient descent ... ')

# Choose some alpha value
alpha = 0.01
num_iters = 1000

# Init theta and run gradient descent
theta = np.zeros([3, 1])
theta, J_history = gradientDescentMulti(X_norm, y, theta, alpha, num_iters)

# Plot the convergence graph
fig1, ax1 = plt.subplots()
ax1.plot(np.arange(num_iters), J_history, 'b')
ax1.set_xlabel('Iterations')
ax1.set_ylabel(r'$J(\theta)$')
ax1.set_title(r'Convergence of $J(\theta)$')

# Display gradient descent's result
print('Theta computed from gradient descent: \n', theta)

""" Part 3: Normal Equations """
print('\nSolving with normal equations ... ')

# Add intercept term to X
X = np.c_[np.ones([m, 1]), X]

# Calculate the parameters from the normal equation
theta1 = normalEqn(X, y)
print('Theta computed from normal equations: \n', theta1)

plt.show()