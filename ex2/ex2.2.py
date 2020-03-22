# -*- coding: utf-8 -*-
# @Time    : 2020/3/9 16:28
# @Author  : Thistledown
# @Email   : 120768091@qq.com
# @File    : ex2.2.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

''' Part 0: Loading Data and Defining Functions '''
# Feature mapping function to polynomial features
def mapFeature(X1, X2):
    degree = 6
    X = np.ones([len(X1), 1])
    for i in np.arange(1, degree + 1, 1):
        for j in range(i + 1):
            X = np.c_[X, X1**(i-j) * X2**(j)]
    return X

# Hypothesis function
def h(X, theta):
    return 1 / (1 + np.exp(-np.dot(X, theta)))

# Compute cost and gradient for logistic regression with regularization
def costFunctionReg(theta, X, y, reg_param):
    m = len(y)
    cost1 = - np.sum(y * np.log(h(X, theta)) + (1 - y) * np.log(1 - h(X, theta))) / m
    cost2 = 0.5 * reg_param * np.dot(theta[1:].T, theta[1:]) / m    # Don't penalize theta_0
    cost = cost1 + cost2
    grad = np.dot(X.T, h(X, theta) - y) / m
    grad[1:] += (reg_param * theta / m)[1:]
    return cost, grad

# Use Batch Gradient Descent algorithm to minimize cost
def batchGradientDescent(X, y, theta, alpha=0.1, iters = 2000, reg=1):
    J_history = np.zeros(iters)
    for i in range(iters):
        cost, grad = costFunctionReg(theta, X, y, reg)
        theta = theta - alpha * grad
        J_history[i] = cost
    return theta, J_history

# The first two columns contains the exam scores and the third column contains the label
print('Loading Data ... ')
data = np.loadtxt('ex2data2.txt', dtype=float, delimiter=',')
X, y = data[:, 0:2], data[:, 2:3]

# Plot data
fig0, ax0 = plt.subplots()
label1, label0 = np.where(y.ravel() == 1), np.where(y.ravel() == 0)
ax0.scatter(X[label1, 0], X[label1, 1], marker='+', color='k', label='y = 1')
ax0.scatter(X[label0, 0], X[label0, 1], marker='x', color='y', label='y = 0')

''' Part 1: Regularized Logistic Regression '''
m = len(y)
X = mapFeature(X[:, 0], X[:, 1])

# Initialize fitting parameters
initial_theta = np.zeros([X.shape[1], 1])

# Set regularization parameter lambda to 1
reg_param = 1

# Compute and display inital cost gradient for regularized logistic regression
cost0, grad0 = costFunctionReg(initial_theta, X, y, reg_param)
print('\nCost at initial theta (zeros): ', cost0.flatten())
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only: '
      , np.around(grad0[0:5], 4).flatten())
print('Expected gradients (approx) - first five values only: ',
      '0.0085 0.0188 0.0001 0.0503, 0.0115')

# Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = np.ones([X.shape[1], 1])
cost1, grad1 = costFunctionReg(test_theta, X, y, reg_param=10)

print('\nCost at test theta (with lambda = 10): ', cost1.flatten())
print('Expected cost (approx): 3.16')
print('Gradient at test theta - first five values only: '
      , np.around(grad1[0:5], 4).flatten())
print('Expected gradients (approx) - first five values only: ',
      '0.3460 0.1614 0.1948 0.2269, 0.0922')

''' Part 2: Regularization and Accuracies '''
# Optimize
theta, J_history = batchGradientDescent(X, y, initial_theta, reg=reg_param)
# fig1, ax1 = plt.subplots()
# ax1.plot(np.arange(2000), J_history, 'c')

# Plot boundary
poly = PolynomialFeatures(6)
x1min, x1max, x2min, x2max = X[:, 1].min(), X[:, 1].max(), X[:, 2].min(), X[:, 2].max()
xx1, xx2 = np.meshgrid(np.linspace(x1min, x1max), np.linspace(x2min, x2max))
bd = 1 / (1 + np.exp(-poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta)))
bd = bd.reshape(xx2.shape)
CS = ax0.contour(xx1, xx2, bd, [0.5], colors='c')
CS.collections[0].set_label('Decision\nBoundary')
ax0.set_title(r'$\lambda$ = '+str(reg_param))
ax0.legend(loc='upper right')
ax0.set_xlabel('Microchip Test 1')
ax0.set_ylabel('Microchip Test 2')
plt.show()

# Compute accuracy on our training set
p = np.where(h(X, theta) >= 0.5, 1.0, 0.0)
print('\nTrain Accuracy: ', np.mean(p == y) * 100, '%')
print('Expected accuracy (with lambda = 1): 83.1 % (approx)')

print('Done')