# -*- coding: utf-8 -*-
# @Time    : 2020/3/7 20:36
# @Author  : Thistledown
# @Email   : 120768091@qq.com
# @File    : ex2.py.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

''' Part 0: Loading Data and Defining Functions '''
# Hypothesis function
def h(X, theta):
    return 1 / (1 + np.exp(-np.dot(X, theta)))

# Logistic regression cost function
def costFunction(theta, X, y):
    m = len(y)
    cost = - np.sum(y * np.log(h(X, theta)) + (1 - y) * np.log(1 - h(X, theta))) / m
    grad = np.dot(X.T, h(X, theta) - y) / m
    return cost, grad

# The objective function to be minimized
def costFunc(params, *args):
    X, y = args
    [m, n] = X.shape
    theta = params.reshape([n, 1])
    cost = - np.sum(y * np.log(h(X, theta)) + (1 - y) * np.log(1 - h(X, theta))) / m
    return cost

# Method for computing the gradient vector
def gradFunc(params, *args):
    X, y = args
    [m, n] = X.shape
    theta = params.reshape([n, 1])
    grad = np.dot(X.T, h(X, theta) - y) / m
    return grad.flatten()

# The first two columns contains the exam scores and the third column contains the label
print('Loading Data ... ')
data = np.loadtxt('ex2data1.txt', dtype=float, delimiter=',')
X, y = data[:, 0:2], data[:, 2:3]

''' Part 1: Plotting '''
fig0, ax0 = plt.subplots()
label1, label0 = np.where(y.ravel() == 1), np.where(y.ravel() == 0)
ax0.scatter(X[label1, 0], X[label1, 1], marker='+', color='g', label='Admitted')
ax0.scatter(X[label0, 0], X[label0, 1], marker='x', color='r', label='Not admitted')
ax0.legend(loc='upper right')
ax0.set_xlabel('Exam 1 Score')
ax0.set_ylabel('Exam 2 Score')

''' Part 2: Compute Cost and Gradient '''
[m, n] = X.shape
X = np.c_[np.ones([m, 1]), X]           # Add intercept term to x and X_test
initial_theta = np.zeros([n + 1, 1])    # Initialize fitting parameters

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

print('\nCost at initial theta : ', cost.ravel())
print('Expected cost (approx): 0.693')
print('Gradient at initial theta :', grad.ravel())
print('Expected gradients (approx): \t-0.1000\t -12.0092\t -11.2628')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array(([-24], [0.2], [0.2]))
cost, grad = costFunction(test_theta, X, y)

print('\nCost at test theta : ', cost.ravel())
print('Expected cost (approx): 0.218')
print('Gradient at test theta :', grad.ravel())
print('Expected gradients (approx): \t0.043\t 2.5662\t 2.647')

''' Part 3: Optimizing using fminunc '''
params = np.zeros([n + 1, 1])
args = (X, y)

# uUse Newton Conjugate Gradient algorithm to obtain the optimal theta
res = op.minimize(fun=costFunc, x0=params, args=args, method='TNC', jac=gradFunc)
cost, theta = res.fun, res.x

print('\nCost at theta found by fminunc: ', cost)
print('Expected cost (approx): 0.203')
print('theta: ', theta)
print('Expected theta (approx): \t-25.161\t 0.206\t 0.201')

# Plot boundary
x1 = np.arange(min(X[:, 1]), max(X[:, 1]), 1)
x2 = (-theta[0] - theta[1] * x1) / theta[2]
plt.plot(x1, x2, color='blue')
# plt.show()

''' Part 4: Predict and Accuracies '''
prob = h(np.array(([1, 45, 85])), theta)
print('\nFor a student with scores 45 and 85, we predict an adimission probability of ', prob)
print('Expected value: 0.775 +/- 0.002')

p = np.where(h(X, theta) > 0.5, 1.0, 0.0)
print('Train accuracy: ', np.mean(p == y.flatten()) * 100, '%')
print('Expected accuracy (approx): 89.0 %\n')