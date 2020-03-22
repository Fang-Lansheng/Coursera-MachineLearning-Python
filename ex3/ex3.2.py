# -*- coding: utf-8 -*-
# @Time    : 2020/3/14 11:26
# @Author  : Thistledown
# @Email   : 120768091@qq.com
# @File    : ex3.2.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

''' Part 0: Functions and Parameters '''
# Load training data
def loadData(path):
    data = scio.loadmat(path)
    X, y = data['X'], data['y']
    m = len(y)
    return X, y, m

# Randomly select several data points to display
def randomDisplay(data, num=100, cmap='binary', transpose=True):
    sample = data[np.random.choice(len(data), num)]
    size, size1 = int(np.sqrt(num)), int(np.sqrt(sample[0].shape[0]))
    fig0, ax0 = plt.subplots(nrows=size, ncols=size, sharex=True, sharey=True, figsize=(8, 8))
    order = 'F' if transpose else 'C'
    for i in range(size):
        for j in range(size):
            ax0[i, j].imshow(sample[size * i + j].reshape([size1, size1], order=order), cmap=cmap)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(str(num)+' examples from the dataset', fontsize=24)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Predict the label of an input given a trained neural network
def predict(theta1, theta2, X):
    # a1 = np.c_[np.ones([X.shape[0], 1]), X]
    # z2 = np.dot(a1, theta1.T)
    # z2 = np.c_[np.ones([z2.shape[0], 1]), z2]
    # a2 = sigmoid(z2)
    # z3 = np.dot(a2, theta2.T)
    # a3 = sigmoid(z3)
    # pred = np.argmax(a3, axis=1)

    a1 = np.c_[np.ones([X.shape[0], 1]), X]
    z2 = np.dot(a1, theta1.T)
    a2 = np.c_[np.ones([z2.shape[0], 1]), sigmoid(z2)]
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    pred = np.argmax(a3, axis=1)
    return pred + 1

# Setup the parameters
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10         # 10 labels, form 1 to 10 (note that we have mapped '0' to label '10')

''' Part 1: Loading and Visualizing Data '''
print('Loading and Visualizing Data ... ')
# Load Training Data
X, y, m = loadData('ex3data1.mat')

# Randomly select 100 data points to display
# randomDisplay(X, 100, transpose=True)
# plt.show()

''' Part 2: Loading Parameters '''
weights = scio.loadmat('ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

''' Part 3: Implement Predict '''
y_predict = predict(theta1, theta2, X)
print('\nTraining Set Accuracy: ', np.mean(y_predict == y.flatten()) * 100, '%')

# Randomly select examples
random_index = np.random.choice(m, size=10)
print('Indexes of examples :', random_index)
for i, idx in enumerate(random_index):
    # print('Example[%d]\t is number %d,\t we predict it as %d' % (i + 1, y[idx], y_predict[idx    ]))
    print('Example[{:0>2}] is number {:^2}, we predict it as {:^2}'.
          format(str(i + 1), str(int(y[idx])), str(y_predict[idx])))
print('END')