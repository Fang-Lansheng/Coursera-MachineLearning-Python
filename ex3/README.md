> See notes on [https://my-thistledown.com/2020/03/10/ML-Ng-4/](https://my-thistledown.com/2020/03/10/ML-Ng-4/)

## Ex3: Multi-class Classification and Neural Networks👨‍💻

#### Ex3.1 Multi-class Classification

**Instruction:**
For this exercise, you will use logistic regression and neural networks to recognize handwritten digits (from 0 to 9). Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. This exercise will show you how the methods you’ve learned can be used for this
classification task.

**Code:**

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.optimize as opt

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

# Compute cost and gradient for logistic regression with regularization
def lrCostFunction(theta, X, y, lam):
    m = len(y)
    cost1 = - np.sum(y * np.log(sigmoid(np.dot(X, theta))) + (1 - y) * np.log(1 - sigmoid(np.dot(X, theta)))) / m
    cost2 = 0.5 * lam * np.dot(theta[1:].T, theta[1:]) / m
    cost = cost1 + cost2
    grad = np.dot(X.T, sigmoid(np.dot(X, theta)) - y) / m
    grad[1:] += (lam * theta / m)[1:]
    return cost, grad

def costFunc(param, *args):
    X, y, lam = args
    m, n = X.shape
    theta = param.reshape([n, 1])
    cost1 = - np.sum(y * np.log(sigmoid(np.dot(X, theta))) + (1 - y) * np.log(1 - sigmoid(np.dot(X, theta)))) / m
    cost2 = 0.5 * lam * np.dot(theta[1:].T, theta[1:]) / m
    return cost1 + cost2

def gradFunc(param, *args):
    X, y, lam = args
    m, n = X.shape
    theta = param.reshape(-1, 1)
    grad = np.dot(X.T, sigmoid(np.dot(X, theta)) - y) / m
    grad[1:] += (lam * theta / m)[1:]
    return grad.ravel()

# oneVsAll() trains multiple logistic regression classifiers and returns all
# the classifiers in a matrix all_theta, where the i-th row of all_theta
# corresponds to the classifier for label i
def oneVsAll(X, y, num_labels, lam):
    m, n = X.shape
    all_theta = np.zeros([num_labels, n + 1])
    X = np.c_[np.ones([m, 1]), X]
    for i in range(1, num_labels + 1):
        params = np.zeros(n + 1)
        args = (X, y == i, lam)
        res = opt.minimize(fun=costFunc, x0=params, args=args, method='TNC', jac=gradFunc)
        all_theta[i - 1, :] = res.x

    return all_theta

# Predict the label for a trained one-vs-all classifier
def predictOneVsAll(X, all_theta):
    m, n = X.shape
    X = np.c_[np.ones([m, 1]), X]
    h_argmax = np.argmax(sigmoid(np.dot(X, all_theta.T)), axis=1)
    return h_argmax + 1

# Setup the parameters
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10         # 10 labels, form 1 to 10 (note that we have mapped '0' to label '10')

''' Part 1: Loading and Visualizing Data '''
print('Loading and Visualizing Data ... ')
# Load Training Data
X, y, m = loadData('ex3data1.mat')

# Randomly select 100 data points to display
randomDisplay(X, 100)
plt.show()

''' Part 2.1: Vectorize Logistic Regression '''
# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')
theta_test = np.array(([-2], [-1], [1], [2]))
X_test = np.c_[np.ones([5, 1]), np.linspace(1, 15, 15).reshape([5, 3], order='F') / 10]
y_test = np.array(([1], [0], [1], [0], [1]))
lam_test = 3

cost, grad = lrCostFunction(theta_test, X_test, y_test, lam_test)

print('Cost: ', cost.flatten())
print('Expected cost: 2.534819')
print('Gradients: ', grad.flatten())
print('Expected gradients: 0.146561\t -0.548558\t 0.724722\t 1.398003')

''' Part 2.2: One-vs-All Training '''
print('\nTraining One-vs-All Logistic Regression...')
theta = oneVsAll(X, y, num_labels=num_labels, lam=1.0)

''' Part 3: Predict for One-Vs-All '''
y_predict = predictOneVsAll(X, all_theta=theta)
print('Training Set Accuracy: ', np.mean(y.ravel() == y_predict) * 100, '%')
```

**Output:**

- Console:
    ![image.png](https://i.loli.net/2020/03/14/ZKdwaXmAJPWcn2e.png)
- Randomly select several data points to display:
    ![image.png](https://i.loli.net/2020/03/14/DzwfdCreQkvbxX3.png)

#### Ex3.2 Neural Networks

**Instruction:**

In this part of the exercise, you will implement a neural network to recognize handwritten digits using the same training set as before. The neural network will be able to represent complex models that form non-linear hypotheses. Your goal is to implement the feedforward propagation algorithm to use our weights for prediction. 

Our neural network is shown below. It has 3 layers – an input layer, a hidden layer and an output layer. Recall that our inputs are pixel values of digit images. Since the images are of size $20×20$, this gives us 400 input layer units (excluding the extra bias unit which always outputs +1). As before, the training data will be loaded into the variables $X$ and $y$.
![image-20200314113224437](https://i.loli.net/2020/03/14/jEwbRfxuKt4517F.png)

**Code:**

```python
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
randomDisplay(X, 100, transpose=True)
plt.show()

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
```

**Output:**

- Console: 
    ![image.png](https://i.loli.net/2020/03/14/hAL8WIqmCGFYnk7.png)





































