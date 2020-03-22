> See notes on [https://my-thistledown.com/2020/03/04/ML-Ng-3/](https://my-thistledown.com/2020/03/04/ML-Ng-3/)

## Ex2: Logistic Regression👨‍💻

#### Ex2.1 Logistic Regression

**Instruction:**
In this part of the exercise, you will build a logistic regression model to predict whether a student gets admitted into a university.

**Code:**

```python
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
plt.show()

''' Part 4: Predict and Accuracies '''
prob = h(np.array(([1, 45, 85])), theta)
print('\nFor a student with scores 45 and 85, we predict an adimission probability of ', prob)
print('Expected value: 0.775 +/- 0.002')

p = np.where(h(X, theta) > 0.5, 1.0, 0.0)
print('Train accuracy: ', np.mean(p == y.flatten()) * 100, '%')
print('Expected accuracy (approx): 89.0 %\n')
```

**Output:**

- Console
    ![image.png](https://i.loli.net/2020/03/09/kEjqygun1US25Xc.png)
- Training data with decision boundary
    ![Figure_1.png](https://i.loli.net/2020/03/09/8PeaQu5iAqz3kFm.png)

#### Ex2.2 Regularized Logistic Regression

**Instruction:**
In this part of the exercise, you will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly.

**Code:**

```python
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
```

**Output:**

- Console ((λ = 1)
    ![image.png](https://i.loli.net/2020/03/10/mL5Uq16nKgifRuB.png)
- Training data with decision boundary (λ = 1)
    ![image.png](https://i.loli.net/2020/03/10/K29fFZimhoAuxrb.png)
- Too much regularization (Underfitting) (λ = 100)
    ![image.png](https://i.loli.net/2020/03/10/eG36QAIqsiwbZ45.png)



