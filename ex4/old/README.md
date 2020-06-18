> See notes on [https://my-thistledown.com/2020/03/10/ML-Ng-4/](https://my-thistledown.com/2020/03/10/ML-Ng-4/)

## Ex4: Neural Networks Learning👨‍💻

#### Ex4.1&4.2 Neural Networks&Backpropagation

**Instruction:**

In the previous exercise, you implemented feedforward propagation for neural networks and used it to predict handwritten digits with the weights we provided. In this  exercise, you will implement the backpropagation algorithm to learn the parameters for the neural network. 

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

# Implements the neural network cost function for a two layer neural network
# which performs classification
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, reg_param):
    # Reshape nn_params back into the parameters Theta1 and Theta2,
    # the weight matrices for our 2 layer neural network
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].\
        reshape([hidden_layer_size, (input_layer_size + 1)], order='F')
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].\
        reshape([num_labels, (hidden_layer_size + 1)], order='F')

    # Setup some useful variables
    m = X.shape[0]

    # Feedforward Propagation
    a1 = np.c_[np.ones([X.shape[0], 1]), X]
    z2 = np.dot(a1, Theta1.T)
    a2 = np.c_[np.ones([z2.shape[0], 1]), sigmoid(z2)]
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    y = np.eye(num_labels)[y.reshape(-1) - 1]

    # Add regularization term
    cost1 = - np.sum((np.log(a3) * y) + np.log(1 - a3) * (1 - y)) / m
    cost2 = 0.5 * reg_param * (np.sum(np.square(Theta1[:, 1:])) +
                               np.sum(np.square(Theta2[:, 1:]))) / m
    cost = cost1 + cost2

    # Backpropagation
    delta3 = a3 - y
    delta2 = np.dot(delta3, Theta2)[:, 1:] * sigmoidGradient(z2)
    Delta1, Delta2 = np.dot(delta2.T, a1), np.dot(delta3.T, a2)

    # Add regularization to gradient
    Theta1_grad = Delta1 / m
    Theta1_grad[:, 1:] += reg_param * Theta1[:, 1:] / m
    Theta2_grad = Delta2 / m
    Theta2_grad[:, 1:] += reg_param * Theta2[:, 1:] / m

    # Unroll gradients
    grad = np.r_[Theta1_grad.ravel(order='F'), Theta2_grad.ravel(order='F')]

    return cost, grad

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# sigmoidGradient() returns the gradient of the sigmoid function evaluated at z
def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Randomly initialize the weights of a layer with L_in incoming connections and L_out outgoing connections
def randInitWeights(L_in, L_out, epsilon_init=0.12):
    # epsilon_init = np.sqrt(6) / np.sqrt(L_in + L_out)
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W

# Initialize the weights of a layer with fan_in incoming connections and fan_out
# outgoing connections using a fixed strategy, this will help you later in debugging
def debugInitWeights(fan_in, fan_out):
    # Initialize W using "sin", this ensures that W is always of the same values
    # and will be useful for debugging
    W = np.sin(np.arange(1, 1 + (1 + fan_in) * fan_out)) / 10.0
    W = W.reshape(fan_out, 1 + fan_in, order='F')
    return W

# Computes the gradient using "finite differences" nd gives us
# a numerical estimate of the gradient.
def computeNumericalGradient(costFunc, nn_params, e=1e-4):
    numgrad = np.zeros(nn_params.shape)
    perturb = np.diag(e * np.ones(nn_params.shape))
    for i in range(nn_params.size):
        loss1, _ = costFunc(nn_params - perturb[:, i])
        loss2, _ = costFunc(nn_params + perturb[:, i])
        numgrad[i] = (loss2 - loss1) / (2 * e)
    return numgrad

# Creates a small neural network to check the backpropagation gradients
def checkNNGradients(nnCostFunction, reg_param=0, visually_examine=False):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitWeights(input_layer_size, hidden_layer_size)
    Theta2 = debugInitWeights(hidden_layer_size, num_labels)
    # Reusing debugInitWeights to generate X
    X = debugInitWeights(input_layer_size - 1, m)
    y = np.arange(1, 1 + m) % num_labels

    # Unroll parameters
    nn_params = np.r_[Theta1.ravel(order='F'), Theta2.ravel(order='F')]

    # Short hand for cost function
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, reg_param)
    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.
    # The two columns you get should be very similar
    if (visually_examine):
        print(np.stack([numgrad, grad], axis=1))
        print(' - The above two columns you get should be very similar.')
        print(' - (Left: Your Numerical Gradient; Right: Analytical Gradient)')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming that you used EPSILON = 1e-4
    # in 'computeNumericalGradient()', then diff below should be less that 1e-9
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print(' - If your backpropagation implementation is correct, \n'
          ' - then the relative difference will be small (less than 1e-9).\n'
          ' - Relative Difference:', diff)

# Predict the label of an input given a trained neural network
def predict(theta1, theta2, X):
    a1 = np.c_[np.ones([X.shape[0], 1]), X]
    z2 = np.dot(a1, theta1.T)
    a2 = np.c_[np.ones([z2.shape[0], 1]), sigmoid(z2)]
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    pred = np.argmax(a3, axis=1)
    return pred + 1

# Visualize what the representations captured by the hidden units
def plotHiddenUnits(X, figsize=(8, 8)):
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        m, n = 1, X.size
        X = X[None] # Promote to a 2-D array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    size = int(np.sqrt(n))
    rows = cols = int(np.sqrt(m))

    fig1, ax1 = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=figsize)
    ax1 = [ax1] if m == 1 else ax1.ravel()
    for i, ax in enumerate(ax1):
        ax.imshow(X[i].reshape([size, size], order='F'), cmap='Greys')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Visualization of Hidden Units', fontsize=24)

# Setup the parameters
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10         # 10 labels, form 1 to 10 (mapping '0' to label '10')

''' Part 1: Loading and Visualizing Data '''
print('1. Loading and Visualizing Data ... \n')
# Load Training Data
X, y, m = loadData('ex4data1.mat')

# Randomly select 100 data points to display
randomDisplay(X, 100, transpose=True)

''' Part 2: Loading Parameters '''
# Load the weights into variables Theta1 and Theta2
print('2. Loading Saved Neural Network Parameters ...\n')
weights = scio.loadmat('ex4weights.mat')
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# Unroll parameters
nn_params = np.r_[Theta1.ravel(order='F'), Theta2.ravel(order='F')]

''' Part 3: Compute Cost (Feedforward) '''
print('3. Feedforward Using Neural Network ...')
# Set weight regularization parameter to 0 (i.e. no regularization)
cost, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, reg_param=0)
print(' - Cost at parameters (loaded form ex4weights.mat): ', cost)
print(' - (This value should be about 0.287629.)\n')

''' Part 4: Implement Regularization '''
print('4. Checking Cost Function (w/ Regularization) ...')
# Weight regularization parameter (we set this to 1 here)
cost, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                            num_labels, X, y, reg_param=1)
print(' - Cost at parameters (loaded form ex4weights.mat): ', cost)
print(' - (This value should be about 0.383770.)\n')

''' Part 5: Sigmoid Gradient '''
print('5. Evaluating sigmoid gradient ...')
g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
print(' - Sigmoid gradient evaluated at [-1, -0.5, 0, 0.5, 1]:\n -', g)

''' Part 6: Initializing Parameters '''
print('\n6. Initializing Neural Network Parameters ...')
init_Theta1 = randInitWeights(input_layer_size, hidden_layer_size)
init_Theta2 = randInitWeights(hidden_layer_size, num_labels)

# Unroll parameters
init_nn_params = np.r_[init_Theta1.ravel(order='F'), init_Theta2.ravel(order='F')]

''' Part 7: Implement Backpropagation '''
print('\n7. Checking Backpropagation ...')

# Check gradients by running checkNNGradients
checkNNGradients(nnCostFunction, reg_param=0, visually_examine=True)

''' Part 8: Implement Regularization '''
print('\n8. Checking Backpropagation (w/ Regularization) ...')
debug_lambda = 3.0
checkNNGradients(nnCostFunction, reg_param=debug_lambda, visually_examine=False)

debug_cost, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                               num_labels, X, y, debug_lambda)
print(' - Cost at (fixed) debugging parameters (w/ lambda = ', debug_lambda, '):',
      debug_cost, '\n - (for lambda = 3, this value should be about 0.576051)\n')

''' Part 9: Training Neural Network '''
print('9. Training Neural Network ...')
# After you have completed the assignment, change the MaxIter
# to a larger value to see how training helps
options = {'maxiter': 400}

# You should also try different values of lambda
train_lambda = 3.0

# Create "short hand" for the cost function to be minimized
costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                    num_labels, X, y, train_lambda)

# Now, 'costFunc' is a function that takes in only one argument
# (the neural network parameters)
res = opt.minimize(costFunc, init_nn_params, jac=True, method='TNC', options=options)
nn_params = res.x

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].\
    reshape([hidden_layer_size, input_layer_size + 1], order='F')
Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].\
    reshape([num_labels, hidden_layer_size + 1], order='F')

# Implement predict
y_predict = predict(Theta1, Theta2, X)
print(' - When lambda = {:}, MaxIter = {:}, Training set accuracy = {:.2f}%\n'.
      format(train_lambda, options['maxiter'], np.mean(y_predict == y.flatten()) * 100))

''' Part 10: Visualize Weights '''
print('10. Visualizing Neural Network ...\n')
plotHiddenUnits(Theta1[:, 1:])
plt.show()
```

**Outputs:**

- Console
    ![image.png](https://i.loli.net/2020/03/22/5kTym3FtB9hNHKx.png)
- Randomly select 100 data points to display
    ![Figure_1.png](https://i.loli.net/2020/03/22/TPfSY1IXHJGWqNL.png)
- Visualization of hidden units
    ![Figure_2.png](https://i.loli.net/2020/03/22/kPufBb16UHv9wm3.png)



















