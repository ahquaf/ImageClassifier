import numpy as np
import h5py
import matplotlib.pyplot as plt

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


# Load Datasets into model
def load_data():

    train_dataset = h5py.File('datasets/train_catvnoncat.h5',"r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

"""
2 layer deep neural network
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters
"""

# N-layer deep neural network
def initialize_parameters_deep(layers_dim):

    np.random.seed(1)
    parameters = {}
    L = len(layers_dim)

    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layers_dim[l], layers_dim[l-1]) / np.sqrt(layers_dim[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dim[l],1))

    return parameters
"""
parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("B1 = " + str(parameters["b1"]))
"""
def linear_forward(A, W, b):

    # Calculate value of Z = WX + b
    Z = np.dot(W,A) + b

    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_forward_activation(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def L_layer_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_forward_activation(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    A_last, cache = linear_forward_activation(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)

    assert(A_last.shape == (1, X.shape[1]))
    return A_last, caches

def compute_cost(A_last, Y):

    #get the shape of number of testing examples output array
    m = Y.shape[1]

    #cost = -np.sum(np.multiply(Y, np.log(A_last)) + np.multiply(1-Y,np.log(1-A_last))) / m
    cost = (1. / m) * (-np.dot(Y, np.log(A_last).T) - np.dot(1 - Y, np.log(1 - A_last).T))
    cost = np.squeeze(cost)

    assert(cost.shape == ())
    return cost

### calculate backward propagation
def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    """
    Now we calculate dW, dB, dA[l-1]
    dW = 1/m(dZ[l]*A[l-1].T)
    dB = 1/m(summation)(dZ[l]i)(where i = 1 to m)
    dA[l-1] = W[l].T * dZ[l] (T = transpose)
    """

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W.T,dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    return dA_prev, dW, db

def linear_backward_activation(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Initializing the back propagation
    dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))

    current_cache = caches[L-1]
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_backward_activation(dAL, current_cache, activation = "sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_t, dW_t, db_t = linear_backward_activation(grads["dA"+str(l+1)], current_cache,activation = "relu")
        grads["dA"+str(l)] = dA_prev_t
        grads["dW"+str(l+1)] = dW_t
        grads["db"+str(l+1)] = db_t

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - (learning_rate*grads["dW"+str(l+1)])
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - (learning_rate*grads["db"+str(l+1)])

    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_layer_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p
