import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from deep_l_proj.deep_learning import *

np.random.seed(1)
# load the data
train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data()

# reshape th images by vectorizing them
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

#Standardize data to have values between 0 & 1 by dividing them by 255
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

#Setting hidden layer dimensions(Constant)
layer_dims = [12288, 20, 7, 5, 1] #4 layer model

#Building a L Layer model for classifying the images
def L_layer_model(X, Y, layer_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):

    np.random.seed(1)
    costs = []  #keep track of the cost

    parameters = initialize_parameters_deep(layer_dims)

    #Loop (Gradient Descent) reduce the cost to improve accuracy
    for i in range(0, num_iterations):

        # Forward propagation
        AL, caches = L_layer_forward(X, parameters)

        #compute the cost
        cost = compute_cost(AL, Y)

        #Backward propagation
        grads = L_model_backward(AL, Y, caches)

        #update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        #print the cost every 100 training iterations
        if print_cost and i%100 == 0:
            print("Cost after %i: %f" %(i, cost))
        if print_cost and i%100 == 0:
            costs.append(cost)

    #plot the costs
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("Number of iterations")
    plt.title("Learning Rate = "+str(learning_rate))
    plt.show()

    return parameters

#get the value of parameters after training (N) iterations time
parameters = L_layer_model(train_x, train_y_orig, layer_dims, num_iterations=2500, print_cost = True)

pred_test = predict(test_x, test_y_orig, parameters)