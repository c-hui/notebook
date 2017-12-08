"""
This is a python implement of the programming assignment in machine learning by Andrew Ng.
The programming assignment implements the backpropagation algorithm for neural networks and apply it
to the task of hand-written digit recognition
"""

import numpy as np
import scipy.io as sio
import random
from matplotlib import pyplot as plt
import scipy.optimize as op

def display_data(X, example_width = None):
    """Display 2D data in a nice grid
 
    displays 2D data stored in X in a nice grid.
    """

    # Set example_width automatically if not passed in
    if example_width is None:
        example_width = int(round(np.sqrt(X.shape[1])))

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1
    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad), \
                               pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, :]))
            pos_x = pad + j * (example_height + pad)
            pos_y = pad + i * (example_width + pad)
            display_array[ pos_x : pos_x + example_height, pos_y : pos_y + example_width] = \
            X[curr_ex, :].reshape(example_height, example_width) / max_val
            curr_ex = curr_ex + 1
        if curr_ex >= m:
            break 

    plt.figure()
    # Display Image
    plt.imshow(display_array.T, cmap='gray')
    # Do not show axis
    plt.axis("off")
    plt.show()
    
def sigmoid(z):
    """Compute sigmoid function
    
    computes the sigmoid of z.    
    """
    return 1 / (1 + np.exp(-z))
    
def sigmoid_gradient(z):
    """returns the gradient of the sigmoid function evaluated at z
    
    computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element.
    """
    return sigmoid(z)*(1-sigmoid(z))
    


def nn_cost_function(nn_params, input_layer_size,\
        hidden_layer_size, num_labels, X, y, Lambda):
    """Implements the neural network cost function for a two layer neural network
    which performs classification
    
    computes the cost and gradient of the neural network. The parameters for the
    neural network are "unrolled" into the vector nn_params and need to be 
    converted back into the weight matrices. 
    
    Returns:
        The returned parameter grad should be a "unrolled" vector of the
        partial derivatives of the neural network.
    """
    
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, (input_layer_size + 1)))
    
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, (hidden_layer_size + 1)))
    
    # Setup some useful variables
    m = X.shape[0]                
    
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    h = a3
    y = (np.asarray([range(1, num_labels+1)]*m) == y).astype(int)
    J = -y*np.log(h)-(1-y)*np.log(1-h)
    J = np.sum(J)/m
    J = J + (np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2)) * Lambda/2/m
    
    delta3 = a3 - y
    delta2 = np.dot(delta3, Theta2[:, 1:]) * sigmoid_gradient(z2)
    Delta1 = np.dot(delta2.T, a1)
    Delta2 = np.dot(delta3.T, a2)
    Theta1_grad = Delta1/m
    Theta2_grad = Delta2/m
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + Theta1[:, 1:]*Lambda/m
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + Theta2[:, 1:]*Lambda/m
    
    # Unroll gradients
    grad = np.hstack((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return J, grad

def rand_initialize_weights(L_in, L_out):
    """Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections
    
    Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    the first column of W handles the "bias" terms
    """
    return np.random.random((L_out, 1 + L_in)) * 2 * 0.12 - 0.12

def debug_initialize_weights(fan_out, fan_in):
    """intialize the weights of a layer with fan_in
    incoming connections and fan_out outgoing connections using a fixed
    strategy, this will help you later in debugging
    
    """
    
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    return np.sin(np.arange(fan_out * (1 + fan_in))).reshape(fan_out, 1 + fan_in, order='F')/10
    
def compute_numerical_gradient(J, theta):
    """Computes the gradient using "finite differences" and gives us a
    numerical estimate of the gradient.
    
    computes the numerical gradient of the function J around theta.
    Calling y = J(theta) should return the function value at theta.
    """   
    
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb.ravel()[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad.ravel()[p] = (loss2 - loss1) / (2*e)
        perturb.ravel()[p] = 0
    return numgrad

def check_nn_gradients(Lambda=None):
    """Creates a small neural network to check the backpropagation gradients
    
    Creates a small neural network to check the backpropagation gradients,
    it will output the analytical gradients produced by your backprop code
    and the numerical gradients (computed using compute_numerical_gradient).
    These two gradient computations should result in very similar values.
    """

    if Lambda is None:
        Lambda = 0

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    # Reusing debug_initialize_weights to generate X
    X  = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(range(1, m+1), num_labels).reshape(-1, 1)
    
    # Unroll parameters
    nn_params = np.hstack((Theta1.flatten(), Theta2.flatten()))
    
    # Short hand for cost function
    cost_func = lambda p: nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, \
                                            X, y, Lambda)

    cost, grad = cost_func(nn_params)
    numgrad = compute_numerical_gradient(lambda p: cost_func(p)[0], nn_params)
    
    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar. 
    for i in range(grad.shape[0]):
        print(numgrad[i], grad[i])
    print('The above two columns you get should be very similar.\n' \
            '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')
    
    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

    print('If your backpropagation implementation is correct, then \n', \
         'the relative difference will be small (less than 1e-9). \n', \
         '\nRelative Difference: ', diff)

def predict(Theta1, Theta2, X):
    """Predict the label of an input given a trained neural network

    output the predicted label of X given the trained weights of a 
    neural network (Theta1, Theta2)
    """
    # Useful values
    m = len(X)
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    z3 = np.hstack((np.ones((m, 1)), a2)).dot(Theta2.T)
    a3 = sigmoid(z3)
    p = np.argmax(a3, 1) + 1
    return p.reshape(-1, 1)

    
if __name__=="__main__":
    ## Setup the parameters you will use for this exercise
    input_layer_size  = 400;  # 20x20 Input Images of Digits
    hidden_layer_size = 25;   # 25 hidden units
    num_labels = 10;          # 10 labels, from 1 to 10   
                              # (note that we have mapped "0" to label 10)
                         
    # =========== Part 1: Loading and Visualizing Data =============

    # Load Training Data
    print('Loading and Visualizing Data ...')

    data = sio.loadmat('ex4data1.mat') 
    # training data stored in arrays X, y
    X = data['X']
    y = data['y']
    m = X.shape[0]

    # Randomly select 100 data points to display
    rand_indices = random.sample(range(m), 100)
    sel = X[rand_indices, :]

    display_data(sel)                         

    # ================ Part 2: Loading Parameters ================

    print('\nLoading Saved Neural Network Parameters ...')

    # Load the weights into variables Theta1 and Theta2
    weights = sio.loadmat('ex4weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    
    # Unroll parameters 
    nn_params = np.hstack((Theta1.flatten(), Theta2.flatten()))
    
    # ================ Part 3: Compute Cost (Feedforward) ================

    print('\nFeedforward Using Neural Network ...')

    # Weight regularization parameter (we set this to 0 here).
    Lambda = 0
    
    J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, \
                    num_labels, X, y, Lambda)
    
    print('Cost at parameters (loaded from ex4weights): ', J, \
            '\n(this value should be about 0.287629)')
    
    ## =============== Part 4: Implement Regularization ===============
    
    print('\nChecking Cost Function (w/ Regularization) ... ')
    
    # Weight regularization parameter (we set this to 1 here).
    Lambda = 1
    
    J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, \
                    num_labels, X, y, Lambda)
    
    print('Cost at parameters (loaded from ex4weights): ', J,
            '\n(this value should be about 0.383770)')

    ## ================ Part 5: Sigmoid Gradient  ================
    
    print('\nEvaluating sigmoid gradient...')
    
    g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
    print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:')
    print(g)
    print('\n')
    
    ## ================ Part 6: Initializing Pameters ================
    
    print('\nInitializing Neural Network Parameters ...')
    
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    
    # Unroll parameters
    initial_nn_params = np.hstack((initial_Theta1.flatten(), initial_Theta2.flatten()))
    
    ## =============== Part 7: Implement Backpropagation ===============

    print('\nChecking Backpropagation... ')
    
    #  Check gradients by running check_nn_gradients
    check_nn_gradients()

    ## =============== Part 8: Implement Regularization ===============
    
    print('\nChecking Backpropagation (w/ Regularization) ... ')
    
    #  Check gradients by running check_nn_gradients
    
    Lambda = 3;
    check_nn_gradients(Lambda)
    
    # Also output the costFunction debugging values
    debug_J, _  = nn_cost_function(nn_params, input_layer_size, \
                            hidden_layer_size, num_labels, X, y, Lambda)
    
    print('\n\nCost at (fixed) debugging parameters (w/ lambda =', Lambda, '):', debug_J, \
            '\n(for lambda = 3, this value should be about 0.576051)\n')
    
    ## =================== Part 8: Training NN ===================
    
    print('\nTraining Neural Network... ')
    
    #  You should also try different values of lambda
    Lambda = 1
    
    res = op.minimize(fun=nn_cost_function, x0=initial_nn_params, \
                          args=(input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)\
                          , jac=True, method = 'TNC', options={'maxiter': 200})
    nn_params, cost = res.x, res.fun
    
    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, (input_layer_size + 1)))
    
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, (hidden_layer_size + 1)))
        
    ## ================= Part 9: Visualize Weights =================
  
    print('\nVisualizing Neural Network... ')
    
    display_data(Theta1[:, 1:])
    
    ## ================= Part 10: Implement Predict =================
    
    pred = predict(Theta1, Theta2, X)
    
    print('\nTraining Set Accuracy: ', np.mean((pred==y).astype(int)) * 100)