"""
This is a python implementation of the programming assignment in machine learning by Andrew Ng.
The programming assignment is about one-vs-all logistic regression to recognize hand-written digits.
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

    m, n = X.shape
    
    # Set example_width automatically if not passed in
    if example_width is None:
        example_width = int(round(np.sqrt(n)))

    # Compute rows, cols
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

def lr_cost_function(theta, X, y, Lambda):
    """Compute cost and gradient for logistic regression with regularization
    
    compute the cost of using theta as the parameter for regularized logistic
    regression and the gradient of the cost w.r.t. to the parameters. 
    """
    # Initialize some useful values
    m, n = X.shape
    theta = theta.reshape(-1, 1)

    h = sigmoid(X.dot(theta))
    J = np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m + Lambda*np.sum(theta[1:]**2)/m/2
    grad = X.T.dot(h-y)/m
    grad[1:] = grad[1:] + Lambda*theta[1:]/m
    return J, grad.ravel()

def one_vs_all(X, y, num_labels, Lambda):
    """train multiple logistic regression classifiers and returns all
    the classifiers in a matrix all_theta, where the i-th row of all_theta 
    corresponds to the classifier for label i
    """
    # Some useful variables
    m, n = X.shape

    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.hstack((np.ones((m, 1)), X))
    for i in range(num_labels):
        initial_theta = np.zeros((n + 1, 1))
        res = op.minimize(fun=lr_cost_function, x0=initial_theta, \
                          args=(X, (y == i+1).astype(int), Lambda), jac=True, method = 'TNC', options={'maxiter': 50})
        all_theta[i,:] = res.x
    return all_theta

def predict_one_vs_all(all_theta, X):
    """Predict the label for a trained one-vs-all classifier. The labels 
    are in the range 1..K, where K = len(all_theta).
    
    return a vector of predictions for each example in the matrix X. Note that X
    contains the examples in rows. all_theta is a matrix where the i-th row is a 
    trained logistic regression theta vector for the i-th class. You should set
    p to a vector of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1,
    3, 1, 2 for 4 examples)
    """
    m = X.shape[0]
    num_labels = len(all_theta)

    # Add ones to the X data matrix
    X = np.hstack((np.ones((m, 1)), X))
    
    p = np.argmax(sigmoid(X.dot(all_theta.T)), 1) + 1
    return p.reshape(-1, 1)
    
if __name__=="__main__":
    # Setup the parameters you will use for this part of the exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    num_labels = 10          # 10 labels, from 1 to 10
                             # (note that we have mapped "0" to label 10)

    # =========== Part 1: Loading and Visualizing Data =============

    # Load Training Data
    print('Loading and Visualizing Data ...')

    data1 = sio.loadmat('ex3data1.mat') 
    # training data stored in arrays X, y
    X = data1['X']
    y = data1['y']
    m = X.shape[0]

    # Randomly select 100 data points to display
    rand_indices = random.sample(range(m), 100)
    sel = X[rand_indices, :]

    display_data(sel)
    
    # ============ Part 2a: Vectorize Logistic Regression ============
    
    # Test case for lrCostFunction
    print('\nTesting lrCostFunction() with regularization')
    
    theta_t = np.array([-2, -1, 1, 2]).reshape(-1, 1)
    X_t = np.hstack((np.ones((5,1)), np.arange(1, 16).reshape(5, 3,  order='F')/10))
    y_t = np.array([1,0,1,0,1]).reshape(-1, 1)
    lambda_t = 3
    J, grad = lr_cost_function(theta_t, X_t, y_t, lambda_t)
    
    print('\nCost:', J)
    print('Expected cost: 2.534819')
    print('Gradients:')
    print(grad)
    print('Expected gradients:')
    print(' 0.146561\n -0.548558\n 0.724722\n 1.398003')
    
    # ============ Part 2b: One-vs-All Training ============
    print('\nTraining One-vs-All Logistic Regression...')
    
    Lambda = 0.1
    all_theta = one_vs_all(X, y, num_labels, Lambda)
    
    # ================ Part 3: Predict for One-Vs-All ================
    
    pred = predict_one_vs_all(all_theta, X)
    
    print('\nTraining Set Accuracy:', np.mean((pred==y).astype(int)) * 100)