"""
This is a python implement of the programming assignment in machine learning by Andrew Ng.
The programming assignment is about linear regression with multiple variables.
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg

def feature_normalize(X):
    """FEATURENORMALIZE Normalizes the features in X 

    returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma

def compute_cost_multi(X, y, theta):
    """COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    
    computes the cost of using theta as the parameter for linear regression
    to fit the data points in X and y
    """
    # Initialize some useful values
    m = len(y) # number of training examples
    J = (X.dot(theta)-y).T.dot(X.dot(theta)-y)/2/m;
    return J[0][0]
    
def gradient_descent_multi(X, y, theta, alpha, num_iters):
    """GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    
    updates theta by taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters):
        theta = theta - alpha*(X.T.dot(X.dot(theta)-y))/m
        J_history[iter] = compute_cost_multi(X, y, theta)
    
    return theta, J_history
    
def normal_eqn(X, y):
    """NORMALEQN Computes the closed-form solution to linear regression 

    computes the closed-form solution to linear regression using the normal equations.
    """
    theta = linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta
    
if __name__ == "__main__":
    # ================ Part 1: Feature Normalization ================

    print('Loading data ...')

    # Load Data
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2].reshape(-1, 1)
    m = len(y)

    # Print out some data points
    print('First 10 examples from the dataset:')
    for i in range(10):
        print(' x = [%.0f %.0f], y = %.0f' % (X[i, 0], X[i, 1], y[i][0]))

    print()

    # Scale features and set them to zero mean
    print('Normalizing Features ...')

    X, mu, sigma = feature_normalize(X)

    # Add intercept term to X
    X = np.hstack([np.ones((m, 1)), X])
    
    # ================ Part 2: Gradient Descent ================
    
    print('Running gradient descent ...');

    # Choose some alpha value
    alpha = 0.01
    num_iters = 400

    # Init Theta and Run Gradient Descent 
    theta = np.zeros((3, 1))
    theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    plt.figure()
    plt.plot(range(0, len(J_history)), J_history, '-b', linewidth=2);
    plt.xlabel('Number of iterations');
    plt.ylabel('Cost J');
    plt.show()

    # Display gradient descent's result
    print('Theta computed from gradient descent:')
    print(theta)
    print()

    # Estimate the price of a 1650 sq-ft, 3 br house
    # Recall that the first column of X is all-ones. Thus, it does
    # not need to be normalized.
    price = np.hstack((np.array([1]), (np.array([1650, 3]) - mu)/sigma)).dot(theta)

    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):')
    print(price[0])

    print()
    
    # ================ Part 3: Normal Equations ================

    print('Solving with normal equations...')

    # Load Data
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2].reshape(-1, 1)
    m = len(y)

    # Add intercept term to X
    X = np.hstack([np.ones((m, 1)), X])

    # Calculate the parameters from the normal equation
    theta = normal_eqn(X, y)

    # Display normal equation's result
    print('Theta computed from the normal equations:')
    print(theta)
    print()


    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.array([1, 1650, 3]).dot(theta)

    print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):')
    print(price[0])
    
