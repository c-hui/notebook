"""
This is a python implementation of the programming assignment in machine learning by Andrew Ng.
The programming assignment is about logistic regression.
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as op

def plot_data(X, y):
    """Plots the data points X and y into a new figure
    
    plots the data points with + for the positive examples and o for 
    the negative examples. X is assumed to be a Mx2 matrix.
    """
    
    plt.figure()
    # Find Indices of Positive and Negative Examples
    pos = (y==1).T[0]
    neg = (y==0).T[0]
    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=2, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)
    

def sigmoid(z):
    """Compute sigmoid function
    
    computes the sigmoid of z.    
    """
    return 1 / (1 + np.exp(-z))
    
def cost_function(theta, X, y):
    """Compute cost and gradient for logistic regression
    
    computes the cost of using theta as the parameter for logistic
    regression and the gradient of the cost w.r.t. to the parameters.
    """

    # Initialize some useful values
    m, n = X.shape
    theta = theta.reshape(n, 1)
    h = sigmoid(X.dot(theta))
    J = np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m
    grad = X.T.dot(h-y)/m
    return J, grad.ravel()

def plot_decision_boundary(theta, X, y):
    """Plots the data points X and y into a new figure with
    the decision boundary defined by theta
    
    plots the data points with + for the positive examples 
    and o for the negative examples. X is assumed to be a either 
    1) Mx3 matrix, where the first column is an all-ones column for the 
        intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    """
    # Plot Data
    plot_data(X[:,1:3], y)
    
    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,2])+2])

        # Calculate the decision boundary line
        plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)
    
        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        u, v = np.meshgrid(u, v)
        z = np.zeros(u.shape)
        
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                z[i, j] = map_feature(np.array([u[i, j]]), np.array([v[i, j]])).dot(theta)
                
        plt.contour(u, v, z, [0], linewidth=2)
    
def predict(theta, X):
    """Predict whether the label is 0 or 1 using learned logistic 
    regression parameters theta
    
    computes the predictions for X using a threshold at 0.5 
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """
    
    return (sigmoid(X.dot(theta)) >= 0.5).astype(int).reshape(-1,1)
  
def map_feature(X1, X2):
    """Feature mapping function to polynomial features

    maps the two input features to quadratic features used 
    in the regularization exercise.
    
    Inputs X1, X2 must be the same size.

    Returns:
        a new feature array with more features, comprising of 
        X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """

    degree = 6
    out = np.ones((len(X1),1))
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.column_stack((out, (X1**(i-j))*(X2**j)))
    return out
  
if __name__=="__main__":
    # Load Data
    # The first two columns contains the exam scores and the third column
    # contains the label.

    data = np.loadtxt('ex2data1.txt', delimiter=',')
    X = data[:, [0, 1]]
    y = data[:, 2].reshape(-1, 1)
    
    # ==================== Part 1: Plotting ====================
    #  We start the exercise by first plotting the data to understand the 
    #  the problem we are working with.

    print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

    plot_data(X, y)
    # Labels and Legend
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    # Specified in plot order
    plt.legend(['Admitted', 'Not admitted'])
    plt.show()

    print('\n')
    
    # ============ Part 2: Compute Cost and Gradient ============

    #  Setup the data matrix appropriately, and add ones for the intercept term
    m, n = X.shape

    # Add intercept term to x and X_test
    X = np.hstack([np.ones((m, 1)), X])

    # Initialize fitting parameters
    initial_theta = np.zeros((n + 1, 1))

    # Compute and display initial cost and gradient
    cost, grad = cost_function(initial_theta, X, y)

    print('Cost at initial theta (zeros): ', cost)
    print('Expected cost (approx): 0.693')
    print('Gradient at initial theta (zeros): ')
    print(grad)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')

    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array([-24, 0.2, 0.2]).reshape(-1,1)
    cost, grad = cost_function(test_theta, X, y)

    print('\nCost at test theta: ', cost)
    print('Expected cost (approx): 0.218')
    print('Gradient at test theta: ')
    print(grad)
    print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')

    print('\n')

    # ============= Part 3: Optimizing using scipy.optimize.minimize =============

    #  Run minimize to obtain the optimal theta
    res = op.minimize(fun=cost_function, x0=initial_theta, \
        args=(X, y), jac=True, method = 'TNC', options={'maxiter': 400})
    theta = res.x
    cost = res.fun

    # Print theta to screen
    print('Cost at theta found by minimize: ', cost)
    print('Expected cost (approx): 0.203')
    print('theta: ')
    print(theta)
    print('Expected theta (approx):')
    print(' -25.161\n 0.206\n 0.201')

    # Plot Boundary
    plot_decision_boundary(theta, X, y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()
    print('\n')

    # ============== Part 4: Predict and Accuracies ==============

    #  Predict probability for a student with score 45 on exam 1 
    #  and score 85 on exam 2 

    prob = sigmoid(np.array([1, 45, 85]).dot(theta))
    print('For a student with scores 45 and 85, we predict an admission probability of ', prob)
    print('Expected value: 0.775 +/- 0.002\n')

    # Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: ', np.mean((p == y).astype(int)) * 100)
    print('Expected accuracy (approx): 89.0')