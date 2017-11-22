"""
This is a python implement of the programming assignment in machine learning by Andrew Ng.
The exercise covers regularization with logistic regression.
"""

import numpy as np
from ex2 import plot_data, sigmoid, plot_decision_boundary, predict, map_feature
from matplotlib import pyplot as plt
import scipy.optimize as op

def cost_function_reg(theta, X, y, Lambda):
    """Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression
    and the gradient of the cost w.r.t. to the parameters. 
    """
    # Initialize some useful values
    m, n = X.shape
    theta = theta.reshape(n, 1)

    h = sigmoid(X.dot(theta))
    J = np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m + Lambda*np.sum(theta[1:]**2)/m/2
    grad = X.T.dot(h-y)/m
    grad[1:] = grad[1:] + Lambda*theta[1:]/m
    return J, grad



if __name__=="__main__":
    # Load Data
    #  The first two columns contains the X values and the third column
    #  contains the label (y).

    data = np.loadtxt('ex2data2.txt', delimiter=',')
    X = data[:, [0, 1]]
    y = data[:, 2].reshape(-1,1)

    plot_data(X, y);

    # Put some labels
    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    # Specified in plot order
    plt.legend(['y = 1', 'y = 0'])
    plt.show()
    
    # =========== Part 1: Regularized Logistic Regression ============
    #  In this part, you are given a dataset with data points that are not
    #  linearly separable. However, you would still like to use logistic
    #  regression to classify the data points.

    #  To do so, you introduce more features to use -- in particular, you add
    #  polynomial features to our data matrix (similar to polynomial
    #  regression).


    # Add Polynomial Features

    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X = map_feature(X[:,0], X[:,1])

    # Initialize fitting parameters
    initial_theta = np.zeros((X.shape[1], 1))

    # Set regularization parameter Lambda to 1
    Lambda = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = cost_function_reg(initial_theta, X, y, Lambda)

    print('Cost at initial theta (zeros): ', cost)
    print('Expected cost (approx): 0.693')
    print('Gradient at initial theta (zeros) - first five values only:')
    print(grad[:5])
    print('Expected gradients (approx) - first five values only:')
    print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')


    # Compute and display cost and gradient
    # with all-ones theta and Lambda = 10
    test_theta = np.ones((X.shape[1], 1))
    cost, grad = cost_function_reg(test_theta, X, y, 10)

    print('\nCost at test theta (with Lambda = 10): ', cost)
    print('Expected cost (approx): 3.16')
    print('Gradient at test theta - first five values only:')
    print(grad[:5])
    print('Expected gradients (approx) - first five values only:')
    print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n\n')



    # ============= Part 2: Regularization and Accuracies =============
    #  Optional Exercise:
    #  In this part, you will get to try different values of lambda and
    #  see how regularization affects the decision coundart

    #  Try the following values of lambda (0, 1, 10, 100).

    #  How does the decision boundary change when you vary lambda? How does
    #  the training set accuracy vary?


    # Initialize fitting parameters
    initial_theta = np.zeros((X.shape[1], 1))

    # Set regularization parameter lambda to 1 (you should vary this)
    Lambda = 1

    # Optimize
    res = op.minimize(fun=cost_function_reg, x0=initial_theta, \
        args=(X, y, Lambda), jac=True, method = 'TNC', options={'maxiter': 400})
    theta = res.x
    J = res.fun

    # Plot Boundary
    plot_decision_boundary(theta, X, y)
    plt.title('lambda = ' + str(Lambda))

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
    plt.show()
    
    # Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: ', np.mean((p == y).astype(int)) * 100)
    print('Expected accuracy (with lambda = 1): 83.1 (approx)')