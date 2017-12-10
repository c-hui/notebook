"""
This is a python implementation of the programming assignment in machine learning by Andrew Ng.
The programming assignment is about linear regression with one variable.
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def warm_up_exercise():
    """Example function.
    
    Returns:
        A 5x5 identity matrix
    """
    return np.eye(5, dtype=int)
    
def plot_data(x, y):
    """Plots the data points x and y into a new figure 
    
    plots the data points and gives the figure axes labels of
    population and profit.
    """
    plt.figure()
    plt.plot(x, y, 'rx', markersize=10)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    

def compute_cost(X, y, theta):
    """Compute cost for linear regression   
    
    computes the cost of using theta as the parameter for 
    linear regression to fit the data points in X and y
    """
    # Initialize some useful values
    m = len(y) # number of training examples
    J = np.sum((X.dot(theta)-y)**2)/2/m
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta
    
    updates theta by taking num_iters gradient steps with
    learning rate alpha
    """
    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        theta = theta - alpha*(X.T.dot(X.dot(theta)-y))/m
        # Save the cost J in every iteration
        J_history[iter] = compute_cost(X, y, theta)
    return theta, J_history
    
if __name__ == "__main__":
    # ==================== Part 1: Basic Function ====================
    # Complete warmUpExercise.m
    print('Running warmUpExercise ...')
    print('5x5 Identity Matrix:')
    print(warm_up_exercise())
    
    print()
    
    # ======================= Part 2: Plotting =======================
    print('Plotting Data ...')
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    X = data[:, 0]
    y = data[:, 1].reshape(-1, 1)
    m = len(y) # number of training examples

    # Plot Data
    plot_data(X, y)
    plt.show()

    print()
    
    # =================== Part 3: Cost and Gradient descent ===================

    X = np.hstack((np.ones((m, 1)), X.reshape(-1, 1))) # Add a column of ones to x
    theta = np.zeros((2, 1)) # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    print('\nTesting the cost function ...')
    # compute and display initial cost
    J = compute_cost(X, y, theta);
    print('With theta = [0 ; 0]\nCost computed = ', J)
    print('Expected cost value (approx) 32.07');

    # further testing of the cost function
    J = compute_cost(X, y, np.array([[-1] ,[2]]))
    print('\nWith theta = [-1 ; 2]\nCost computed = ', J)
    print('Expected cost value (approx) 54.24\n')

    print()

    print('\nRunning Gradient Descent ...')
    # run gradient descent
    theta, _ = gradient_descent(X, y, theta, alpha, iterations)

    # print theta to screen
    print('Theta found by gradient descent:')
    print(theta)
    print('Expected theta values (approx)')
    print(' -3.6303\n  1.1664\n');

    # Plot the linear fit
    plot_data(X[:,1], y)
    plt.plot(X[:,1], X.dot(theta), '-')
    plt.legend(['Training data', 'Linear regression'])
    plt.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.array([1, 3.5]).dot(theta)
    print('For population = 35,000, we predict a profit of ', predict1[0]*10000)
    predict2 = np.array([1, 7]).dot(theta)
    print('For population = 70,000, we predict a profit of ', predict2[0]*10000)

    print()
    
    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros(theta0_vals.shape);

    # Fill out J_vals
    for i in range(theta0_vals.shape[0]):
        for j in range(theta0_vals.shape[1]):
            t = np.array([[theta0_vals[i][j]], [theta1_vals[i][j]]])
            J_vals[i,j] = compute_cost(X, y, t)

    # Surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    plt.show()
    
    # Contour plot
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    plt.figure()
    plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
    plt.show()