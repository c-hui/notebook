"""
This is a python implement of the programming assignment in machine learning by Andrew Ng.
The programming assignment implements regularized linear regression and uses it to study
models with different bias-variance properties.
"""
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as op

def linear_reg_cost_function(X, y, theta, Lambda):
    """Compute cost and gradient for regularized linear regression
    with multiple variables
    
    computes the cost of using theta as the parameter for linear
    regression to fit the data points in X and y. 
    
    Returns:
        the cost in J and the gradient in grad
    """
    # Initialize some useful values
    m = X.shape[0] # number of training examples
    
    theta = theta.reshape(X.shape[1], 1)
    
    J = (np.sum((np.dot(X, theta)-y)**2) + Lambda*np.sum(theta[1:]**2))/m/2
    grad = np.dot(X.T, (X.dot(theta)-y))/m
    grad[1:] = grad[1:] + Lambda*theta[1:]/m
    grad = grad.flatten()
    
    return J, grad

def train_linear_reg(X, y, Lambda):
    """Train linear regression given a dataset (X, y) and aregularization
    parameter lambda
    
    train linear regression using the dataset (X, y) and regularization
    parameter lambda.
    
    Returns: 
        the trained parameters theta.
    """
    
    # Initialize Theta
    initial_theta = np.zeros((X.shape[1], 1)) 
    
    # Create "short hand" for the cost function to be minimized
    cost_function = lambda t: linear_reg_cost_function(X, y, t, Lambda)
    
    res = op.minimize(fun=cost_function, x0=initial_theta, \
        jac=True, method = 'TNC', options={'maxiter': 200})
    return res.x

def learning_curve(X, y, Xval, yval, Lambda):
    """Generate the train and cross validation set errors needed to plot a
    learning curve
    
    In this function, you will compute the train and test errors for
    dataset sizes from 1 up to m. In practice, when working with larger
    datasets, you might want to do this in larger intervals.
    
    returns:
        the train and cross validation set errors for a learning curve.
        In particular, it returns two vectors of the same length - 
        error_train and error_val. Then, error_train(i) contains the
        training error for i examples (and similarly for error_val(i)).
    """
    
    # Number of training examples
    m = X.shape[0]
    
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))
    
    for i in range(m):
        theta = train_linear_reg(X[0:i+1, :], y[0:i+1], Lambda)
        error_train[i], _ = linear_reg_cost_function(X[0:i+1, :], y[0:i+1], theta, 0)
        error_val[i], _ = linear_reg_cost_function(Xval, yval, theta, 0)
    
    return error_train, error_val

def poly_features(X, p):
    """Map X (1D vector) into the p-th power
    
    take a data matrix X (size m x 1) and maps each example into its
    polynomial features where X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
    """
    
    X_poly = np.zeros((X.shape[0], p))
    
    for i in range(p):
        X_poly[:,i] = (X**(i+1)).flatten()
    
    return X_poly

def feature_normalize(X):
    """Normalize the features in X
    
    Returns:
        a normalized version of X where the mean value of each feature
        is 0 and the standard deviation is 1. This is often a good
        preprocessing step to do when working with learning algorithms.
    """
    
    mu = np.mean(X, 0)
    X_norm = X - mu
    
    sigma = np.std(X, 0, ddof=1)
    X_norm = X_norm/sigma
    
    return X_norm, mu, sigma

def plot_fit(min_x, max_x, mu, sigma, theta, p):
    """Plot a learned polynomial regression fit over an existing figure.
    Also works with linear regression.
    
    plot the learned polynomial fit with power p and feature normalization (mu, sigma).
    """
    
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)
    
    # Map the X values 
    X_poly = poly_features(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma
    
    # Add ones
    X_poly = np.hstack((np.ones((x.shape[0], 1)), X_poly))
    
    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--', linewidth=2)
    
def validation_curve(X, y, Xval, yval):
    """ Generate the train and validation errors needed to
    plot a validation curve that we can use to select lambda
    
    Returns:
        the train and validation errors (in error_train, error_val)
        for different values of lambda. You are given the training set (X,
        y) and validation set (Xval, yval).
    """
    
    # Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    
    error_train = np.zeros((lambda_vec.size, 1))
    error_val = np.zeros((lambda_vec.size, 1))
    
    for i in range(lambda_vec.size):
        Lambda = lambda_vec[i]
        theta = train_linear_reg(X, y, Lambda)
        error_train[i], _ = linear_reg_cost_function(X, y, theta, 0)
        error_val[i], _ = linear_reg_cost_function(Xval, yval, theta, 0)
    
    return lambda_vec, error_train, error_val
    
if __name__ == "__main__":
    ## =========== Part 1: Loading and Visualizing Data =============
    
    # Load Training Data
    print('Loading and Visualizing Data ...')
    
    # Load from ex5data1: 
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = sio.loadmat('ex5data1.mat')
    X, y, Xval, yval, Xtest, ytest = data['X'], data['y'], \
        data['Xval'], data['yval'], data['Xtest'], data['ytest']
    
    # m = Number of examples
    m = X.shape[0]
    
    # Plot training data
    plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()
    
    ## =========== Part 2: Regularized Linear Regression Cost =============
    
    theta = np.array([[1], [1]])
    J, _ = linear_reg_cost_function(np.hstack((np.ones((m, 1)), X)), y, theta, 1)
    
    print('Cost at theta = [1 ; 1]: ', J, \
         '\n(this value should be about 303.993192)\n')

    ## =========== Part 3: Regularized Linear Regression Gradient =============
    
    theta = np.array([[1], [1]])
    J, grad = linear_reg_cost_function(np.hstack((np.ones((m, 1)), X)), y, theta, 1)
    
    print('Gradient at theta = [1 ; 1]:  [', grad[0], ';', grad[1], ']', \
            '\n(this value should be about [-15.303016; 598.250744])')

    ## =========== Part 4: Train Linear Regression =============
    
    #  Train linear regression with lambda = 0
    Lambda = 0
    theta = train_linear_reg(np.hstack((np.ones((m, 1)), X)), y, Lambda)
    
    #  Plot fit over the data
    plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.plot(X, np.dot(np.hstack((np.ones((m, 1)), X)), theta), '--', linewidth=2)
    plt.show()

    ## =========== Part 5: Learning Curve for Linear Regression =============
    
    Lambda = 0
    error_train, error_val = learning_curve(np.hstack((np.ones((m, 1)), X)), y, \
        np.hstack((np.ones((Xval.shape[0], 1)), Xval)), yval, Lambda)
    
    plt.plot(range(1, m+1), error_train, range(1, m+1), error_val)
    plt.title('Learning curve for linear regression')
    plt.legend(('Train', 'Cross Validation'))
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 150])
    plt.show()
    
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f\n' % (i+1, error_train[i], error_val[i]))
        
    ## =========== Part 6: Feature Mapping for Polynomial Regression =============
    
    p = 8
    
    # Map X onto Polynomial Features and Normalize
    X_poly = poly_features(X, p)
    X_poly, mu, sigma = feature_normalize(X_poly)  # Normalize
    X_poly = np.hstack((np.ones((m, 1)), X_poly))                   # Add Ones
    
    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = poly_features(Xtest, p)
    X_poly_test = X_poly_test - mu
    X_poly_test = X_poly_test / sigma
    X_poly_test = np.hstack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))         # Add Ones
    
    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = poly_features(Xval, p)
    X_poly_val = X_poly_val - mu
    X_poly_val = X_poly_val / sigma
    X_poly_val = np.hstack((np.ones((X_poly_val.shape[0], 1)), X_poly_val))           # Add Ones
    
    print('Normalized Training Example 1:')
    print(X_poly[0, :])
    
    ## =========== Part 7: Learning Curve for Polynomial Regression =============
    
    Lambda = 0
    theta = train_linear_reg(X_poly, y, Lambda)
    
    # Plot training data and fit
    plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
    plot_fit(np.min(X), np.max(X), mu, sigma, theta, p)
    plt.xlabel('Change in water level (x)');
    plt.ylabel('Water flowing out of the dam (y)');
    plt.title('Polynomial Regression Fit (lambda = %f)' % Lambda)
    
    plt.figure()
    error_train, error_val = learning_curve(X_poly, y, X_poly_val, yval, Lambda)
    plt.plot(range(1,m+1), error_train, range(1,m+1), error_val)
    
    plt.title('Polynomial Regression Learning Curve (lambda = %f)' % Lambda)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 100])
    plt.legend(['Train', 'Cross Validation'])
    plt.show()
    
    print('Polynomial Regression (lambda = %f)\n' % Lambda)
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f\n' % (i+1, error_train[i], error_val[i]))

    ## =========== Part 8: Validation for Selecting Lambda =============

    lambda_vec, error_train, error_val = validation_curve(X_poly, y, X_poly_val, yval)

    plt.plot(lambda_vec, error_train, lambda_vec, error_val)
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.show()
    
    print('lambda\t\tTrain Error\tValidation Error')
    for i in range(lambda_vec.size):
        print(' %f\t%f\t%f\n' % (lambda_vec[i], error_train[i], error_val[i]))