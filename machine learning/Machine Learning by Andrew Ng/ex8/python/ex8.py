#!/usr/bin/env python

"""
This is a python implementation of the programming assignment in machine learning by Andrew Ng.
The programming assignment implements the anomaly detection algorithm and apply it to 
detect failing servers on a network. 
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import io as sio

def estimate_gaussian(X):
    """Estimate the parameters of a Gaussian distribution using the data in X

    Inputs:
        The input X is the dataset with each n-dimensional data point in one row
        
    Returns:
        The output is an n-dimensional vector mu, the mean of the data set
        and the variances sigma^2, an n x 1 vector
    """
    
    mu = np.mean(X, 0)
    sigma2 = np.var(X, 0)
    
    return mu, sigma2

def multivariate_gaussian(X, mu, Sigma2):
    """Compute the probability density function of the multivariate gaussian distribution.
    
    Compute the probability density function of the examples X under the multivariate gaussian 
    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is treated as the
    covariance matrix. If Sigma2 is a vector, it is treated as the \sigma^2 values of the
    variances in each dimension (a diagonal covariance matrix)
    """

    k = mu.shape[0]
    
    if Sigma2.ndim == 1 or Sigma2.shape[0] == 1 or Sigma2.shape[1] == 1:
        Sigma2 = np.diag(Sigma2)
    
    X = X - mu
    p = (2 * np.pi) ** (- k / 2) * np.linalg.det(Sigma2) ** (-0.5)\
        * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(Sigma2)) * X, 1))
    
    return p
    
def visualize_fit(X, mu, sigma2):
    """Visualize the dataset and its estimated distribution.
    
    This visualization shows you the probability density function
    of the Gaussian distribution. Each example has a location (x1, x2)
    that depends on its feature values.
    """

    x = np.arange(0, 35.5, 0.5)
    X1, X2 = np.meshgrid(x, x)
    Z = multivariate_gaussian(np.hstack((X1.flatten().reshape(-1, 1), X2.flatten().reshape(-1, 1))),mu,sigma2)
    Z = Z.reshape(X1.shape)
    
    plt.plot(X[:, 0], X[:, 1],'bx')
    # Do not plot if there are infinities
    if (np.sum(np.isinf(Z)) == 0):
        plt.contour(X1, X2, Z, 10**np.arange(-20, 1, 3, dtype=float))

def select_threshold(yval, pval):
    """Find the best threshold (epsilon) to use for selecting outliers
    
    Find the best threshold to use for selecting outliers based on the
    results from a validation set (pval) and the ground truth (yval).
    """

    best_epsilon = 0
    best_F1 = 0
    F1 = 0

    stepsize = (np.max(pval) - np.min(pval)) / 1000
    for epsilon in np.arange(np.min(pval), np.max(pval), stepsize):
        cv_predictions = (pval < epsilon)
        tp = np.sum((cv_predictions == 1) * (yval.flatten() == 1))
        fp = np.sum((cv_predictions == 1) * (yval.flatten() == 0))
        fn = np.sum((cv_predictions == 0) * (yval.flatten() == 1))
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1 = 2*prec*rec/(prec+rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1
    
        
if __name__=="__main__":
    ## ================== Part 1: Load Example Dataset  ===================

    print('Visualizing example dataset for outlier detection.')

    #  The following command loads the dataset. You should now have the
    #  variables X, Xval, yval in your environment
    data1 = sio.loadmat('ex8data1.mat')
    X = data1['X']
    Xval = data1['Xval']
    yval = data1['yval']
    
    #  Visualize the example dataset
    plt.plot(X[:, 0], X[:, 1], 'bx')
    plt.axis([0, 30, 0, 30])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()
    
    ## ================== Part 2: Estimate the dataset statistics ===================

    print('Visualizing Gaussian fit.\n')
    
    #  Estimate my and sigma2
    mu, sigma2 = estimate_gaussian(X)
    
    #  Returns the density of the multivariate normal at each data point (row) 
    #  of X
    p = multivariate_gaussian(X, mu, sigma2)
    
    #  Visualize the fit
    visualize_fit(X,  mu, sigma2)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()
    
    ## ================== Part 3: Find Outliers ===================

    pval = multivariate_gaussian(Xval, mu, sigma2)
    
    epsilon, F1 = select_threshold(yval, pval)
    print('Best epsilon found using cross-validation:', epsilon)
    print('Best F1 on Cross Validation Set:', F1)
    print('   (you should see a value epsilon of about 8.99e-05)')
    print('   (you should see a Best F1 value of  0.875000)\n')
    
    #  Find the outliers in the training set and plot the
    outliers = p < epsilon
    
    #  Draw a red circle around those outliers
    plt.plot(X[outliers, 0], X[outliers, 1], 'ro', linewidth=2, markersize=10)
    visualize_fit(X,  mu, sigma2)
    plt.show()
    
    ## ================== Part 4: Multidimensional Outliers ===================

    #  Loads the second dataset. You should now have the
    #  variables X, Xval, yval in your environment
    data2 = sio.loadmat('ex8data2.mat')
    X = data2['X']
    Xval = data2['Xval']
    yval = data2['yval']
    
    #  Apply the same steps to the larger dataset
    mu, sigma2 = estimate_gaussian(X)
    
    #  Training set 
    p = multivariate_gaussian(X, mu, sigma2)
    
    #  Cross-validation set
    pval = multivariate_gaussian(Xval, mu, sigma2)
    
    #  Find the best threshold
    epsilon, F1 = select_threshold(yval, pval)
    
    print('Best epsilon found using cross-validation:', epsilon)
    print('Best F1 on Cross Validation Set:', F1)
    print('   (you should see a value epsilon of about 1.38e-18)')
    print('   (you should see a Best F1 value of 0.615385)')
    print('# Outliers found:', np.sum(p < epsilon))