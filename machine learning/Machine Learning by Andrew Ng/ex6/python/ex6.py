"""
This is a python implement of the programming assignment in machine learning by Andrew Ng.
The programming assignment uss support vector machines (SVMs) with various example 2D datasets.
"""

import numpy as np
from scipy import io as sio
from matplotlib import pyplot as plt
from sklearn import svm

def plot_data(X, y):
    """Plot the data points X and y into a new figure 

    plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.

    Note: This was slightly modified such that it expects y = 1 or y = 0
    """
    # Find Indices of Positive and Negative Examples
    pos = (y==1).T[0]
    neg = (y==0).T[0]

    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k+',linewidth=1, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)

def visualize_boundary_linear(X, y, model):
    """plot a linear decision boundary learned by the SVM

    plot a linear decision boundary learned by the SVM and
    overlays the data on it
    """
    w = model.coef_[0]
    b = model.intercept_
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    yp = - (w[0]*xp + b)/w[1]
    plot_data(X, y)
    plt.plot(xp, yp, '-b')
    plt.show()
    
def gaussian_kernel(x1, x2, sigma):
    """return a radial basis function kernel between x1 and x2
    
    Returns:
        a gaussian kernel between x1 and x2 and returns the
        value in sim
    """
    # Ensure that x1 and x2 are column vectors
    x1 = x1.ravel()
    x2 = x2.ravel()
    
    sim = np.exp(-np.sum((x1-x2)**2)/2/sigma**2)
    return sim
    
def visualize_boundary(X, y, model):
    """plot a non-linear decision boundary learned by the SVM
    
    plot a non-linear decision boundary learned by the SVM and
    overlays the data on it
    """
    # Plot the training data on top of the boundary
    plot_data(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(0, X1.shape[1]):
        this_X = np.hstack((X1[:,i], X2[:, i])).reshape(-1,2, order='F')
        vals[:, i] = model.predict(this_X)
    
    # Plot the SVM boundary
    plt.contour(X1, X2, vals, [0.5], colors='b')
    plt.show()
    
def dataset3_params(X, y, Xval, yval):
    """returns your choice of C and sigma for Part 3 of the exercise
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel
    
    returns:
        your choice of C and sigma. You should complete this function to return
        the optimal C and sigma based on a cross-validation set.
    """

    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3
    error = 1
    parameters = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for i in range(8):
        for j in range(8):
            clf = svm.SVC(C=parameters[i], gamma=1/parameters[j])
            clf.fit(X, y.ravel())
            predictions = clf.predict(Xval)
            error0 = np.mean(predictions != yval.ravel())
            if error0 < error:
                error = error0
                C = parameters[i]
                sigma = parameters[j]
    return C, sigma
    
if __name__=="__main__":
    ## =============== Part 1: Loading and Visualizing Data ================
    print('Loading and Visualizing Data ...\n')

    # Load from ex6data1: 
    # You will have X, y in your environment
    data1 = sio.loadmat('ex6data1.mat')
    X = data1['X']
    y = data1['y']
    
    # Plot training data
    plot_data(X, y)
    plt.show()
    
    ## ==================== Part 2: Training Linear SVM ====================

    # Load from ex6data1: 
    # You will have X, y in your environment
    data1 = sio.loadmat('ex6data1.mat')
    X = data1['X']
    y = data1['y']
    
    print('\nTraining Linear SVM ...')
    
    # You should try to change the C value below and see how the decision
    # boundary varies (e.g., try C = 1000)
    C = 1
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X, y.ravel())
    visualize_boundary_linear(X, y, clf)

    ## =============== Part 3: Implementing Gaussian Kernel ===============
    
    print('\nEvaluating the Gaussian Kernel ...')
    
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1]) 
    sigma = 2
    sim = gaussian_kernel(x1, x2, sigma)
    
    print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f : \
            \n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))
            
    ## =============== Part 4: Visualizing Dataset 2 ================
    
    print('Loading and Visualizing Data ...')
    
    # Load from ex6data2: 
    # You will have X, y in your environment
    data2 = sio.loadmat('ex6data2.mat')
    X = data2['X']
    y = data2['y']
    
    # Plot training data
    plot_data(X, y)
    plt.show()
    
    ## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
    
    print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...')
    
    # Load from ex6data2: 
    # You will have X, y in your environment
    data2 = sio.loadmat('ex6data2.mat')
    X = data2['X']
    y = data2['y']
    
    # SVM Parameters
    C = 1
    sigma = 0.01
    
    clf = svm.SVC(C=C, gamma=1/sigma)
    clf.fit(X, y.ravel())
    visualize_boundary(X, y, clf)
    
    ## =============== Part 6: Visualizing Dataset 3 ================
    print('Loading and Visualizing Data ...')
    
    # Load from ex6data3: 
    # You will have X, y in your environment
    data3 = sio.loadmat('ex6data3.mat')
    X = data3['X']
    y = data3['y']
    
    # Plot training data
    plot_data(X, y)
    plt.show()
    
    ## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

    # Load from ex6data3: 
    # You will have X, y in your environment
    data3 = sio.loadmat('ex6data3.mat')
    X = data3['X']
    y = data3['y']
    Xval = data3['Xval']
    yval = data3['yval']
    
    # Try different SVM Parameters here
    C, sigma = dataset3_params(X, y, Xval, yval)
    
    # Train the SVM
    clf = svm.SVC(C=C, gamma=1/sigma)
    clf.fit(X, y.ravel())
    visualize_boundary(X, y, clf)