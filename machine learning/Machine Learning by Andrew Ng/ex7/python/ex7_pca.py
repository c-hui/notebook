"""
This is a python implement of the programming assignment in machine learning by Andrew Ng.
The programming assignment uses principal component analysis to find a low-dimensional
representation of face images.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as sio
from scipy.misc import imread
from numpy.linalg import svd
import random
from mpl_toolkits.mplot3d import Axes3D
from ex7 import kMeans_init_centroids, run_kMeans

def feature_normalize(X):
    """Normalize the features in X 
    
    return a normalized version of X where the mean value of each
    feature is 0 and the standard deviation is 1. This is often a
    good preprocessing step to do when working with learning algorithms.
    """

    mu = np.mean(X, 0)   
    sigma = np.std(X, 0, ddof=1)
    X_norm = (X-mu)/sigma
    
    return X_norm, mu, sigma

def pca(X):
    """PCA Run principal component analysis on the dataset X
    
    computes eigenvectors of the covariance matrix of X
    
    Returns:
        the eigenvectors U, the eigenvalues (on diagonal) in S
    """
    
    # Useful values
    m, n = X.shape
    
    Sigma = np.dot(X.T, X)/m
    U, S, _ = svd(Sigma)
    return U, np.diag(S)

def draw_line(p1, p2, *varargin, **others):
    """Draw a line from point p1 to point p2
       
    Draw a line from point p1 to point p2 and holds the current figure
    """
    
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], *varargin, **others)

def project_data(X, U, K):
    """Compute the reduced data representation when projecting only on to
    the top k eigenvectors
    
    compute the projection of the normalized inputs X into the reduced
    dimensional space spanned by the first K columns of U. It returns
    the projected examples in Z.
    """
    Z = np.dot(X, U[:,0:K])
    
    return Z

def recover_data(Z, U, K):
    """Recover an approximation of the original data when using the 
    projected data
    
    recover an approximation the original data that has been reduced
    to K dimensions. It returns the approximate reconstruction in X_rec.
    """
    X_rec = np.dot(Z, U[:,0:K].T)

    return X_rec

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

    # Display Image
    plt.imshow(display_array.T, cmap='gray')
    # Do not show axis
    plt.axis("off")

def plot_data_points(X, idx, K):
    """plot data points in X, coloring them so that those with the same index 
    assignments in idx have the same color
    """
    
    # Create palette
    palette = plt.cm.get_cmap('hsv', K+1)
    colors = [palette(int(i)) for i in idx]
    
    # Plot the data
    plt.scatter(X[:,0], X[:,1], s=15, c=colors);
    
if __name__=="__main__":
    ## ================== Part 1: Load Example Dataset  ===================
    
    print('Visualizing example dataset for PCA.\n')
    
    #  The following command loads the dataset. You should now have the 
    #  variable X in your environment
    data1 = sio.loadmat('ex7data1.mat')
    X = data1['X']
    
    #  Visualize the example dataset
    plt.plot(X[:, 0], X[:, 1], 'bo')
    plt.axis([0.5, 6.5, 2, 8])
    plt.axis('square')
    plt.show()
    
    ## =============== Part 2: Principal Component Analysis ===============

    print('\nRunning PCA on example dataset.\n')
    
    #  Before running PCA, it is important to first normalize X
    X_norm, mu, sigma = feature_normalize(X)
    
    #  Run PCA
    U, S = pca(X_norm)
    
    #  Compute mu, the mean of the each feature
    #
    #  Draw the eigenvectors centered at mean of data. These lines show the
    #  directions of maximum variations in the dataset.
    
    plt.plot(X[:, 0], X[:, 1], 'bo')
    plt.axis([0.5, 6.5, 2, 8])
    plt.axis('square')
    draw_line(mu, mu + 1.5 * S[0,0] * U[:,0], '-k', linewidth = 2)
    draw_line(mu, mu + 1.5 * S[1,1] * U[:,1], '-k', linewidth = 2)
    plt.show()
    
    print('Top eigenvector: ')
    print(' U[:,0] =', U[0,0], U[1,0])
    print('\n(you should expect to see -0.707107 -0.707107)')
    
    ## =================== Part 3: Dimension Reduction ===================

    print('\nDimension reduction on example dataset.\n')

    #  Plot the normalized dataset (returned from pca)
    plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
    plt.axis([-4, 3, -4, 3]) 
    plt.axis('square')
    
    #  Project the data onto K = 1 dimension
    K = 1
    Z = project_data(X_norm, U, K)
    print('Projection of the first example:', Z[1])
    print('\n(this value should be about 1.481274)\n')
    
    X_rec  = recover_data(Z, U, K)
    print('Approximation of the first example:', X_rec[0, 0], X_rec[0, 1])
    print('\n(this value should be about  -1.047419 -1.047419)\n')
    
    #  Draw lines connecting the projected points to the original points
    plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
    for i in range(X_norm.shape[0]):
        draw_line(X_norm[i,:], X_rec[i,:], '--k', linewidth = 1)
    plt.show()
    
    ## =============== Part 4: Loading and Visualizing Face Data =============

    print('\nLoading face dataset.\n')

    #  Load Face dataset
    faces = sio.loadmat('ex7faces.mat')
    X = faces['X']
    
    #  Display the first 100 faces in the dataset
    display_data(X[0:100, :])
    plt.show()
    
    ## =========== Part 5: PCA on Face Data: Eigenfaces  ===================

    print('\nRunning PCA on face dataset.')
    print('(this might take a minute or two ...)\n')
    
    #  Before running PCA, it is important to first normalize X by subtracting 
    #  the mean value from each feature
    X_norm, mu, sigma = feature_normalize(X)
    
    #  Run PCA
    U, S = pca(X_norm)
    
    #  Visualize the top 36 eigenvectors found
    display_data(U[:, 0:36].T)
    plt.show()
    
    ## ============= Part 6: Dimension Reduction for Faces =================

    print('\nDimension reduction for face dataset.\n')
    
    K = 100
    Z = project_data(X_norm, U, K)
    
    print('The projected data Z has a size of: ')
    print(Z.shape)
    
    ## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====

    print('\nVisualizing the projected (reduced dimension) faces.\n')

    K = 100
    X_rec  = recover_data(Z, U, K)
    
    plt.figure(12)
    
    # Display normalized data
    plt.subplot(121)
    display_data(X_norm[0:100,:])
    plt.title('Original faces')
    
    # Display reconstructed data from only k eigenfaces
    plt.subplot(122)
    display_data(X_rec[0:100,:])
    plt.title('Recovered faces')
    #plt.axis('square')
    plt.show()
    
    ## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===

    # Reload the image from the previous exercise and run K-Means on it
    # For this to work, you need to complete the K-Means assignment first
    A = imread('bird_small.png')
    A = A / 255
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], 3)
    K = 16
    max_iters = 10
    initial_centroids = kMeans_init_centroids(X, K)
    centroids, idx = run_kMeans(X, initial_centroids, max_iters)
    
    #  Sample 1000 random indexes (since working with all the data is
    #  too expensive. If you have a fast computer, you may increase this.
    sel = random.sample(range(X.shape[0]), 1000)
    
    #  Setup Color Palette
    palette = plt.cm.get_cmap('hsv', K)
    colors = [palette(int(i)) for i in idx[sel]]
    
    #  Visualize the data and centroid memberships in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], s=10, c=colors)
    ax.set_title('Pixel dataset plotted in 3D. Color shows centroid memberships')
    plt.show()
    
    ## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
    # Use PCA to project this cloud to 2D for visualization

    # Subtract the mean to use PCA
    X_norm, mu, sigma = feature_normalize(X)
    
    # PCA and project the data to 2D
    U, S = pca(X_norm)
    Z = project_data(X_norm, U, 2);
    
    # Plot in 2D
    plt.figure()
    plot_data_points(Z[sel, :], idx[sel], K)
    plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    plt.show()
