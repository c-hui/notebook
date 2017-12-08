"""
This is a python implement of the programming assignment in machine learning by Andrew Ng.
The programming assignment implements the K-means clustering algorithm and applys it to
compress an image.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as sio
from scipy.misc import imread
import random

def find_closest_centroids(X, centroids):
    """compute the centroid memberships for every example
     
    return the closest centroids in idx for a dataset X where each
    row is a single example. idx = m x 1 vector of centroid assignments
    (i.e. each entry in range [0..K-1])
    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        x = X[i, :]
        L = np.sum((centroids-x)**2, 1)
        idx[i] = np.argmin(L)
    
    return idx

def compute_centroids(X, idx, K):
    """return the new centroids by computing the means of the data points
    assigned to each centroid.
  
    return the new centroids by computing the means of the data points
    assigned to each centroid. It is given a dataset X where each row is a
    single data point, a vector idx of centroid assignments (i.e. each
    entry in range [1..K]) for each example, and K, the number of centroids.
    You should return a matrix centroids, where each row of centroids is 
    the mean of the data points assigned to it.
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))
    
    for i in range(K):
        I = (idx==i)
        centroids[i, :] = np.sum(X[I, :], 0)/np.sum(I)

    return centroids

def run_kMeans(X, initial_centroids, max_iters):
    """run the K-Means algorithm on data matrix X, where each row of X
    is a single example
    
    runs the K-Means algorithm on data matrix X, where each row of X is
    a single example. It uses initial_centroids used as the initial
    centroids. max_iters specifies the total number of interactions 
    of K-Means to execute. runkMeans returns centroids, a Kxn matrix of
    the computed centroids and idx, a m x 1 vector of centroid 
    assignments (i.e. each entry in range [0..K-1])
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)
    
    # Run K-Means
    for i in range(max_iters):
        
        # Output progress
        print('K-Means iteration %d/%d...' % (i+1, max_iters))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    
    return centroids, idx

def kMeans_init_centroids(X, K):
    """initialize K centroids that are to be used in K-Means on the dataset X
    
    return K initial centroids to be used with the K-Means on the dataset X
    """

    # Randomly reorder the indices of examples
    randidx = random.sample(range(X.shape[0]), K)
    # Take the first K examples as centroids
    centroids = X[randidx, :]

    return centroids


if __name__=="__main__":
    ## ================= Part 1: Find Closest Centroids ====================

    print('Finding closest centroids.\n')
    
    # Load an example dataset that we will be using
    data2 = sio.loadmat('ex7data2.mat')
    X = data2['X']
    
    # Select an initial set of centroids
    K = 3 # 3 Centroids
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    
    # Find the closest centroids for the examples using the
    # initial_centroids
    idx = find_closest_centroids(X, initial_centroids)
    
    print('Closest centroids for the first 3 examples: ')
    print('', idx[0:3].flatten())
    print('\n(the closest centroids should be 0, 2, 1 respectively)')
    
    ## ===================== Part 2: Compute Means =========================
    
    print('\nComputing centroids means.\n')
    
    #  Compute means based on the closest centroids found in the previous part.
    centroids = compute_centroids(X, idx, K)
    
    print('Centroids computed after initial finding of closest centroids: ')
    print(centroids)
    print('\n(the centroids should be')
    print('   [ 2.428301 3.157924 ]')
    print('   [ 5.813503 2.633656 ]')
    print('   [ 7.119387 3.616684 ]\n')
    
    ## =================== Part 3: K-Means Clustering ======================

    print('\nRunning K-Means clustering on example dataset.\n')
    
    # Load an example dataset
    data2 = sio.loadmat('ex7data2.mat')
    X = data2['X']
    
    # Settings for running K-Means
    K = 3
    max_iters = 10
    
    # For consistency, here we set centroids to specific values
    # but in practice you want to generate them automatically, such as by
    # settings them to be random examples (as can be seen in
    # kMeansInitCentroids).
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    
    # Run K-Means algorithm.
    centroids, idx = run_kMeans(X, initial_centroids, max_iters)
    print('\nK-Means Done.\n')
    
    ## ============= Part 4: K-Means Clustering on Pixels ===============

    print('\nRunning K-Means clustering on pixels from an image.\n')
    
    #  Load an image of a bird
    A = imread('bird_small.png')
    
    A = A / 255 # Divide by 255 so that all values are in the range 0 - 1
    
    # Size of the image
    img_size = A.shape
    
    # Reshape the image into an Nx3 matrix where N = number of pixels.
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X that we will use K-Means on.
    X = A.reshape(img_size[0] * img_size[1], 3)
    
    # Run your K-Means algorithm on this data
    # You should try different values of K and max_iters here
    K = 16
    max_iters = 10
    
    # When using K-Means, it is important the initialize the centroids
    # randomly. 
    # You should complete the code in kMeansInitCentroids.m before proceeding
    initial_centroids = kMeans_init_centroids(X, K)
    
    # Run K-Means
    centroids, idx = run_kMeans(X, initial_centroids, max_iters)
    
    ## ================= Part 5: Image Compression ======================

    print('\nApplying K-Means to compress an image.\n')
    
    # Find closest cluster members
    idx = find_closest_centroids(X, centroids)
    
    # Essentially, now we have represented the image X as in terms of the
    # indices in idx. 
    #
    # We can now recover the image from the indices (idx) by mapping each pixel
    # (specified by its index in idx) to the centroid value
    X_recovered = centroids[idx.astype(int),:]
    
    # Reshape the recovered image into proper dimensions
    X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)
    
    # Display the original image 
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(A.reshape(img_size[1], img_size[0], 3, order='F')) 
    ax[0].set_title('Original')
    
    # Display compressed image side by side
    ax[1].imshow(X_recovered.reshape(img_size[1], img_size[0], 3, order='F'))
    ax[1].set_title('Compressed, with %d colors.' % K)
    plt.show()