#!/usr/bin/env python

"""
This is a python implementation of the programming assignment in machine learning by Andrew Ng.
The programming assignment uses collaborative filtering to build a recommender system for movies.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import io as sio
import scipy.optimize as op

def cofi_cost_func(params, Y, R, num_users, num_movies, num_features, Lambda):
    """Collaborative filtering cost function.
    
    Returns:
        The cost and gradient for the collaborative filtering problem.
    """

    # Unfold the U and W matrices from params
    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)
    
    J = np.sum(((np.dot(X, Theta.T) - Y)*R)**2)/2 + Lambda * np.sum(X**2)/2 + Lambda * np.sum(Theta**2)/2
    
    X_grad = np.dot((np.dot(X, Theta.T) - Y) * R, Theta) + Lambda * X
    
    Theta_grad = np.dot(((np.dot(X, Theta.T) - Y) * R).T, X) + Lambda * Theta
    
    grad = np.hstack((X_grad.flatten(), Theta_grad.flatten()))
    
    return J, grad

def compute_numerical_gradient(J, theta):
    """Compute the gradient using "finite differences" and gives us a
    numerical estimate of the gradient.

    Compute the numerical gradient of the function J around theta.
    Calling y = J(theta) should return the function value at theta.
    
    Notes: The following code implements numerical gradient checking, and 
           returns the numerical gradient.It sets numgrad(i) to (a numerical 
           approximation of) the partial derivative of J with respect to the 
           i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
           be the (approximately) the partial derivative of J with respect 
           to theta(i).)
    """
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb.ravel()[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad.ravel()[p] = (loss2 - loss1) / (2*e)
        perturb.ravel()[p] = 0
    return numgrad
    
def check_cost_function(Lambda=0):
    """Create a collaborative filering problem to check your cost 
    function and gradients.

    Create a collaborative filering problem to check your cost 
    function and gradients, it will output the analytical gradients
    produced by your code and the numerical gradients 
    (computed using compute_numerical_gradient). These two gradient 
    computations should result in very similar values.
    """
    
    ## Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)
    
    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(*Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1
    
    ## Run Gradient Checking
    X = np.random.randn(*X_t.shape)
    Theta = np.random.randn(*Theta_t.shape)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]
    
    numgrad = compute_numerical_gradient(lambda t: cofi_cost_func(t, Y, R, num_users,\
        num_movies, num_features, Lambda), np.hstack((X.flatten(), Theta.flatten())))
    
    cost, grad = cofi_cost_func(np.hstack((X.flatten(), Theta.flatten())),  Y, R, num_users,\
        num_movies, num_features, Lambda)
    
    for i in range(grad.shape[0]):
        print(numgrad[i], grad[i])
    print('The above two columns you get should be very similar.\n' \
            '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')
    
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your cost function implementation is correct, then \n \
            the relative difference will be small (less than 1e-9). \n\
            \nRelative Difference: ', diff)
    
def load_movie_list():
    """Read the fixed movie list in movie.txt and returns a list of the words
    """
    
    movie_list = []
    with open("movie_ids.txt", 'r', errors='ignore') as in_file:
        for line in in_file:
            movie_list.append(line[line.find(' ')+1 :].strip())
    
    return movie_list

def normalize_ratings(Y, R):
    """Preproces data by subtracting mean rating for every movie (every row)
    
    Normalize Y so that each movie has a rating of 0 on average, and returns
    the mean rating in Ymean.
    """

    [m, n] = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = (R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean
    
    
if __name__=="__main__":
    ## =============== Part 1: Loading movie ratings dataset ================
  
    print('Loading movie ratings dataset.\n')
    
    #  Load data
    movies = sio.loadmat('ex8_movies.mat')
    Y = movies['Y']
    R = movies['R']
    #  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
    #  943 users
    #
    #  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
    #  rating to movie i
    #
    #  From the matrix, we can compute statistics like average rating.
    print('Average rating for movie 1 (Toy Story): %f / 5\n' \
           % np.mean(Y[0, R[0, :]]))
    
    #  We can "visualize" the ratings matrix by plotting it with imagesc
    plt.imshow(Y)
    plt.ylabel('Movies')
    plt.xlabel('Users')
    plt.show()
    
    ## ============ Part 2: Collaborative Filtering Cost Function ===========

    #  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
    movie_params = sio.loadmat('ex8_movieParams.mat')
    X = movie_params['X']
    Theta = movie_params['Theta']
    num_users = movie_params['num_users']
    num_movies = movie_params['num_movies']
    num_features = movie_params['num_features']
    
    #  Reduce the data set size so that this runs faster
    num_users = 4
    num_movies = 5
    num_features = 3
    X = X[0:num_movies, 0:num_features]
    Theta = Theta[0:num_users, 0:num_features]
    Y = Y[0:num_movies, 0:num_users]
    R = R[0:num_movies, 0:num_users]
    
    #  Evaluate cost function
    J, _ = cofi_cost_func(np.hstack((X.flatten(), Theta.flatten())), Y, R, num_users, num_movies, num_features, 0)
            
    print('Cost at loaded parameters:', J)
    print('(this value should be about 22.22)')
    
    ## ============== Part 3: Collaborative Filtering Gradient ==============
 
    print('\nChecking Gradients (without regularization) ... ')
    
    #  Check gradients by running checkNNGradients
    check_cost_function()
    
    ## ========= Part 4: Collaborative Filtering Cost Regularization ========  
    
    #  Evaluate cost function
    J, _ = cofi_cost_func(np.hstack((X.flatten(), Theta.flatten())), Y, R, num_users,\
        num_movies, num_features, 1.5)
            
    print('Cost at loaded parameters (lambda = 1.5):', J)
    print('(this value should be about 31.34)\n')
    
    ## ======= Part 5: Collaborative Filtering Gradient Regularization ======
  
    print('\nChecking Gradients (with regularization) ... ')
    
    #  Check gradients by running checkNNGradients
    check_cost_function(1.5)
    
    # ============== Part 6: Entering ratings for a new user ===============

    movie_list = load_movie_list()
    
    #  Initialize my ratings
    my_ratings = np.zeros((1682, 1))
    
    # Check the file movie_idx.txt for id of each movie in our dataset
    # For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
    my_ratings[0] = 4
    
    # Or suppose did not enjoy Silence of the Lambs (1991), you can set
    my_ratings[97] = 2
    
    # We have selected a few movies we liked / did not like and the ratings we
    # gave are as follows:
    my_ratings[6] = 3
    my_ratings[11]= 5
    my_ratings[53] = 4
    my_ratings[63]= 5
    my_ratings[65]= 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354]= 5
    
    print('\n\nNew user ratings:')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0 :
            print('Rated %d for %s' % (my_ratings[i], movie_list[i]))
            
    ## ================== Part 7: Learning Movie Ratings ====================

    print('\nTraining collaborative filtering...')
    
    #  Load data
    movies = sio.loadmat('ex8_movies.mat')
    Y = movies['Y']
    R = movies['R']
    
    #  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
    #  943 users
    #
    #  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
    #  rating to movie i
    
    #  Add our own ratings to the data matrix
    Y = np.hstack((my_ratings, Y))
    R = np.hstack(((my_ratings != 0).astype(int), R))
    
    #  Normalize Ratings
    Ynorm, Ymean = normalize_ratings(Y, R)
    
    #  Useful Values
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10
    
    # Set Initial Parameters (Theta, X)
    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_users, num_features)
    
    initial_parameters = np.hstack((X.flatten(), Theta.flatten()))
    
    # Set Regularization
    Lambda = 10
    res = op.minimize(fun=cofi_cost_func, x0=initial_parameters, \
        args=(Ynorm, R, num_users, num_movies,num_features, Lambda),\
        jac=True, method = 'TNC', options={'maxiter': 100})
    theta = res.x
    
    # Unfold the returned theta back into U and W
    X = theta[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = theta[num_movies*num_features:].reshape(num_users, num_features)
    
    print('Recommender system learning completed.')
    
    ## ================== Part 8: Recommendation for you ====================

    p = np.dot(X, Theta.T)
    my_predictions = p[:,0] + Ymean.flatten()
    
    movie_list = load_movie_list()
    
    ix = np.argsort(my_predictions)
    print('\nTop recommendations for you:')
    for i in range(1,11):
        j = ix[-i]
        print('Predicting rating %.1f for movie %s' % (my_predictions[j], movie_list[j]))
    
    print('\n\nOriginal ratings provided:')
    for i in range(len(my_predictions)):
        if my_ratings[i] > 0: 
            print('Rated %d for %s' % (my_ratings[i], movie_list[i]))