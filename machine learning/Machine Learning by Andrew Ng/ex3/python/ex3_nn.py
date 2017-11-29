"""
This is a python implement of the programming assignment in machine learning by Andrew Ng.
The programming assignment is about neural networks to recognize hand-written digits.
"""

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import scipy.optimize as op
import random
from ex3 import display_data, sigmoid

def predict(Theta1, Theta2, X):
    """Predict the label of an input given a trained neural network

    outputs the predicted label of X given the trained weights of a 
    neural network (Theta1, Theta2)
    """
    # Useful values
    m = len(X)
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    z3 = np.hstack((np.ones((m, 1)), a2)).dot(Theta2.T)
    a3 = sigmoid(z3)
    p = np.argmax(a3, 1) + 1
    return p.reshape(-1, 1)

if __name__=="__main__":
    ## Setup the parameters you will use for this exercise
    input_layer_size  = 400;  # 20x20 Input Images of Digits
    hidden_layer_size = 25;   # 25 hidden units
    num_labels = 10;          # 10 labels, from 1 to 10   
                              # (note that we have mapped "0" to label 10)
    
    ## =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset. 
    #  You will be working with a dataset that contains handwritten digits.
    #
    
    # Load Training Data
    print('Loading and Visualizing Data ...')
    
    data1 = sio.loadmat('ex3data1.mat') 
    X = data1['X']
    y = data1['y']
    m = len(X)
    
    # Randomly select 100 data points to display
    rand_indices = random.sample(range(m), 100)
    sel = X[rand_indices, :]
    
    display_data(sel)
    
    ## ================ Part 2: Loading Pameters ================
    # In this part of the exercise, we load some pre-initialized 
    # neural network parameters.
    
    print('\nLoading Saved Neural Network Parameters ...')
    
    # Load the weights into variables Theta1 and Theta2
    data2 = sio.loadmat('ex3weights.mat')
    Theta1 = data2['Theta1']
    Theta2 = data2['Theta2']
    
    
    ## ================= Part 3: Implement Predict =================
    #  After training the neural network, we would like to use it to predict
    #  the labels. You will now implement the "predict" function to use the
    #  neural network to predict the labels of the training set. This lets
    #  you compute the training set accuracy.
    
    pred = predict(Theta1, Theta2, X)
    
    print('\nTraining Set Accuracy: ', np.mean((pred==y).astype(int)) * 100)
    
    print('Program paused. Press enter to continue.')
    
    #  To give you an idea of the network's output, you can also run
    #  through the examples one at the a time to see what it is predicting.
    
    #  Randomly permute examples
    rp = list(range(m))
    random.shuffle(rp)
    
    for i in range(m):
        # Display 
        print('\nDisplaying Example Image')
        display_data(X[rp[i], :].reshape(1, -1))
    
        pred = predict(Theta1, Theta2, X[rp[i],:].reshape(1, -1))
        print('\nNeural Network Prediction: %d (digit %d)' % (pred, pred % 10))
        
        # Pause with quit option
        s = input('Paused - press enter to continue, q to exit:')
        if s == 'q':
            break