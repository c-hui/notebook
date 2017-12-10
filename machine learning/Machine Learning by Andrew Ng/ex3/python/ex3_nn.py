"""
This is a python implementation of the programming assignment in machine learning by Andrew Ng.
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

    output the predicted label of X given the trained weights of a 
    neural network (Theta1, Theta2)
    """
    # Useful values
    m = X.shape[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    p = np.argmax(a3, 1) + 1
    return p.reshape(-1, 1)

if __name__=="__main__":
    ## Setup the parameters you will use for this exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10   
                             # (note that we have mapped "0" to label 10)
    
    ## =========== Part 1: Loading and Visualizing Data =============
    
    # Load Training Data
    print('Loading and Visualizing Data ...')
    
    data1 = sio.loadmat('ex3data1.mat') 
    X = data1['X']
    y = data1['y']
    m = X.shape[0]
    
    # Randomly select 100 data points to display
    rand_indices = random.sample(range(m), 100)
    sel = X[rand_indices, :]
    
    display_data(sel)
    
    ## ================ Part 2: Loading Pameters ================
    
    print('\nLoading Saved Neural Network Parameters ...')
    
    # Load the weights into variables Theta1 and Theta2
    weights = sio.loadmat('ex3weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    
    
    ## ================= Part 3: Implement Predict =================
    pred = predict(Theta1, Theta2, X)
    
    print('\nTraining Set Accuracy: ', np.mean((pred==y).astype(int)) * 100)
    
    #  To give you an idea of the network's output, you can also run
    #  through the examples one at the a time to see what it is predicting.
    
    #  Randomly permute examples
    rp = list(range(m))
    random.shuffle(rp)
    
    for i in range(m):
    
        pred = predict(Theta1, Theta2, X[rp[i],:].reshape(1, -1))
        print('\nNeural Network Prediction: %d (digit %d)' % (pred, pred % 10))
        
        # Display 
        print('\nDisplaying Example Image')
        display_data(X[rp[i], :].reshape(1, -1))
        
        # Pause with quit option
        s = input('Paused - press enter to continue, q to exit:')
        if s == 'q':
            break