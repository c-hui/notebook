"""
This is a python implementation of the programming assignment in machine learning by Andrew Ng.
The programming assignment uses SVMs to build your own spam filter.
"""

import numpy as np
from scipy import io as sio
from matplotlib import pyplot as plt
from sklearn import svm
import re
import nltk

def get_vocab_list():
    """read the fixed vocabulary list in vocab.txt and returns a
    dict of the words
    
    read the fixed vocabulary list in vocab.txt and returns a dict
    of the words.
    """
    vocab_list = np.loadtxt("vocab.txt", dtype=str)
    return dict((v, int(k)) for k,v in vocab_list)

def process_email(email_contents):
    """ preprocesse a the body of an email andreturns a list
    of word_indices 

    preprocesse the body of an email and returns a list of
    indices of the words contained in the email. 
    """

    # Load Vocabulary
    vocab_list = get_vocab_list()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    #% Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = email_contents.index('\n\n')
    # email_contents = email_contents[hdrstart+2:]

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)


    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n==== Processed Email ====\n')

    # Process file
    l = 0

    tokenizer = nltk.RegexpTokenizer(r'[a-zA-Z]+')
    stemmer = nltk.PorterStemmer()
    
    for word in tokenizer.tokenize(email_contents):
        str = stemmer.stem(word)
        
        # Look up the word in the dictionary and add to word_indices if
        # found
        if str in vocab_list:
            word_indices.append(vocab_list[str])
        
        # Print to screen, ensuring that the output lines are not too long
        if l + len(str) + 1 > 78:
            print()
            l = 0
        print(str, end=' ')
        l = l + len(str) + 1
    
    # Print footer
    print('\n\n=========================')
    
    return word_indices

def email_features(word_indices):
    """take in a word_indices vector and produces a feature vector
    from the word indices
    
    take in a word_indices vector and produces a feature vector
    from the word indices. 
    """
    # Total number of words in the dictionary
    n = 1899
    
    x = np.zeros((n, 1))
    x[np.array(word_indices)-1] = 1
    return x
    
    
if __name__=="__main__":
    ## ==================== Part 1: Email Preprocessing ====================
    
    print('\nPreprocessing sample email (emailSample1.txt)')
    
    # Extract Features
    with open('emailSample1.txt', 'r') as infile:
        file_contents = infile.read()
    word_indices  = process_email(file_contents)
    
    # Print Stats
    print('Word Indices: ')
    print('', word_indices)
    print('\n')
    
    ## ==================== Part 2: Feature Extraction ====================
    
    print('\nExtracting features from sample email (emailSample1.txt)')
    
    # Extract Features
    with open('emailSample1.txt', 'r') as infile:
        file_contents = infile.read()
    word_indices = process_email(file_contents)
    features = email_features(word_indices)
    
    # Print Stats
    print('Length of feature vector:', len(features))
    print('Number of non-zero entries:', np.sum(features > 0))
    
    ## =========== Part 3: Train Linear SVM for Spam Classification ========

    # Load the Spam Email dataset
    # You will have X, y in your environment
    train = sio.loadmat('spamTrain.mat')
    X = train['X']
    y = train['y']
    
    print('\nTraining Linear SVM (Spam Classification)')
    print('(this may take 1 to 2 minutes) ...')
    
    C = 0.1;
    clf = svm.SVC(C, 'linear')
    clf.fit(X, y.ravel())
    
    p = clf.predict(X)
    
    print('Training Accuracy:', np.mean(p == y.ravel()) * 100)
    
    ## =================== Part 4: Test Spam Classification ================

    # Load the test dataset
    # You will have Xtest, ytest in your environment
    test = sio.loadmat('spamTest.mat')
    Xtest = test['Xtest']
    ytest = test['ytest']
    
    print('\nEvaluating the trained Linear SVM on a test set ...')
    
    p = clf.predict(Xtest)
    
    print('Test Accuracy:', np.mean(p == ytest.ravel()) * 100)
    
    ## ================= Part 5: Top Predictors of Spam ====================

    # Sort the weights and obtin the vocabulary list
    idx = np.argsort(clf.coef_[0])
    vocab_list = dict((v, k) for k,v in get_vocab_list().items())
    
    print('\nTop predictors of spam: ')
    for i in range(1, 16):
        print(' %-15s (%f) ' % (vocab_list[idx[-i]+1], clf.coef_[0][idx[-i]]))
    
    print('\n')
    
    ## =================== Part 6: Try Your Own Emails =====================

    # Set the file to be read in (change this to spamSample2.txt,
    # emailSample1.txt or emailSample2.txt to see different predictions on
    # different emails types). Try your own emails as well!
    filename = 'spamSample1.txt'
    
    # Read and predict
    with open(filename, 'r') as infile:
        file_contents = infile.read()
    word_indices  = process_email(file_contents)
    x = email_features(word_indices)
    p = clf.predict(x.reshape(1,-1))
    
    print('\nProcessed %s\n\nSpam Classification: %d' % (filename, p))
    print('(1 indicates spam, 0 indicates not spam)\n')
    