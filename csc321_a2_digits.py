from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
#from scipy.misc import imread
#from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
import random
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat

def plot10Img(data, name, i):
    """ Plots images from data, from index i and onwards. """
    f, axarr = plt.subplots(1, 10)
    plt.subplots_adjust(hspace=0.01) 
    plt.gray()
    j, name = 0, str(name)
    while i < i + 10 and j < 10:
        print data.shape
        axarr[j].axis('off')
        axarr[j].imshow(data[i].reshape((28,28)), cmap = cm.coolwarm)        
        i += 1
        j += 1        
    plt.title(name)
    savefig('10Imgs_'+ name)
    
def plot2Img(data, name, i):
    """ Plots images from data, from index i and onwards. """
    f, axarr = plt.subplots(1, 2)
    plt.subplots_adjust(hspace=0.01) 
    plt.gray()
    j, name = 0, str(name)
    while i < i + 2 and j < 2:
        axarr[j].axis('off')
        axarr[j].imshow(data[i].reshape((28,28)), cmap = cm.coolwarm)        
        i += 1
        j += 1        
    plt.title(name)
    savefig('2Imgs_'+ name)

def plotTrain(M,name):
    for i in range(10):
        plot10Img(M[name+str(i)], 'train'+str(i), 10)

def softmax(output):
    '''Return the output of the softmax function for the matrix of output. output
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''    
    return exp(output)/tile(sum(exp(output),0), (len(output),1))

def part2_output_layer(x,W,b):
    '''
    x is an Nx784 matrix where N is the number of inputs.    
    W is an 784x10 matrix where N is the number of inputs.    
    b is a 10x1 vector where N is the number of outputs.    
    Returns an Nx10 matrix 
    '''
    return dot(x,W)+b.T



def part3_gradient(inputs_train, prediction, targets_train, W, b):
    '''
    Inputs' shape: (784, 50) (10, 50) (10, 50) (784, 10) (10, 1)
    Output shape: (784, 10) (10, 1)
    '''
    # Compute derivation 
    dEbydlogit = prediction - targets_train
    
    # Backprop
    dEbydh_output = np.dot(W, dEbydlogit)
    dEbydh_input = dEbydh_output * inputs_train * (1 - inputs_train)

    # Gradients for weights and biases.
    dEbydW = np.dot(inputs_train, dEbydlogit.T)
    dEbydb = np.sum(dEbydlogit, axis=1).reshape(-1, 1)    
   
    return dEbydW, dEbydb
    
    
def forwardProp(inputs, W, b):  
    """inputs: matrix of shape (784, N)
    Output: Prediction.shape (10, N)"""     
    logit = np.dot(W.T, inputs) + b
    prediction = softmax(logit)  # Output prediction. 
    return prediction 


##--------num_inputs=N----num_hiddens=300------num_outputs=10----------
def InitNN(num_inputs, num_hiddens, num_outputs):
    """Initializes NN parameters."""
    W1 = 0.01 * np.random.randn(num_inputs, num_hiddens)
    W2 = 0.01 * np.random.randn(num_hiddens, num_outputs)
    b1 = np.zeros((num_hiddens, 1))
    b2 = np.zeros((num_outputs, 1))
    return W1, W2, b1, b2

def TrainNN_softmax(M, eps, momentum, num_epochs):    
    inputs_train, targets_train, inputs_test, targets_test = generate_set(M, 100, 50)
    
    inputs_train = inputs_train.T
    inputs_test = inputs_test.T
    targets_train = targets_train.T
    targets_test = targets_test.T
    W, W1, b, b1 = InitNN(inputs_train.shape[0], targets_train.shape[0], 0)
    
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    num_train_cases = inputs_train.shape[1]
    train_rates_collection = []
    train_costs_collection = []
    test_rates_collection = []
    test_costs_collection = []
    
    for epoch in range(num_epochs):
        prediction = forwardProp(inputs_train, W, b)    
        dEbydW, dEbydb = part3_gradient(inputs_train,prediction,targets_train,W,b)    
        #%%%% Update the weights at the end of the epoch %%%%%%
        dW = momentum * dW - (eps / num_train_cases) * dEbydW
        db = momentum * db - (eps / num_train_cases) * dEbydb
    
        W = W + dW
        b = b + db
        
        train_mce= (np.argmax(targets_train.T,axis=1)==np.argmax(prediction.T,axis=1)).mean() 
        train_cost = part2_cost_function(targets_train, inputs_train, W, b)
        train_rates_collection.append(train_mce)
        train_costs_collection.append(train_cost)
        #print "1-mce=", train_mce, " at epoch:", epoch, "train_cost=", train_cost
        
        test_prediction = forwardProp(inputs_test, W, b) 
        test_mce= (np.argmax(targets_test.T,axis=1)==np.argmax(test_prediction.T,axis=1)).mean() 
        test_cost = part2_cost_function(targets_test, inputs_test, W, b)
        test_rates_collection.append(test_mce)
        test_costs_collection.append(test_cost)
    #print "1-mce=", train_mce, " at epoch:", epoch, "train_cost=", train_cost         
        
    return W, b, inputs_train, prediction, targets_train, train_rates_collection, train_costs_collection, inputs_test, test_prediction, targets_test, test_rates_collection, test_costs_collection
    

def generate_set(data_set, train_size, test_size):
    '''
    helper function used to generate the training input, training target, test
    input, test target used the given size.
    
    returns:
    train_inputs: train_size*10 x 784 matrix
    train_targets: train_size*10 x 10 matrix 
    test_inputs: test_size*10 x 784 matrix
    test_targets: test_size*10 x 10 matrix
    '''
    #here the train_size is 50/10=5.

    test_targets=np.zeros((test_size*10, 10))
    train_targets=np.zeros((train_size*10, 10))
    for i in range(10):
        key_train='train'+str(i)
        key_test='test'+str(i)
        #random.sample(population, k)
        available_train=[n for n in range(data_set[key_train].shape[0])]
        available_test=[m for m in range(data_set[key_test].shape[0])]
        want_train=random.sample(available_train,train_size)
        want_test=random.sample(available_test,test_size)
        new_train_inputs=data_set[key_train][want_train,]/255.0
        new_test_inputs=data_set[key_test][want_test,]/255.0
        if i==0:
            train_inputs=new_train_inputs
            test_inputs=new_test_inputs
        else:
            train_inputs=np.concatenate((train_inputs,new_train_inputs), axis=0)
            test_inputs=np.concatenate((test_inputs,new_test_inputs), axis=0)            
        train_targets[i*train_size:(i+1)*train_size,i]=1
        test_targets[i*test_size:(i+1)*test_size,i]=1
    return train_inputs, train_targets, test_inputs, test_targets    

def part2_cost_function(target, x, w, b):    
    prediction = forwardProp(x, w,b)
    return -np.sum(target*np.log(prediction))

def part4_check_gradient(prediction_f,gradient_f,cost_f,target, x,w,b,delta,i,j):
    dw,db=np.zeros(w.shape),np.zeros(b.shape)
    dw[j,i], db[j,0] = delta, delta
    prediction = prediction_f(x.T, w.T,b)
    suppose_wij = ((cost_f(target.T, x.T, w.T+dw.T, b)-cost_f(target.T, x.T, w.T-dw.T,b))/2*float(delta))*1e+12
    suppose_bj = ((cost_f(target.T, x.T, w.T, b+db)-cost_f(target.T, x.T, w.T,b-db))/2*float(delta))*1e+12
    want_w, want_b=gradient_f(x.T, prediction, target.T, w.T, b)
    want_wij, want_bj=want_w[i,j], want_b[j,0]
    print "i, j =", i, j
    print "correce wij", suppose_wij, "computed wij", want_wij, "corrected bj", suppose_bj, "computed bj", want_bj
    print 'diff wrt wij:', suppose_wij-want_wij,'diff wrt bj:',suppose_bj-want_bj
    

def part7_forwardProp(X, W1, b1, W2, b2):
    """inputs: matrix of shape (784, N)
    w1: matrix of shape (784, 300)    
    b1: matrix of shape (300,1)    
    w2: matrix of shape (300, 10)    
    b2: matrix of shape (10,1)    
    """     
    Z1 = dot(W1.T, X) + b1
    H = tanh(Z1)
    Z2 = dot(W2.T, H) + b2
    O = softmax(Z2)
    return O

def part7_gradient(X, O, T, W1, b1, W2, b2):
    """
    X: train inputs (784,500)
    O: prediction (10, 50)
    T: targets (10, 50)
    w1: (784, 300)
    b1: (300, 1)
    w2: (300, 10)
    b2: (10, 1) 
    """    
    # Forward prop
    Z1 = dot(W1.T, X) + b1
    H = tanh(Z1)
    Z2 = dot(W2.T, H) + b2
    O = softmax(Z2)
    
    #dEbydZ2 = T * (1 - O)
    dEbydZ2 = T - O
    
    dHbydZ1 = 1 - H ** 2 
    dEbydH = dot(W2, dEbydZ2)
    dEbydZ1 = dEbydH * dHbydZ1
    
    # Gradients for weights and biases.
    dEbydW2 = dot(H, dEbydZ2.T)
    dEbydb2 = dot(ones((1,X.shape[1])), dEbydZ2.T)    
    dEbydW1 = dot(X, dEbydZ1.T)
    dEbydb1 = dot(ones((1,X.shape[1])), dEbydZ1.T)
    
    return - dEbydW1, - dEbydb1.T, - dEbydW2, - dEbydb2.T
    

def TrainNN_tanh(inputs_train, target_train, inputs_valid, target_valid, num_hiddens, eps, momentum, num_epochs):
    """Trains a single hidden layer NN.
  
    Inputs:
      num_hiddens: NUmber of hidden units.
      eps: Learning rate.
      momentum: Momentum.
      num_epochs: Number of epochs to run training for.
  
    Returns:
      W1: First layer weights.
      W2: Second layer weights.
      b1: Hidden layer bias.
      b2: Output layer bias.
      train_error: Training error at at epoch.
      valid_error: Validation error at at epoch.
    """
    inputs_train = inputs_train.T
    inputs_valid = inputs_valid.T
    target_train = target_train.T
    target_valid = target_valid.T
    train_rates_collection=[]
    train_costs_collection=[]
    valid_rates_collection=[]
    valid_costs_collection=[]    
    
    W1, W2, b1, b2 = InitNN(inputs_train.shape[0], num_hiddens, target_train.shape[0])
    dW1 = np.zeros(W1.shape)
    dW2 = np.zeros(W2.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    num_train_cases = inputs_train.shape[1]
    for epoch in range(num_epochs): 
       
        ## Forward prop
        prediction = part7_forwardProp(inputs_train, W1, b1, W2, b2) 
        
        ## Gradients for weights and biases.
        dEbydW1, dEbydb1, dEbydW2, dEbydb2 = part7_gradient(inputs_train, prediction, target_train, W1, b1, W2, b2)
    
        ##%%%% Update the weights at the end of the epoch %%%%%%
        dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1
        dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2
        db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1
        db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2
    
        W1 = W1 + dW1
        W2 = W2 + dW2
        b1 = b1 + db1
        b2 = b2 + db2
        
        ##%%%% Computes statistics for training and validation set %%%%
        train_predicted = part7_forwardProp(inputs_train, W1, b1, W2, b2)
        train_rate= (np.argmax(target_train.T,axis=1)==np.argmax(train_predicted.T,axis=1)).mean() 
        train_rates_collection.append(train_rate)
        train_cost = part7_cost(target_train, inputs_train, W1, W2, b1, b2)
        train_costs_collection.append(train_cost)        
        print "%Train%: 1-mce=", train_rate, " at epoch:", epoch, " train_cost:", train_cost
        
        valid_predicted = part7_forwardProp(inputs_valid, W1, b1, W2, b2)        
        valid_rate= (np.argmax(target_valid.T,axis=1)==np.argmax(valid_predicted.T,axis=1)).mean()
        valid_rates_collection.append(valid_rate)
        valid_cost = part7_cost(target_valid, inputs_valid, W1, W2, b1, b2)
        valid_costs_collection.append(valid_cost)         
        #print "%Test%: 1-mce=", valid_rate, " at epoch:", epoch, " train_cost:", valid_cost
            
    return W1, b1, W2, b2, inputs_train, target_train, train_predicted, inputs_valid, target_valid, valid_predicted, train_rates_collection, train_costs_collection, valid_rates_collection, valid_costs_collection

def part7_cost(target, inputs, w1, w2, b1, b2):
    prediction = part7_forwardProp(inputs, w1, b1, w2, b2)
    return -np.sum(target * np.log(prediction))  
   
def increment(w, h, j, k=-1):
    a = w
    if k == -1: a[j] += h
    else: a[j, k] += h
    return a

def increment(w, j, k, h):
    dw = zeros(w.shape)
    if k == -1: #w is b1 or b2
        dw[j] = h
    else:
        dw[j, k] = h
    return w + dw

def part7_check_grad(t, x, w1, b1, w2, b2, i, j, k, h=5e-8):
    if i == 1: 
        c = increment(w1, j, k, h)
        result = (part7_cost(t, x, c, w2, b1, b2) - part7_cost(t, x, w1, w2, b1, b2)) /h
    elif i == 2: 
        c = increment(w2, j, k, h)
        result = (part7_cost(t, x, w1, c, b1, b2) - part7_cost(t, x, w1, w2, b1, b2)) /h
    elif i == 3: 
        c = increment(b1, j, -1, h)
        result = (part7_cost(t, x, w1, w2, c, b2) - part7_cost(t, x, w1, w2, b1, b2)) /h
    else: 
        c = increment(b2, j, -1, h)
        result = (part7_cost(t, x, w1, w2, b1, c) - part7_cost(t, x, w1, w2, b1, b2)) /h    
    return result
    
##---------------answer questions------------------------
def p1(M,name):
    plotTrain(M,name)
    
def p3_and_4(M, i_lst=[207, 321], j_lst=[3,6]):
    inputs_train, targets_train, inputs_test, targets_test = generate_set(M, 5, 5)
    #Plot 10 images of each digits
    #plotTrain('train')
    w = 0.01 * np.random.randn(10,inputs_train.shape[1])
    b = 0.01 * np.random.randn(10,1)
    TrainNN_softmax(M, 0.01, 1, 75)
    for i in i_lst:
        for j in j_lst:
            part4_check_gradient(forwardProp,part3_gradient,part2_cost_function,targets_train, inputs_train, w,b,0.000001,i,j)  
            
def p5(M):
    ## Part 5
    num_iter=200
    xs=[i for i in range(num_iter)]
    W, b, inputs_train, prediction, targets_train, train_rates_collection, train_costs_collection, inputs_test, test_prediction, targets_test, test_rates_collection, test_costs_collection=TrainNN_softmax(M, 0.01, 1, num_iter)
   
    plt.clf()
    plt.plot(xs, test_rates_collection, 'g', label="Test Classification Rates")
    plt.plot(xs, train_rates_collection, 'r', label='Train Classification Rates')
    plt.ylabel('Classification Rates')
    plt.xlabel("Number of Iterations")
    plt.title("Classification Rate Vs Iterations Plot")    
    plt.legend(loc = 'best')
    plt.savefig('part5(classificationrate).jpg')
    plt.clf()        
    plt.plot(xs, test_costs_collection, 'g', label="Test Cost")
    plt.plot(xs, train_costs_collection, 'r', label="Train Cost") 
    plt.ylabel('Costs')
    plt.xlabel("Number of Iterations")
    plt.title("Cost vs Iterations Plot")    
    plt.legend(loc = 'best')
    plt.savefig('part5(costs).jpg')
    plt.clf()
    plot_20_correct(inputs_test, targets_test, test_prediction, '5')
    plot_10_incorrect(inputs_test, targets_test, test_prediction, '5')
    
def p6(M):
    #mW, mb,rates, costs = TrainNN_softmax(M, 0.01, 1, 20)
    W, b, inputs_train, prediction, targets_train, train_rates_collection, train_costs_collection, inputs_test, test_prediction, targets_test, test_rates_collection, test_costs_collection = TrainNN_softmax(M, 0.01, 1, 20)
    plot10Img(W.T, "weights_p6", 0)
    
def p7_and_8(M):
    num_hidden = 300
    train_x, train_t, valid_x, valid_t = generate_set(M, 100, 50)
    w1,w2,b1,b2=InitNN(train_x.shape[1], num_hidden, 10)   
        
    ##--- Compare gradients ---##
    i_w1, j_w1 = 180, 110
    i_w2, j_w2 = 154, 6
    i_b1 = 11
    i_b2 = 5
    ##--- Compute the approximated gradient ---###
    expected_w1 = part7_check_grad(train_t.T, train_x.T, w1, b1, w2, b2, 1, i_w1, j_w1)
    expected_w2 = part7_check_grad(train_t.T, train_x.T, w1, b1, w2, b2, 2, i_w2, j_w2)
    expected_b1 = part7_check_grad(train_t.T, train_x.T, w1, b1, w2, b2, 3, i_b1, -1)
    expected_b2 = part7_check_grad(train_t.T, train_x.T, w1, b1, w2, b2, 4, i_b2, -1)
    ##--- Computed the precise gradient ---##
    prediction = part7_forwardProp(train_x.T, w1, b1, w2, b2)
    found_w1, found_b1, found_w2, found_b2 \
        = part7_gradient(train_x.T, prediction, train_t.T, w1, b1, w2, b2)
    
    
    print "i_w1, j_w1 =", i_w1, j_w1, "i_b1 =",i_b1, \
          "\napprox w1:", expected_w1, "precise w1:", found_w1[i_w1, j_w1],\
          "\napprox b1:", expected_b1, "precise b1:", found_b1[i_b1][0],\
          "\ndiff wrt w1:", expected_w1 - found_w1[i_w1, j_w1],\
          "\ndiff wrt b1:", expected_b1 - found_b1[i_b1][0], "\n" 
    
    print "i_w2, j_w2 =", i_w2, j_w2, "i_b2 =", i_b2,\
          "\napprox w2:", expected_w2, "precise w2: ", found_w2[i_w2, j_w2],\
          "\napprox b2:", expected_b2, "precise b2", found_b2[i_b2][0],\
          "\ndiff wrt w2:", expected_w2 - found_w2[i_w2, j_w2],\
          "\ndiff wrt b2:", expected_b2 - found_b2[i_b2][0], "\n"
    
    ##---Train model---##
    TrainNN_tanh(train_x, train_t, valid_x, valid_t, num_hidden, 0.01, 1, 400)    

def p9_and_10(M):
    ## Part 5
    num_iter = 200
    num_hidden = 300
    train_x, train_t, test_x, test_t = generate_set(M, 50, 50)
    xs=[i for i in range(num_iter)]
    W1, b1, W2, b2, inputs_train, targets_train, train_prediction, inputs_test, targets_test, test_prediction, train_rates_collection, train_costs_collection, test_rates_collection, test_costs_collection=TrainNN_tanh(train_x, train_t, test_x, test_t, 300, 0.01, 1, num_iter)
    
    
    #TrainNN_tanh(inputs_train, target_train, inputs_valid, target_valid, num_hiddens, eps, momentum, num_epochs):
   
    plt.plot(xs, test_rates_collection, 'g', label="Test Classification Rates")
    plt.plot(xs, train_rates_collection, 'r', label='Train Classification Rates')
    plt.ylabel('Classification Rates')
    plt.xlabel("Number of Iterations")
    plt.title("Classification Rate Vs Iterations Plot")    
    plt.legend(loc = 'best')
    plt.savefig('part9(classificationrate).jpg')
    plt.clf()        
    plt.plot(xs, test_costs_collection, 'g', label="Test Cost")
    plt.plot(xs, train_costs_collection, 'r', label="Train Cost") 
    plt.ylabel('Costs')
    plt.xlabel("Number of Iterations")
    plt.title("Cost vs Iterations Plot")    
    plt.legend(loc = 'best')
    plt.savefig('part9(costs).jpg')
    plt.clf()
    plot_20_correct(inputs_test, targets_test, test_prediction, '9')
    plot_10_incorrect(inputs_test, targets_test, test_prediction, '9')
    plot10Img(W1.T, "weights_p10", 0)
    
def plot_20_correct(test_inputs, test_labels, prediction, part_num):
    i=0
    test_inputs = test_inputs.T
    prediction = prediction.T
    test_labels = test_labels.T
    print test_inputs.shape
    print test_labels.shape
    print prediction.shape
    for j in range(test_inputs.shape[0]):
        if i != 20:
            if np.argmax(test_labels[j])==np.argmax(prediction[j]):
                img=test_inputs[j].reshape(28,28)
                imsave('part_' + part_num + 'correct'+str(i+1),img)
                i+=1
        else:
            break

def plot_10_incorrect(test_inputs, test_labels, prediction, part_num):
    i=0
    test_inputs = test_inputs.T
    prediction = prediction.T
    test_labels = test_labels.T
    for j in range(test_inputs.shape[0]):
        if i != 10:
            if np.argmax(test_labels[j])!=np.argmax(prediction[j]):
                img=test_inputs[j].reshape(28,28)
                imsave('part_' + part_num + 'incorrect'+str(i+1),img)
                i+=1
        else:
            break
        

        
    
if __name__ == "__main__":
    
    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")
    #p1(M,'train')
    #p3_and_4(M)
    #p5(M)
    #p6(M)
    #p7_and_8(M)
    p9_and_10(M)
    
    





 