from pylab import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat


t = int(time.time())
#t = 1454219613
print "t=", t
random.seed(t)

ntargets = 6
ndim = 32
ndimsqr = ndim ** 2

  

import tensorflow as tf

def get_train_batch(M, N):
    n = N/ntargets
    batch_xs = np.zeros((0,)+M['train0'][0].shape)
    batch_y_s = np.zeros( (0, ntargets))
    
    train_k =  ["train"+str(i) for i in range(ntargets)]

    train_size = len(M[train_k[0]])
    #train_size = 5000
    
    for k in range(ntargets):
        train_size = len(M[train_k[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = np.zeros(ntargets)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s
    
def get_test(M):
    batch_xs = np.zeros((0,)+M['test0'][0].shape)
    batch_y_s = np.zeros( (0, ntargets))
    
    test_k =  ["test"+str(i) for i in range(ntargets)]
    for k in range(ntargets):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = np.zeros(ntargets)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    

def get_valid(M):
    batch_xs = np.zeros((0,)+M['valid0'][0].shape)
    batch_y_s = np.zeros( (0, ntargets))
    
    valid_k =  ["valid"+str(i) for i in range(ntargets)]
    for k in range(ntargets):
        batch_xs = vstack((batch_xs, ((array(M[valid_k[k]])[:])/255.)  ))
        one_hot = np.zeros(ntargets)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[valid_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    

def get_train(M):
    batch_xs = np.zeros((0,)+M['train0'][0].shape)
    batch_y_s = np.zeros( (0, ntargets))
    
    train_k =  ["train"+str(i) for i in range(ntargets)]
    for k in range(ntargets):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = np.zeros(ntargets)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s

def plotImg(M):
    for i in range (5):        
        for img in M['train' + str(i)]:
            imarray = np.reshape(np.array(img),(ndim,ndim))
            plt.imshow(imarray, interpolation='nearest')
            plt.show()    



def part1(nhid=300):
    test_x, test_y = get_test(M)
    valid_x, valid_y = get_valid(M)
    train_x, train_y = get_train(M)    
    x = tf.placeholder(tf.float32, [None, ndimsqr])    
    
    nhid = nhid
    W0 = tf.Variable(tf.random_normal([ndimsqr, nhid], stddev=0.01))/10
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))/10
    
    W1 = tf.Variable(tf.random_normal([nhid, ntargets], stddev=0.01))/10
    b1 = tf.Variable(tf.random_normal([ntargets], stddev=0.01))/10
    
    #snapshot = cPickle.load(open("new_snapshot"+str(1999)+".pkl"))
    #W0 = tf.Variable(snapshot["W0"])
    #b0 = tf.Variable(snapshot["b0"])
    #W1 = tf.Variable(snapshot["W1"])
    #b1 = tf.Variable(snapshot["b1"])
        
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1
    
    W = tf.Variable(tf.random_normal([ndimsqr, ntargets], stddev=0.01))/10
    b = tf.Variable(tf.random_normal([ntargets], stddev=0.01))/10
    layer = tf.matmul(x, W)+b
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, ntargets])
    
    lam = 0.0005
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    NLL = -tf.reduce_sum(y_*tf.log(y)+decay_penalty)
    
    train_step = tf.train.GradientDescentOptimizer(5e-2).minimize(NLL)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)

    
    train_mce = []
    valid_mce = []
    test_mce = []
    
    for i in range(100000):        
        #print i  
        batch_xs, batch_ys = get_train_batch(M, 50)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
      
        if i % 1 == 0:        
            print "i=",i
            
            loss = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
            #print "Valid:", loss
            valid_mce.append(loss)
            
            loss = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
            #print "Test:", loss
            test_mce.append(loss)
            
            loss = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
            #print "Train:", loss
            train_mce.append(loss)
            if (loss > 0.99):
                break            
            
            print "Penalty:", sess.run(decay_penalty)
            
    snapshot = {}
    snapshot["W0"] = sess.run(W0)
    snapshot["W1"] = sess.run(W1)
    snapshot["b0"] = sess.run(b0)
    snapshot["b1"] = sess.run(b1)
    print snapshot["W0"].shape
    print snapshot["W1"].shape
    #final_hidden_layer = sess.run(layer1, feed_dict={x: test_x, y_: test_y})
    j = 0
    for i in range(snapshot["W0"].shape[1]):
        hidden_unit = snapshot["W0"][:,i]
        print hidden_unit.shape
        img = hidden_unit.reshape((ndim,ndim))
        plt.clf()
        plt.imshow(img, cmap = cm.coolwarm)
        
        raw = raw_input("Please enter 'break' to stop displaying. \nEnter 'save' to save the image. \nEnter anything else to display next image\n")
        if ("break" == raw):
            break
        elif ("save" == raw):
            plt.savefig("Part3_" + str(nhid) + "_Weight_" + str(j))
            j += 1
        
        

    cPickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))  
    plt.clf()   
    plt.plot(train_mce, "r", label="train")
    plt.plot(valid_mce, "g", label="valid")
    plt.plot(test_mce, "b", label="test")
    title = "Learning curves (nhid=" + str(nhid) + ")"
    plt.title(title)  
    plt.legend(loc="best")
    plt.savefig(title)
    
    raw_input("Press Enter to finish run")
    return train_mce, valid_mce, test_mce
  
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    
    assert c_i%group==0
    assert c_o%group==0
    
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())


def part3():
    part1(300)
    part1(800)


def convolution():
    image = tf.placeholder(tf.float32,[1,100,100,3])
    net_data=load("bvlc_alexnet.npy").item() 
   
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(image, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)
    
    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    
    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    
    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    
    
    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    
    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
    
    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)
    
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    
    ################################################################################
    
    #Output:
    
    #H = loadmat("mnist_all_100.mat")     
    #test_x, test_y = get_test(H)
    #valid_x, valid_y = get_valid(H)
    #train_x, train_y = get_train(H) 
    
    out_ndim = 9600
    conv_sets = [np.empty((1,out_ndim)), 
                 np.empty((1,out_ndim)),
                 np.empty((1,out_ndim))]
    
    data_names = ['test','valid','train']
    data_sets = [test_x, valid_x, train_x]
    for i in range(3):
        set_size = data_sets[i].shape[0]
        data_set = data_sets[i]
        for j in range(set_size):
            im = sess.run(conv4, feed_dict={image:array([data_set[j]])})
            im = np.reshape(im, (1,out_ndim))
            conv_sets[i]=np.concatenate((conv_sets[i],im))
            
    conv_sets[0] = conv_sets[0][1:]
    conv_sets[1] = conv_sets[1][1:]
    conv_sets[2] = conv_sets[2][1:]
    
    J = {}
    for i in range(3):
        set_size = data_sets[i].shape[0]/6
        for j in range(6):
            idx = set_size * j
            J[data_names[i]+str(j)] = conv_sets[i][idx:(idx+set_size)]    
            
    return J


def part2(tol=0.985):
    nhid = 400
    out_ndim = 9600
     
    J = convolution()
##-------------end test-------------
    print "FINISHED FEEDING====="

    x = tf.placeholder(tf.float32, [None, out_ndim]) 
        
    #snapshot = cPickle.load(open("part1_weights.pkl"))
    #W0 = tf.Variable(snapshot["W0"])
    #b0 = tf.Variable(snapshot["b0"])
    #W1 = tf.Variable(snapshot["W1"])
    #b1 = tf.Variable(snapshot["b1"])
        
    

    W1 = tf.Variable(tf.random_normal([out_ndim, ntargets], stddev=0.01))/10
    b1 = tf.Variable(tf.random_normal([ntargets], stddev=0.01))/10
    
    ##--- Load weights saved from part 2
    #snapshot = cPickle.load(open("part2_new_snapshot"+str(i)+".pkl"))
    #W1 = tf.Variable(snapshot["W1"])
    #b1 = tf.Variable(snapshot["b1"])    
    
    layer1 = x
    layer2 = tf.matmul(layer1, W1)+b1   
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, ntargets])
    
    lam = 0.0001


    
    decay_penalty =lam*tf.reduce_sum(lam*tf.reduce_sum(tf.square(W1)))
    NLL = -tf.reduce_sum(y_*tf.log(y)+decay_penalty)
    
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(NLL)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    

    test_x, test_y = get_test(J)
    valid_x, valid_y = get_valid(J)
    train_x, train_y = get_train(J)
    
    train_mce = []
    valid_mce = []
    test_mce = []  
    
    loss = 0
    i =0
    while loss < 0.9 and i < 100000:
        i += 1
        batch_xs, batch_ys = get_train_batch(J, 200)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
        if i % 1 == 0:
        
            #print "i=",i
            
            loss1 = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
            #print "Valid:", loss
            valid_mce.append(loss1)
            pred = sess.run(y, feed_dict={x: valid_x, y_: valid_y})
            
            loss2 = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
            #print "Test:", loss
            test_mce.append(loss2)
            
            loss3 = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
            #print "Train:", loss
            train_mce.append(loss3)
            if ( (i % 50) == 0):
                print i, loss1, loss2, loss3
            
            #print "Penalty:", sess.run(decay_penalty)
    print np.argmax(pred,1)
    snapshot = {}
    snapshot["W1"] = sess.run(W1)
    snapshot["b1"] = sess.run(b1)
    cPickle.dump(snapshot,open("mandyweights.pkl", "w"))
    
    plt.plot(train_mce, "r", label="train")
    plt.plot(valid_mce, "g", label="valid")
    plt.plot(test_mce, "b", label="test")
    title = "Learning curves - part 2"
    plt.title(title)  
    plt.legend(loc="best")
    plt.savefig(title)    
    

def part4(input_set, target_set, k):
    out_ndim = 9600
    image = tf.placeholder(tf.float32,[1,100,100,3])
    
    snapshot_path = "mandyweights.pkl"
    net_data=load("bvlc_alexnet.npy").item() 
   
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(image, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)
    
    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    
    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    
    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    
    
    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    
    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
    
    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.reshape(tf.nn.relu(conv4_in),(1,out_ndim))
    
    
    W1 = tf.Variable(tf.random_normal([out_ndim, ntargets], stddev=0.001))
    b1 = tf.Variable(tf.random_normal([ntargets], stddev=0.001))
    
    ##--- Load weights saved from part 2
    snapshot = cPickle.load(open(snapshot_path))
    W1 = tf.Variable(snapshot["W1"])
    b1 = tf.Variable(snapshot["b1"])    
    

    layer2 = tf.matmul(conv4, W1)+b1   
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, ntargets])
    
    prediction = tf.argmax(y,1)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  
    gradients = tf.gradients(y, image)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    
    picture = array([input_set[k]])        
    im = sess.run(conv4, feed_dict={image:picture})
    
    target = np.reshape(target_set[k],(1,6))
    pred = sess.run(prediction,feed_dict={image:picture, y_:target})
    grad = sess.run(gradients,feed_dict={ image:picture,y_:target})
    print k, pred, target
    return pred, grad
    
def part5(train_x, train_y, k):
    pred, grad = part4(train_x, train_y, k)
    grad[0].clip(0)
    m = np.max(grad[0])
    if m == 0:
        m = 1e-6
    grad[0] = grad[0] / m
    
    
    plt.clf()
    plt.imshow(grad[0][0][:])
    plt.savefig("Part5_gradient")
    
    plt.clf()
    plt.imshow(train_x[k])
    plt.savefig("Part5_original")
    

M = loadmat("mnist_all.mat")
H = loadmat("mnist_all_100.mat")
test_x, test_y = get_test(H)
valid_x, valid_y = get_valid(H)
train_x, train_y = get_train(H)


if __name__ == "__main__":
    
    # NOTE YOU HAVE TO RUN THIS STEP BY STEP
    print "Load data"
    #load_data()
    #raw_input("Press enter to process data for part 1 and 3 (grayscaled)")
    #preprocess()
    #raw_input("Press enter to process data for part 2, 4 and 5 (rgb)")
    #preprocess_rgb()
    #print "Starting part 1"
    #part1()
    #raw_input("Press enter to start part 2")
    #part2()
    #raw_input("Press enter to start part 3")
    #part3()
    #raw_input("Press enter to start part 4. Note we are using the 88th picture from the training set for parts 4 and 5")
    #part4(train_x, train_y, 87)
    #raw_input("Press enter to start part 5")
    #part5(train_x, train_y, 87)