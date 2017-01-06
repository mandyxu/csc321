import numpy as np
from utils import *
import matplotlib.pyplot as mp
from get_data import *
from preprocess import *

def l2_distance(A, B):
    if A.shape[1] != B.shape[1]:
        print "Two matrices should have the same number of features."
    
    aa = np.sum(A.T ** 2, axis=0)
    bb = np.sum(B.T ** 2, axis=0)
    ab = np.dot(A, B.T)

    return np.sqrt(aa[:, np.newaxis] - 2 * ab + bb[np.newaxis, :])
    
def run_knn(k, train_inputs, train_targets, valid_inputs):
    
    dist = l2_distance(valid_inputs, train_inputs)
    
    knn = np.argsort(dist, axis=1)[:,:k]
    valid_targets = train_targets.reshape(-1)[knn]
    results = np.zeros([valid_targets.shape[0],1], int)

    for i in range(valid_targets.shape[0]):
        count = np.bincount(valid_targets[i])
        max_count = np.argmax(count)
        results[i] = max_count 
    return results


def find_knn(k, train_inputs, test_inputs):
    
    dist = l2_distance(test_inputs, train_inputs)
    
    knn = np.argsort(dist, axis=1)[:,:k]
   
    failure_cases = np.zeros([5, test_inputs.shape[1]])

    for i in range(knn.shape[0]):
        for j in range(k):
            failure_cases[i+j] = train_inputs[knn[i][j]]
    return failure_cases, knn


def plotImg(data, i, j):
    """ Plots images from data, from index i to index j.
    """
    while i <= j:
        try:
            mp.imshow(data[i].reshape((32,32)), interpolation='nearest')
            mp.gray() 
        except Exception:
            print "Error!"
            break
        raw_input("image "+str(i))
        i += 1
  
def knn_MCR(predictions, targets):
    """ Calculates the mean classification rate of predictions, given targets.
    """
    return (targets == predictions).mean()

if __name__ == "__main__":
    
    
    # Call the function from get_data.py. Starts downloading images.    
    get_data()
    
    # Call the function from preprocess.py. Starts preprocessing.
    data, genders, names, file_name = preprocess()
    

    ## PART 2
    
    ## Note: expected =  [0='butler', 1='radcliffe', 2='vartan', 3='bracco', 4='gilpin', 5='harmon']
    
    train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets = loadData(file_name, "genders")
    
    print "0"
    k_list = [1,3,5,7,9, 11, 13, 15, 17, 19]
    
    # Pick k to be 15 and find the failure cases:
    #k_list = [15]
    
    frac_correct_train = []
    frac_correct_valid = []
    frac_correct_test = []

    
    for item in k_list:
        # output: predict the labels using k nearest neighbours, where (k = item).
        pred_train = run_knn(item, train_inputs, train_targets, train_inputs)
        pred_valid = run_knn(item, train_inputs, train_targets, valid_inputs)
        pred_test = run_knn(item, train_inputs, train_targets, test_inputs)
        
        
        
        
        # Calculate the classification rate
        frac_correct_train.append(knn_MCR(pred_train, train_targets))
        frac_correct_valid.append(knn_MCR(pred_valid, valid_targets))
        frac_correct_test.append(knn_MCR(pred_test, test_targets))
        
    # Plot Classification Rate of Validation set.
    print "cr_Train: \n", frac_correct_train
    print "cr_Validation: \n", frac_correct_valid
    print "cr_Test: \n", frac_correct_test 

   
    ### PART 3
    # Show which of the predictions are wrong -- it is wrong if there is a "Flase"
    print (pred_test == test_targets).T[0]
    
    # m is one of the index of "False" from the output printed above.
    # The following program will find the 5 nearest neighbours of test_inputs[m]
    # and will plot and save the images of the test input itself and its 5nns' 
    # images to folder 
    m = 32
    failure_cases, knn = find_knn(5, train_inputs, np.array([test_inputs[m]]))
    
    print "Target: ", test_targets[m], " has been predicted as ", pred_test[m]
    plotImg(np.array([train_inputs[m]]), 0,0)
    figname ="part3/"+str(test_targets[m][0])+".png"
    mp.savefig(figname)    
    
    
    print "Plotting the image of 5 nearest neighbours of the target:"
    for i in knn[0]:

        plotImg(np.array([train_inputs[i]]), 0,0)
        figname = "part3/"+str(test_targets[m][0])+"_"+str(i)+"_"+str(train_targets[i])+".png"
        mp.savefig(figname)
    
    
    ### Part 4
    mp.axis([1,max(k_list),0.6,1.0])
    # test: solid green; valid: dashed blue
    train_line = mp.plot(k_list, frac_correct_train, label="train") 
    valid_line = mp.plot(k_list, frac_correct_valid, label="validation", linestyle="--") 
    test_line = mp.plot(k_list, frac_correct_test, label="test", linewidth=4) 
    
    # Note: need to check if the "expected" variable is desired in utils.py
    mp.title("Classification Rates (Partial)")
    mp.xlabel("$k$")
    mp.ylabel("classification rates")
    mp.savefig("CR-g-P.png")
    mp.clf()    
    
    
    