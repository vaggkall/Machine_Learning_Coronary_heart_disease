import numpy as np
from sklearn import model_selection

def baseline(y,CV):
    errors = []
    means = []
    for train_index, test_index in CV.split(y):
        y_train = y[train_index]
        y_test = y[test_index] 
        means.append(y_train.mean())
        errors.append(np.square(y_test-y_train.mean()).sum(axis=0)/y_test.shape[0])
    print("--------------------------------------BASELINE RESULTS-------------------------------------------")
    print(means)
    print(errors)
    opt_baseline_err= np.min(np.asarray(errors)) #finding the best error
    print('Optimal error ' + str(opt_baseline_err))
    opt_mean=means[np.argmin(np.asarray(errors))] # finding the number of hidden units of the best error
    print('optimal mean ' + str(opt_mean))
    print("----------------------------------END OF BASELINE RESULTS-----------------------------------------")
    return opt_baseline_err , opt_mean
    
        
        