# exercise 8.2.6
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

from TransformedDataset import *

       
def ANN_REG(X,y,hidden_units_range,attributeNames,CV,K):
    N, M = X.shape
    
    #y=y.astype(float)
    y=y.reshape(y.shape[0],-1)
    
    X=X.astype(float)
    
    # Normalize data
    X = stats.zscore(X)
    
    # Parameters for neural network classifier
    # n_hidden_units = 2      # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000
    #hidden_units_range=range(1,8)
    # K-fold crossvalidation
    # K = 3                   # only three folds to speed up this example
    # CV = model_selection.KFold(K, shuffle=True)
    
    # Setup figure for display of learning curves and error rates in fold
    summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    # Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
    # Define the model
    
    outer_errors = []       # list of the best errors in each loop
    best_hidden_units = []  # list of the number of hidden units that correspond to the best error 
    best_nets = []          # list of nets that correspond to the best error
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        
        errors = [] # make a list for storing generalizaition error in each loop
        nets = []   # a list of the nets trained in this loop
        
        for n_hidden_units in hidden_units_range:
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            #print('Training model of type:\n\n{}\n'.format(str(model())))
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            
            print('\n\tBest loss: {}\n'.format(final_loss) , " k2 =", k )
            
            # Determine estimated class labels for test set
            y_test_est = net(X_test)
            
            # Determine errors and errors
            se = (y_test_est.float()-y_test.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            errors.append(mse) # store error rate for current CV fold 
            nets.append(net) #store the coresponding net
        #print(errors)
        opt_val_err= np.min(np.asarray(errors)) #finding the best error
        #print(opt_val_err)
        outer_errors.append(opt_val_err)    # apending the best error to the list of best errors
        opt_hidden_unit=hidden_units_range[np.argmin(np.asarray(errors))] # finding the number of hidden units of the best error
        #print(opt_hidden_unit)
        best_hidden_units.append(opt_hidden_unit) # appending the best error to the list of best number of units
        
        net=nets[np.argmin(np.asarray(errors))] # finding the number of hidden units of the best error
        
        #print('Diagram of best neural net in ' + str(k) + ' fold:')
        weights = [net[i].weight.data.numpy().T for i in [0,2]]
        biases = [net[i].bias.data.numpy() for i in [0,2]]
        tf =  [str(net[i]) for i in [1,2]]
        draw_neural_net(weights, biases, tf, attribute_names=attributeNames)
        # Display the learning curve for the best net in the current fold
        h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
        h.set_label('CV fold {0}'.format(k+1))
        summaries_axes[0].set_xlabel('Iterations')
        summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel('Loss')
        summaries_axes[0].set_title('Learning curves')
        best_nets.append(net)   # appenting best neta to the best net list
        
        
        print('\n\ANN: {}\n', "opt_val_err = ", opt_val_err , "opt_hidden_unit = ",
              opt_hidden_unit, " k2 =", k )
        
    # Display the MSE across folds
    summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(outer_errors)), color=color_list)
    summaries_axes[1].set_xlabel('Fold')
    summaries_axes[1].set_xticks(np.arange(1, K+1))
    summaries_axes[1].set_ylabel('MSE')
    summaries_axes[1].set_title('Test mean-squared-error')
    
    net = best_nets[np.argmin(np.asarray(outer_errors))]
    
    #-------------------------------------RESULTS-------------------------------------------
        
    #print('Diagram of best neural net in last fold:')
    weights = [net[i].weight.data.numpy().T for i in [0,2]]
    biases = [net[i].bias.data.numpy() for i in [0,2]]
    tf =  [str(net[i]) for i in [1,2]]
    draw_neural_net(weights, biases, tf, attribute_names=attributeNames)
    
    # Print the average classification error rate
    #print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))
    
    # When dealing with regression outputs, a simple way of looking at the quality
    # of predictions visually is by plotting the estimated value as a function of 
    # the true/known value - these values should all be along a straight line "y=x", 
    # and if the points are above the line, the model overestimates, whereas if the
    # points are below the y=x line, then the model underestimates the value
    plt.figure(figsize=(10,10))
    y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
    axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
    plt.plot(axis_range,axis_range,'k--')
    plt.plot(y_true, y_est,'ob',alpha=.25)
    plt.legend(['Perfect estimation','Model estimations'])
    plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
    plt.ylim(axis_range); plt.xlim(axis_range)
    plt.xlabel('True value')
    plt.ylabel('Estimated value')
    plt.grid()
    
    plt.show()
    
    opt_val_err= np.min(np.asarray(outer_errors))
    opt_hidden_unit=best_hidden_units[np.argmin(np.asarray(outer_errors))] 
    '''
    print(outer_errors)
    print(best_hidden_units)
    print('ANN optional error '+ str(opt_val_err))
    '''
    print('ANN optional number of hidden units ' + str(opt_hidden_unit))
    
    return opt_val_err, opt_hidden_unit, net

#hidden_units_range=range(1,3)
#attributeNames=attributeNames_regression.tolist()+['ldl']

#CV = model_selection.KFold(3, shuffle=True)

#opt_val_err ,opt_hidden_unit, net = ANN_REG(X_regression,y_regression,hidden_units_range,attributeNames,CV,3)

#print('------------------------------REULTS----------------------------------')
#print('ANN optional error '+ str(opt_val_err))
#print('ANN optional number of hidden units ' + str(opt_hidden_unit))