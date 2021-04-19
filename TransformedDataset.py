# Transformation of DataSet
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from scipy.linalg import svd



from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

# Load the Iris csv data using the Pandas library
filename = '../Data/Heart_deseas_Dataset.txt'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = df.values  

# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.

columns=[1,2,3,4,6,7,8,9,10] #column 5 removed
X = raw_data[:,columns]
'''
sbp = X[:,1:2]
tobacco = X[:,2:3]
ldl = X[:,3:4]
adiposity = X[:,4:5] 
typea = X[:,5:6]
obesity = X[:,6:7]
alcohol = X[:,7:8]
age = X[:,8:9]
'''

y=raw_data[:,10]
y_classification= raw_data[:,10]


columns_regression=[1,2,4,6,7,8,9,10] #column 5,3 removed
y_regression=raw_data[:,3]
y_regression=y_regression.astype(float)
X_regression = raw_data[:,columns_regression]


Xn=np.asarray([[0 if i=='Present' else 1 for i in raw_data[:,5]]])

# We need to replace the last column in X (which was the not encoded
# version of the data) with the encoded version:
X = np.concatenate( (X[:, :-1], Xn.T), axis=1) #!concatinate replaces

#in regression we need the both attributes famhist and chd
X_regression = np.concatenate( (X_regression, Xn.T), axis=1) 

# Extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[columns])
attributeNames = np.append(attributeNames, ['famhis'])

attributeNames_regression = np.asarray(df.columns[columns_regression])
attributeNames_regression = np.append(attributeNames_regression, ['famhis'])
attributeNames_regression_ANN = np.append(attributeNames_regression, ['famhis']).tolist() + ['ldl']

# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by 
# extracting the strings for each sample from the raw data loaded from the csv:
classLabels = raw_data[:,-1] # -1 takes the last column
# Then determine which classes are in the data by finding the set of 
# unique class labels 
classNames = np.unique(classLabels)
# We can assign each type of class with a number by making a
# Python dictionary as so:
classDict = dict(zip(classNames,range(len(classNames))))
# The function zip simply "zips" togetter the classNames with an integer,
# like a zipper on a jacket. 
# For instance, you could zip a list ['A', 'B', 'C'] with ['D', 'E', 'F'] to
# get the pairs ('A','D'), ('B', 'E'), and ('C', 'F'). 
# A Python dictionary is a data object that stores pairs of a key with a value. 
# This means that when you call a dictionary with a given key, you 
# get the stored corresponding value. Try highlighting classDict and press F9.
# You'll see that the first (key, value)-pair is ('Iris-setosa', 0). 
# If you look up in the dictionary classDict with the value 'Iris-setosa', 
# you will get the value 0. Try it with classDict['Iris-setosa']

# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels])
#y_regression = np.array([classDict[cl] for cl in classLabels])
# In the above, we have used the concept of "list comprehension", which
# is a compact way of performing some operations on a list or array.
# You could read the line  "For each class label (cl) in the array of 
# class labels (classLabels), use the class label (cl) as the key and look up
# in the class dictionary (classDict). Store the result for each class label
# as an element in a list (because of the brackets []). Finally, convert the 
# list to a numpy array". 
# Try running this to get a feel for the operation: 
# list = [0,1,2]
# new_list = [element+10 for element in list]

# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)


#---------------------Standarization------------------------------------------
# Subtract mean value from data
Y = X.astype(float)
Y = X - np.ones((N,1))*X.mean(axis=0)
Y = Y.astype(float)
X_standarised=Y*(1/np.std(Y,0))



