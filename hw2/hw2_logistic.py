import numpy as np
import pandas as pd
import sys


XTRAIN = sys.argv[3]
YTRAIN = sys.argv[4]
XTEST = sys.argv[5]
PREDICTION = sys.argv[6]

def normalize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    return (X - mean) / std , mean , std

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def read_data(filename):
    return pd.read_csv(filename).as_matrix().astype('float')

def Logistic_regression(X,y,epoch,lr):         
    
    X, mean, std = normalize(X) 
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) #add bias
        
    w = np.ones((X.shape[1],1))
    G = np.ones((X.shape[1],1))
    
    
    for i in range(1,epoch+1):
        y_pred = sigmoid(X.dot(w)) 
        loss = y_pred - y
        cost = -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))  
        grad = X.T.dot(loss)
        G += grad**2
        w -= lr*grad #/ np.sqrt(G)
        
        if i % 200 == 0:
            print('epoch : %d | cost : %f' %(i,cost))
        
    return w, mean, std

def test_predict(X,w, mean, std):
    X = (X - mean) / std
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) #add bias
    y = sigmoid(X.dot(w))
    y[y>=0.5] = 1
    y[y< 0.5] = 0    
    
    finalString = "id,label\n"
    for i in range(len(y)) :
        finalString = finalString + str(i+1) + "," + str(int(y[i][0])) + "\n"
    
    f = open(PREDICTION, "w")####$6
    f.write(finalString)
    f.close()
   
    return y

X = read_data(XTRAIN)####$3
y = read_data(YTRAIN)####$4
X_test  = read_data(XTEST)####$5
w, mean, std= Logistic_regression(X,y,1000,0.0001)
ans = test_predict(X_test,w, mean, std)
