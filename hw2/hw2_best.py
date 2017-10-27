import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier

XTRAIN = sys.argv[3]
YTRAIN = sys.argv[4]
XTEST = sys.argv[5]
PREDICTION = sys.argv[6]

def normalize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    res = (X - mean) / std
    return res 


def read_data(filename):
    return pd.read_csv(filename).as_matrix().astype('float')

def add_bias(X):
    return np.concatenate((np.ones((X.shape[0],1)),X),axis=1)



def cross_validation(X, y, n):
    inds = np.random.permutation(X.shape[0])
    X_train = X[inds[:int(n*X.shape[0])]]
    y_train = y[inds[:int(n*X.shape[0])]]
    X_valid = X[inds[int(n*X.shape[0])]:]
    y_valid = y[inds[int(n*X.shape[0])]:]
    
    return X_train, y_train, X_valid, y_valid



def test_predict(X):

    y = clf.predict(X)
    y[y>=0.5] = 1
    y[y< 0.5] = 0    
    
    finalString = "id,label\n"
    with open(PREDICTION, "w") as f:####$6
        for i in range(len(y)) :
            finalString = finalString + str(i+1) + "," + str(int(y[i])) + "\n"
        f.write(finalString)
    
    
    return y

np.random.seed(0)    
X = read_data(XTRAIN)####$3
y = read_data(YTRAIN)####$4
X_test  = read_data(XTEST)####$5
temp = np.concatenate((X,X_test) , axis=0)
temp = normalize(temp)
X = temp[:X.shape[0]]
X_test = temp[X.shape[0]:]
X, y, X_valid, y_valid = cross_validation(X , y, 0.75)
clf = GradientBoostingClassifier(learning_rate=0.01,max_depth=7,max_features=X.shape[1],n_estimators=512,verbose=True)
clf.fit(X, y.reshape(-1))
mse_train = mean_squared_error(y, clf.predict(X))
mse_valid = mean_squared_error(y_valid, clf.predict(X_valid))
print("train_acc: %f | valid_acc: %f" %(1-mse_train,1-mse_valid))



y_pre = test_predict(X_test)



