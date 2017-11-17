import pandas as pd
import numpy as np
import sys
from keras.models import load_model
from keras.utils import np_utils

TEST = sys.argv[1]
RESULT = sys.argv[2]

def read_data(filename,test=False):

    data = pd.read_csv(filename)
    index = list(data)
    y = data[index[0]]
    X = []
    
    for i in data[index[1]]:
        X.append(i.split(' '))  

    X = np.array(X).reshape(-1,48,48,1).astype('float') / 255           
     
    if test == True:
        return X
    else:
        return X, np_utils.to_categorical(y, 7)


def out_data(X):

    prediction = model.predict(X)
    pred_result = np.argmax(prediction, axis = 1)
    
    with open(RESULT,'w') as f_out:
        f_out.write('id,label\n')
        for i in range(pred_result.shape[0]):
	        string = str(i) + ',' + str(pred_result[i]) + '\n'
	        f_out.write(string)
            
    print('---------------Output finished---------------\n')
    
    




np.random.seed(0)

print('---------------Reading data---------------')


X_test = read_data(TEST,test=True)

model = load_model('./model_00027_0.68896.h5')
model.summary()
out_data(X_test)
