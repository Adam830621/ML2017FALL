import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation, ZeroPadding2D, Conv2D, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
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

    if test == True:
        return np.array(X).reshape(-1,48,48,1).astype('float') / 255
    else:
        return np.array(X).reshape(-1,48,48,1).astype('float') / 255, np_utils.to_categorical(y, 7)


def out_data(X):

    y = model.predict(X)
    pred = np.argmax(y, axis = 1)

    
    with open(RESULT,'w') as f:
        f.write('id,label\n')
        for i in range(pred.shape[0]):
            f.write( str(i) + ',' + str(pred[i]) + '\n')
    print('---------------Output finished---------------\n')
    
    




np.random.seed(0)

print('---------------Reading data---------------')


X_test = read_data(TEST,test=True)

model = load_model('./model_00027_0.68896.h5')
model.summary()
out_data(X_test)
