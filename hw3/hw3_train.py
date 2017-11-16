import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation, ZeroPadding2D, Conv2D, Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization



def cro_val(X, y, n):
    inds = np.random.permutation(X.shape[0])
    return X[inds[:int(n*X.shape[0])]], y[inds[:int(n*X.shape[0])]], X[inds[int(n*X.shape[0]):]], y[inds[int(n*X.shape[0]):]]


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


def out_data(X, filename):

    y = model.predict(X)
    pred = np.argmax(y, axis = 1)

    
    with open(filename+'.csv','w') as f:
        f.write('id,label\n')
        for i in range(pred.shape[0]):
            f.write( str(i) + ',' + str(pred[i]) + '\n')
    print('output finished!!!\n')
    
    


TRAIN = sys.argv[1]

np.random.seed(0)

print('---------------Reading data---------------')

X, y= read_data(TRAIN)


print('---------------Cross validation---------------')


X, y, X_valid, y_valid = cro_val(X, y, 0.8)

datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, shear_range=0.2, horizontal_flip=True)


print('---------------Model construction---------------')



model = Sequential()

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',input_shape=X.shape[1:4], padding='valid',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))


model.add(Flatten())


model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))

adam = Adam(lr=1e-03,decay=1e-06)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])




lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')
csv_logger = CSVLogger('./model.csv')
checkpointer = ModelCheckpoint(filepath='./model_{epoch:05d}_{val_acc:.5f}.h5', save_best_only=True,period=1,monitor='val_acc')

print('---------------Model fitting---------------')

history = model.fit_generator(datagen.flow(X, y, batch_size=128), steps_per_epoch=1000, epochs=8000, validation_data=(X_valid, y_valid), max_queue_size=100, callbacks=[lr_reducer, earlyStopping, csv_logger, checkpointer])

    


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

confusion_matrix = pd.crosstab(np.argmax(y_valid,-1), model.predict_classes(X_valid), rownames=['label'], colnames=['predict'])


print(confusion_matrix)
confusion_matrix.to_csv('./confusion_matrix.csv')


model.save('./'+str(earlyStopping.best)+'.h5')
