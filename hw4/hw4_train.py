import sys
import numpy as np
import pandas as pd
from gensim.models import word2vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Masking, GRU, Bidirectional
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint

TRAIN = sys.argv[1]
UN_TRAIN = sys.argv[2]


#---------Read data-----------
data = pd.read_csv(TRAIN, sep="\+\+\+\$\+\+\+", engine='python', header=None, names=['label', 'text'])
data['text'] = data['text'].apply((lambda x: x.lower()))

word_dim = 69
max_len = 39
data_size = data.shape[0]
tmp = []  
i = 0   

model = word2vec.Word2Vec.load('emb2')


for r in data.text.str.split():
    try:
        tmp.append(np.pad(model[r],((max_len-len(r),0), (0,0)), mode='constant'))
    except:
        string = []
        for ele in r:
            if ele in model.wv.vocab:
                string.append(ele)
        if len(string) == 0 :
            tmp.append(np.zeros([max_len,word_dim]).astype('float32'))
        else:
            tmp.append(np.pad(model[string],((max_len-len(string),0), (0,0)), mode='constant'))
    i += 1
    print("\rtesting data : " + repr(i), end="", flush=True)


x_train = np.array(tmp)
y_train = data['label'].values
#----------Building model-------------
model = Sequential()
model.add(Masking(input_shape=x_train.shape[1:]))
model.add(Bidirectional(GRU(128,activation='tanh',dropout=0.4)))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))    


model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
       
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')
csv_logger = CSVLogger('./model.log')
checkpointer = ModelCheckpoint(filepath='./model.h5', save_best_only=True,period=1,monitor='val_acc')

#----------Model fitting-------------
model.fit(x_train, y_train, batch_size=512, epochs=8000, validation_split=0.1, callbacks=[lr_reducer, earlyStopping, csv_logger, checkpointer])
