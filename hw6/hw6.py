import numpy as np
import pandas as pd
import sys
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


train = np.load(sys.argv[1]) / 255. - 0.5
test = pd.read_csv(sys.argv[2])

input_img = Input(shape=(784,))
encoding_dim=16
encoded = Dense(512, activation='tanh')(input_img)
encoded = Dense(256, activation='tanh')(encoded)
encoded = Dense(128, activation='tanh')(encoded)
encoded = Dense(64, activation='tanh')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

decoded = Dense(64, activation='tanh')(encoder_output)
decoded = Dense(128, activation='tanh')(decoded)
decoded = Dense(256, activation='tanh')(decoded)
decoded = Dense(512, activation='tanh')(decoded)
decoded = Dense(784, activation='linear')(decoded)

autoencoder = Model(input=input_img, output=decoded)   
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(train, train, epochs=30, batch_size=1024, shuffle=False)

model = Model(input=input_img, output=encoder_output)	
reduce = model.predict(train)
data = PCA(n_components=2).fit(reduce)
reduced_data = data.transform(reduce)
kmeans_model = KMeans(n_clusters=2, random_state=1).fit(reduced_data)    
labels = kmeans_model.labels_

test['a'] = test['image1_index'].apply(lambda x:labels[x])
test['b'] = test['image2_index'].apply(lambda x:labels[x])
test['a+b'] = test['a'] + test['b']
test['Ans'] = test['a+b'].apply(lambda x: 1 if x!=1 else 0)

ID = test['ID']
Ans = test['Ans']

out = pd.concat((ID,Ans),axis=1)

out.to_csv(sys.argv[3],index=False)

model.save('./model.h5')