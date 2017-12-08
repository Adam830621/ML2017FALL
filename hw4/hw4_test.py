import sys
import numpy as np
import pandas as pd
from gensim.models import word2vec
from keras.models import load_model


TEST = sys.argv[1]
RESULT = sys.argv[2]



#---------Read data-----------
test = pd.read_csv(TEST, sep="\n", skiprows=1, engine='python', header=None, names=['text'])
X_test = test['text'].str.split(',', 1 , expand=True)
test['text'] = X_test[1].apply((lambda x: x.lower()))

word_dim = 69
max_len = 39
data_size = test.shape[0]
tmp = [] 
i = 0


model = word2vec.Word2Vec.load('./emb2')

   
#-------------Make sure the words are in my model-----
for r in test.text.str.split():
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

    
    
Test_data = np.array(tmp)


model = load_model('./model_best.h5')
model.summary()
pred = model.predict(Test_data, verbose=1, batch_size=1024)

pred[pred <0.5] = 0
pred[pred >=0.5] = 1 



with open(RESULT,'w') as f:
    f.write('id,label\n')
    for i in range(pred.shape[0]):
        f.write( str(i) + ',' + str(int(pred[i])) + '\n')
print('Predict finished\n')









