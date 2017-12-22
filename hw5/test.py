#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import csv
import numpy as np
import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Dense, Dropout
from keras.layers.merge import Dot, Add, Concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from functions import *

TEST = sys.argv[1]
RES = sys.argv[2]
MOVIES = sys.argv[3]
USERS = sys.argv[4]

movies, all_genres = [[]] * 3953, []
with open(MOVIES, 'r', encoding='latin-1') as f:
    f.readline()
    for line in f:
        movieID, title, genre = line[:-1].split('::')
        genre_numbers, all_genres = GTN(genre, all_genres)
        movies[int(movieID)] = genre_numbers
        
        
categories = len(all_genres)
for i, m in enumerate(movies):
    movies[i] = TC(m, categories)
    
    
genders, ages, occupations = [[]]*6041, [[]]*6041, [ [0]*21 ]*6041
categories = 21
with open(USERS, 'r', encoding='latin-1') as f:
    f.readline()
    for line in f:
        userID, gender, age, occu, zipcode = line[:-1].split('::')
        genders[int(userID)] = 0 if gender is 'F' else 1
        ages[int(userID)] = int(age)
        occupations[int(userID)] = TC(int(occu), categories)
        
        
 
 
 
print('============================================================')
print('Test Model')
model = load_model('./model_best.h5', custom_objects={'rmse': rmse})
model.summary()
test = []
with open(TEST, 'r') as f:
    f.readline()
    reader = csv.reader(f)
    for row in reader:
        dataID, userID, movieID = row
        test.append( [dataID, int(userID), int(movieID)] )
test = np.array(test)
ID = np.array(test[:, 0]).reshape(-1, 1)


userID, movieID, userGender, userAge, userOccu, movieGenre, _Y = preprocess(test, genders, ages, occupations, movies)

result = model.predict([userID, movieID, userGender, userAge, userOccu, movieGenre])

print('Output Result')
rating = np.clip(result, 1, 5).reshape(-1, 1)
output = np.array( np.concatenate((ID, rating), axis=1))

print('============================================================')
print('Save Result')

with open(RES, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['TestDataID', 'Rating'])
    writer.writerows(output)
    
    
    