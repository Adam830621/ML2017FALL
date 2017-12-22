# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 14:19:51 2017

@author: Adam
"""
import numpy as np
import keras.backend as K

def preprocess(data, genders, ages, occupations, movies):

    if data.shape[1] == 4:
        print('Shuffle Data')
        np.random.seed(1019)
        index = np.random.permutation(len(data))
        data = data[index]

    print('Get ID')
    userID = np.array(data[:, 1], dtype=int)
    movieID = np.array(data[:, 2], dtype=int)

    print('Get Features')
    userGender = np.array(genders)[userID]
    userAge = np.array(ages)[userID]
    userOccu = np.array(occupations)[userID]
    movieGenre = np.array(movies)[movieID]

    print('Normalize Ages')
    std = np.std(userAge)
    userAge = userAge / std

    Rating = []
    if data.shape[1] == 4:
        print('Get Ratings')
        Rating = data[:, 3].reshape(-1, 1)

    print('userID:', userID.shape)
    print('movieID:', movieID.shape)
    print('userGender:', userGender.shape)
    print('userAge:', userAge.shape)
    print('userOccu:', userOccu.shape)
    print('movieGenre:', movieGenre.shape)
    print('Y:', np.array(Rating).shape)
    return userID, movieID, userGender, userAge, userOccu, movieGenre, Rating
    
def GTN(genres, all_genres):
    result = []
    for g in genres.split('|'):
        if g not in all_genres:
            all_genres.append(g)
        result.append( all_genres.index(g) )
    return result, all_genres

def TC(index, categories):
    categorical = np.zeros(categories, dtype=int)
    categorical[index] = 1
    return list(categorical)

def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )