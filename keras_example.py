
from builtins import range
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def get_normalized_data():
    print("Reading in and transforming data...")

    if not os.path.exists('../../movielens-20m-dataset/train.csv'):
        print('Looking for ../large_files/train.csv')
        print('You have not downloaded the data and/or not placed the files in the correct location.')
        print('Please get the data from: https://www.kaggle.com/c/digit-recognizer')
        print('Place train.csv in the folder large_files adjacent to the class folder')
        exit()

    df = pd.read_csv('../../movielens-20m-dataset/train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0]

    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]

    # normalize the data
    mu = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)
    np.place(std, std == 0, 1)
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std

    return Xtrain, Ytrain




if __name__ == '__main__':


    X, Y = get_normalized_data()
    N, D = X.shape
    K = len(set(Y))
    Y = y2indicator(Y)

    model = Sequential()
    model.add(Dense(units=500, input_dim=D))
    model.add(Activation('relu'))
    model.add(Dense(units=300))
    model.add(Activation('relu'))
    model.add(Dense(units=K))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    r = model.fit(X, Y, validation_split=0.33, epochs=15, batch_size=32)
    print("Returned:", r)
    print(r.history.keys())
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    plt.plot(r.history['acc'], label='acc')
    plt.plot(r.history['val_acc'], label='val_acc')
    plt.legend()
    plt.show()
