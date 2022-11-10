import tslibs
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

BASE_LEN = 31
ENC_LEN = 7

def build_ds(X, _len=BASE_LEN):
    ds_x = []
    for i in range(_len-1,len(X)):
        
        ds_x.append(X.loc[i-_len:i])
    return ds_x

def split(X,test_ratio=0.1):
    total = len(X)
    spl = int(test_ratio*total)

    return X[:spl],X[spl:]



raw = pd.read_csv("data/XRP-USD.csv")

data = build_ds(raw)
train,test = split(data)
print(data[0]["low"])
print(train[0]["low"])


print(len(train),len(test),len(data))



l1 = keras.layers.Dense(BASE_LEN)
l2 = keras.layers.Dense(10)
l3 = keras.layers.Dense(ENC_LEN)
l4 = keras.layers.Dense(10)
l5 = keras.layers.Dense(BASE_LEN)

seq = keras.models.Sequential()
seq.add(l1)
seq.add(l2)
seq.add(l3)
seq.add(l4)
seq.add(l5)

seq.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
seq.fit()

