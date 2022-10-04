import tensorflow as tf
import numpy as np
import keras.layers as layers
import keras.models as models
import data_download as dd
import pandas as pd

from matplotlib import pyplot as plt


def update(sym):
    d1,d2 = dd.to_csv(sym)

    d1.to_csv(f"data/{sym}.csv")
    d2.to_csv(f"data/{sym}-price.csv")

#update("XRP-USD")
def prepare_ds(xx,width):
    Xr = []
    Yr = []
    for i in range(0,len(xx)-width):
        Xr.append(xx[i:i+width])
        Yr.append(xx[i+width])
    Xr = np.array(Xr)
    Yr = np.array(Yr)

    return Xr,Yr

STREAM = "low"
X = pd.read_csv("data/XRP-USD.csv")[STREAM].values 

TEST = 200

xt = X[-TEST:]
xx = X[:-TEST]

WIDTH=31

Xr,Yr = prepare_ds(xx,WIDTH)
Xt, Yt = prepare_ds(xt,WIDTH)


model = models.Sequential()
model.add(layers.Dense(WIDTH))

model.add(layers.Dense(20))
model.add(layers.Dense(10))
#model.add(layers.LSTM(150,input_shape=(1,1)))
model.add(layers.Dense(1))


model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.fit(Xr,Yr,batch_size=10,epochs=20)
yp = []#Yt[0]
yp = model.predict(Xt)

print("Predict:",model.predict(np.array([X[-WIDTH:]])))
print("Pred. Longer:",yp[-1])
print("Actual Longer:",Yt[-1])
import eval_agent
print(eval_agent.profit_agent(yp,Yt))
print(eval_agent.correct_agent(yp,Yt))

plt.plot(yp,label="yp")
plt.plot(Yt,label="yt")
plt.legend()
plt.show()







