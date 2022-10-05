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

update("XRP-USD")
def prepare_ds(xx,width):
    Xr = []
    Yr = []
    for i in range(0,len(xx)-width):
        Xr.append(xx[i:i+width])
        Yr.append(xx[i+width])
    Xr = np.array(Xr)
    Yr = np.array(Yr)

    return Xr,Yr

champion_model_path = "models/champion"
STREAM = "low"
X = pd.read_csv("data/XRP-USD.csv")[STREAM].values 

TEST = 200

xt = X[-TEST:]
xx = X[:-TEST]

WIDTH=31

Xr,Yr = prepare_ds(xx,WIDTH)
Xt, Yt = prepare_ds(xt,WIDTH)


challenger = models.Sequential()
challenger.add(layers.Dense(WIDTH))

challenger.add(layers.Dense(20))
challenger.add(layers.Dense(10))
challenger.add(layers.Dense(1))


challenger.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
challenger.fit(Xr,Yr,batch_size=10,epochs=20)
yp = []#Yt[0]
yp = challenger.predict(Xt)

print("Predict:",challenger.predict(np.array([X[-WIDTH:]])))
print("Pred. Longer:",yp[-1])
print("Actual Longer:",Yt[-1])
import eval_agent
import datetime
chl_perf1 = eval_agent.profit_agent(yp,Yt)
chl_perf2 = eval_agent.correct_agent(yp,Yt)
try:
    champion = models.load_model(champion_model_path)
    yp_chm = champion.predict(Xt)
    chm_perf1 = eval_agent.profit_agent(yp_chm,Yt)
    chm_perf2 = eval_agent.correct_agent(yp_chm,Yt)
    if chl_perf2 > chm_perf2:
        print("Challenger won: ",chl_perf2, " > ", chm_perf2)
        print("Other perf: ",chl_perf1, " > ", chm_perf1)
        challenger.save(champion_model_path)
        print("New Champion ",datetime.datetime.today())
    else:
        print("Champion won: ",chm_perf2,"  vs ",chl_perf2)
except:
    challenger.save(champion_model_path)


plt.plot(yp,label="yp")
plt.plot(Yt,label="yt")
plt.legend()
plt.show()







