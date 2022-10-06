import numpy as np
import keras.layers as layers
import keras.models as models
import data_download as dd
import pandas as pd
import eval_agent
import datetime
import asyncio as aio
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

champion_model_path = "models/champion"
STREAM = "low"
WIDTH=31
TEST=200
WAIT_INTERVAL = 600*24
HOST = "192.168.0.12:8086"
TOKEN = "_klZ_yw6Y8V7CqDesePVuAqWY0BMCYXjOJ3LshdQJpgwfsPrhtvtNZbGJlebZAxCYuGafXpPlnTX11MNpgdOcQ=="
ORG = "fef"
BUCKET = "model-ops"
SYM = "XRP-USD"
model_id = SYM+"_"+STREAM

def update(sym):
    d1,d2 = dd.to_csv(sym)

    d1.to_csv(f"data/{sym}.csv")
    d2.to_csv(f"data/{sym}-price.csv")



def write_model_perf(client,sym,key,value,time,id=model_id):
    
    t = int(datetime.datetime.timestamp(time))*1000_000_000
    print(1665086257573000000,t)
    print(">",sym,id,time,t,key,value)
    p = influxdb_client.Point(sym).tag("id",id).field(key,float(value)).time(t)
    with client.write_api(write_options=SYNCHRONOUS) as w:
        w.write(org=ORG,bucket=BUCKET,record=p)


def prepare_ds(xx,width):
    Xr = []
    Yr = []
    for i in range(0,len(xx)-width):
        Xr.append(xx[i:i+width])
        Yr.append(xx[i+width])
    Xr = np.array(Xr)
    Yr = np.array(Yr)

    return Xr,Yr



async def run(client):
    while True:
        try:
            today = datetime.datetime.today()
            delta = datetime.timedelta(days=1)
            yesterday = today - delta
            print("Time: ",today, yesterday)

            update(SYM)
            X = pd.read_csv(f"data/{SYM}.csv")[STREAM].values 

            xt = X[-TEST:]
            xx = X[:-TEST]

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
            tom_pred = challenger.predict(np.array([X[-WIDTH:]]))
            yest_actual = Yt[-1]
            write_model_perf(client,SYM+"_predicted","pred",tom_pred[0][0],today)
            write_model_perf(client,SYM+"_actual","actual",yest_actual,yesterday)
            print("Predict:",tom_pred)
            print("Pred. Longer:",yp[-1])
            print("Actual Longer:",yest_actual)
            
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
                    write_model_perf(client,SYM+"_meta", "change",1,today)
                else:
                    print("Champion won: ",chm_perf2,"  vs ",chl_perf2)
            except:
                challenger.save(champion_model_path)
        except Exception as e:
            print(e)
        await aio.sleep(WAIT_INTERVAL)

if __name__ == "__main__":

    client = influxdb_client.InfluxDBClient(url=HOST, token=TOKEN,org=ORG, debug=True)
    
    loop = aio.get_event_loop()
    loop.run_until_complete(run(client))








