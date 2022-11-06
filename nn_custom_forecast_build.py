import numpy as np
import keras.layers as layers
import keras.models as models
import keras
import data_download as dd
import pandas as pd
import eval_agent
import datetime
import asyncio as aio
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS


champion_model_path = "models/multi_inputs/champion"
STREAM = "low"
SUPPORT = "open"
WIDTH=31
TEST=200
WAIT_INTERVAL = 60*60*24 #24 hrs
HOST = "192.168.0.12:8086"
TOKEN = "_klZ_yw6Y8V7CqDesePVuAqWY0BMCYXjOJ3LshdQJpgwfsPrhtvtNZbGJlebZAxCYuGafXpPlnTX11MNpgdOcQ=="
ORG = "fef"
BUCKET = "model-ops"
SYM = "XRP-USD"
METRIC_SYM = "XRP-USD-MULTI"
model_id = SYM+"_"+STREAM
ENABLE_DB_WRITE = True
ENABLE_TIMER = True
def update(sym):
    d1 = dd.to_csv(sym)

    d1.to_csv(f"data/{sym}.csv")




def write_model_perf(client,sym,key,value,time,id=model_id):
    if ENABLE_DB_WRITE:
        t = int(datetime.datetime.timestamp(time))*1000_000_000
        p = influxdb_client.Point(sym).tag("id",id).field(key,float(value)).time(t)
        with client.write_api(write_options=SYNCHRONOUS) as w:
            w.write(org=ORG,bucket=BUCKET,record=p)
    else:
        print("Data > ",sym," : ",id,"  ->  ",time,0,key,value)


def prepare_ds(xx,width):
    Xr = []
    Yr = []
    for i in range(0,len(xx)-width):
        Xr.append(xx[i:i+width])
        Yr.append(xx[i+width])
    Xr = np.array(Xr)
    Yr = np.array(Yr)

    return Xr,Yr

def prepare_data(X):
    xt = X[-TEST:]
    xx = X[:-TEST]

    Xr,Yr = prepare_ds(xx,WIDTH)
    Xt, Yt = prepare_ds(xt,WIDTH)

    return Xr,Yr, Xt,Yt

def prepare_support(X):
    xt = X[-TEST:]
    xx = X[:-TEST]

    Xr,Yr = prepare_ds(xx,WIDTH)
    Xt, Yt = prepare_ds(xt,WIDTH)
    return Xr,Xt

def build_forecast_model_for_stream(Xtrain,Ytrain,Xtsupport):
    
    input = layers.Input(WIDTH)
    l1 = layers.Dense(20)(input)
    l2 = layers.Dense(10)(l1)

    input2 = layers.Input(WIDTH)
    l21 = layers.Dense(20)(input2)
    l22 = layers.Dense(10)(l21)

    common = layers.concatenate([l2,l22])
    interm = layers.Dense(10)(common)
    output = layers.Dense(1)(interm)

    challenger = models.Model(inputs=[input,input2],outputs=output)
    challenger.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    challenger.fit([Xtrain,Xtsupport],Ytrain,batch_size=10,epochs=20, verbose=1)
    
    return challenger

def run_championship(champion_model,challenger_model, Xtest,Ytest,Xtsup):
    yp = []
    yp = challenger_model.predict([Xtest,Xtsup])
            
    chl_perf1 = eval_agent.profit_agent(yp,Ytest)
    chl_perf2 = eval_agent.correct_agent(yp,Ytest)

    yp_chm = champion_model.predict([Xtest,Xtsup])
    chm_perf1 = eval_agent.profit_agent(yp_chm,Ytest)
    chm_perf2 = eval_agent.correct_agent(yp_chm,Ytest)
    if chl_perf2 > chm_perf2:
        print("Challenger won: ",chl_perf2, " > ", chm_perf2)
        print("Other perf: ",chl_perf1, " > ", chm_perf1)
        challenger_model.save(champion_model_path)
        print("New Champion ",datetime.datetime.today())

        return True, challenger_model
    else:
        print("Champion won: ",chm_perf2,"  vs ",chl_perf2)
    
        return False, champion_model



async def run(client):
    while True:
        try:
            today = datetime.datetime.today()
            delta = datetime.timedelta(days=1)

            yesterday = today - delta

            print("Multi: Starting: Time (t, t-1): ",today, yesterday)

            update(SYM)
            X = pd.read_csv(f"data/{SYM}.csv") 

            Xrs,Xts = prepare_support(X[SUPPORT].values)
            Xr,Yr,Xt,Yt = prepare_data(X[STREAM].values)
            
            challenger = build_forecast_model_for_stream(Xr,Yr,Xrs)
            
            
            pred_model = None
            try:
                champion = models.load_model(champion_model_path)
                change, pred_model = run_championship(champion,challenger,Xt,Yt,Xts)

                if change:
                    write_model_perf(client,METRIC_SYM+"_meta", "change",1,today)
                
                
            except IOError as e:
                # No existing champion
                challenger.save(champion_model_path)
                pred_model = challenger
                
        

            # Record predictions
            
            yp = pred_model.predict([Xt,Xts])
            
            
            pred_in_data = [np.array([X[STREAM].values[-WIDTH:]]),np.array([X[SUPPORT].values[-WIDTH:]])]
            
            tom_pred = pred_model.predict(pred_in_data)
            yest_actual = Yt[-1]

            write_model_perf(client,METRIC_SYM+"_predicted","pred",tom_pred[0][0],today)
            write_model_perf(client,METRIC_SYM+"_actual","actual",yest_actual,yesterday)

            print("Predict:",tom_pred)
            print("Pred. Longer:",yp[-1])
            print("Actual Longer:",yest_actual)

        except Exception as e:
            print("=== Error ===")
            print(e)

        if ENABLE_TIMER:
            await aio.sleep(WAIT_INTERVAL)

if __name__ == "__main__":

    client = influxdb_client.InfluxDBClient(url=HOST, token=TOKEN,org=ORG, debug=True)
    
    loop = aio.get_event_loop()
    loop.run_until_complete(run(client))








