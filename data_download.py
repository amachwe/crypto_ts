import tslibs.data_client as dc
import pandas as pd



def to_csv(symbol):
    
    stream = dc.get_data_single(symbol)[0]

    size_ = len(stream["open"])
    size_p = len(stream["price"])
    df = pd.DataFrame()
    for c in ["open","close","high","low","volume"]:
        df[c] = stream[c]
    df["time"] = stream["time_"+c]
    
    df["seq"] = [i for i in range(0, size_)]

    dfp = pd.DataFrame()

    dfp["price"] = stream["price"]
    dfp["time"] = stream["time_price"]
    dfp["seq"] = [i for i in range(0,size_p)]

    return df,dfp

    









