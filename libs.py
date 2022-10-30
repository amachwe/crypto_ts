import pandas as pd


def build_rsi(df,cols,rsi_len=14):
    rsi = []
    date = []
    
    
    delta = df[cols].diff().dropna()

    for i in range(rsi_len,len(delta)):
        sum_pos = 0
        c_pos = 0

        sum_neg = 0
        c_neg = 0
        val = []
        for c in cols:
            for s in delta[c][i-rsi_len:i]:
                if s > 0:
                    sum_pos += s
                    c_pos += 1
                else:
                    sum_neg += s
                    c_neg += 1
                
                if c_pos > 0 and c_neg > 0 and sum_neg != 0:
                    avg_pos = sum_pos/c_pos
                    avg_neg = sum_neg/c_neg
        
            val.append(100 - (100/(1+(avg_pos/abs(avg_neg)))))

        rsi.append(val)

        date.append(delta.index[i])

        
        
    return pd.DataFrame(rsi,columns=cols,index=date)
