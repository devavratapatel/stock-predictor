import yfinance as yf

from data_extract import out_data, data
from kaggle_data import out_data_kaggle, data_kaggle
from datetime import datetime, timedelta

out_data = out_data + out_data_kaggle
data = data + data_kaggle

tickers = [(d, count) for (d, count) in out_data]

input_data = []

def get_next_day_return(tickers, post_date):
    if isinstance(tickers, str):
        tickers = [tickers]
    elif not isinstance(tickers, list):
        print(f"Invalid tickers input: {tickers}")
        return None
    if not tickers:
        return None
    
    try:

        post_dt = post_date if isinstance(post_date, datetime) else  datetime.strptime(post_date, "%Y-%m-%d")
        while post_dt.weekday() >= 5: 
            post_dt += timedelta(days=1)
        start_day = post_dt.strftime("%Y-%m-%d")
        end_dt = post_dt + timedelta(days=2)
        end_day = end_dt.strftime("%Y-%m-%d")
        if end_dt >= datetime.today():
            print(f"Skipping future date: {end_day}")
            return None
        returns = []

        for ticker in tickers:
            try:
                dat = yf.download(ticker, start=start_day, end=end_day, auto_adjust=False, progress=False)
                if dat.empty or len(dat) < 2:
                    print(f"No data for {ticker} between {start_day} and {end_day}")
                    continue
                close_0 = dat['Close'].iloc[0]
                close_1 = dat['Close'].iloc[1]
                pct_change = (float(close_1) - float(close_0)) / float(close_0)
                returns.append(pct_change)
            except Exception as e:
                print(f"Error processing ticker {ticker}: {e}")
                continue
        
        if not returns:
            return None
        
        return sum(returns) / len(returns)
            
        
    except Exception as e:
        print(f"Error processing tickers {tickers} on {post_date} : {e}")
        return None

for post in data:
    
    ret = get_next_day_return(post["tickers"], post["date"])
    
    if ret is not None:
        post["next_day_return"] = ret
        post["label"] = 1 if ret > 0 else 0
    else:
        post["next_day_return"] = None
        post["label"] = None

import pandas as pd
import numpy as np
from textblob import TextBlob

df = pd.DataFrame(data)
df["full_text"] = df["title"] + " " + df["text"]
df["log_upvotes"] = np.log1p(df["upvotes"])

df["sentiment"] = df["full_text"].apply(lambda x: TextBlob(x).sentiment.polarity)

df = df.dropna(subset=["next_day_return"])

pd.set_option("display.max_rows", None)        
pd.set_option("display.max_columns", None)     
pd.set_option("display.width", None)           
pd.set_option("display.max_colwidth", None)  

df.to_csv("labeled_data.csv", index=False)
print("Data saved to labeled_data.csv")
