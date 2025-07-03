import torch
from torch.utils.data import Dataset, DataLoader
import praw
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from model import LSTMClassifier
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret"),
    user_agent=os.getenv("user_agent"),
    username=os.getenv("user"),
    password=os.getenv("password"),
)
subreddit = reddit.subreddit("wallstreetbets")

posts = []

start_dt = datetime.now(timezone.utc) - timedelta(days=2)
start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
end_dt = start_dt + timedelta(days=1)

start_ts = int(start_dt.timestamp())
end_ts = int(end_dt.timestamp())

for submission in subreddit.new(limit=100):
    created_ts = int(submission.created_utc)
    if start_ts <= created_ts < end_ts:
        data = {
            'date': datetime.fromtimestamp(created_ts),
            'title': submission.title,
            'text': submission.selftext,
            'upvotes': submission.score,
            'flair': submission.link_flair_text,
        }
        tickers = re.findall(r"\$[A-Z]{1,5}", data['text']) + re.findall(r"\$[A-Z]{1,5}", data['title'])
        data['tickers'] = [ticker[1:] for ticker in tickers]
        posts.append(data)

df = pd.DataFrame(posts)
df["full_text"] = df["title"] + " " + df["text"]
df["log_upvotes"] = np.log1p(df["upvotes"])
df["full_text"] = df["full_text"].fillna("")
df["sentiment"] = df["full_text"].apply(lambda x: TextBlob(x).sentiment.polarity)

df.to_csv("predict_data.csv",index=False)

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
import pandas as pd
from collections import Counter

df = pd.read_csv("C:/Users/Devavrata/proproject/predict_data.csv")

import pickle
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode_sentence(sentence,vocab,max_len=20):
    tokens = word_tokenize(sentence.lower())
    tokens = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    tokens = tokens[:max_len]
    tokens += [0]*(max_len-len(tokens))
    return tokens

df["encoded"] = df["full_text"].apply(lambda x: encode_sentence(x,vocab))

class PredictionDataset(Dataset):
    def __init__(self, df):
        self.x = list(df["encoded"])
        self.upvotes = torch.tensor(df["log_upvotes"].values, dtype=torch.float32)
        self.sentiment = torch.tensor(df["sentiment"].values, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.upvotes[idx]), torch.tensor(self.sentiment[idx])

ds = PredictionDataset(df)
dataloader = DataLoader(ds, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(vocab_size=max(vocab.values())+1, embed_dim=128,hidden_dim=64).to(device)

model.load_state_dict(torch.load("model_epoch_50.pt", map_location=device))

def predict(dataloader):
    all_preds = []
    model.eval()
    raw_outputs = []
    with torch.no_grad():
        for x, upv, sent in dataloader:
            x, upv, sent = x.to(device), upv.to(device), sent.to(device)
            outputs = model(x, upv, sent)
            # preds = (output).long().cpu().numpy()
            raw_outputs.extend(outputs.cpu().numpy())
    print("\nRaw prediction samples:", raw_outputs[:5]) 
    return raw_outputs

preds = predict(dataloader)
df["prediction_prob"] = preds

print("\nPrediction statistics:")
print(f"Min: {df['prediction_prob'].min():.4f}")
print(f"Mean: {df['prediction_prob'].mean():.4f}")
print(f"Max: {df['prediction_prob'].max():.4f}")
print(f"Std: {df['prediction_prob'].std():.4f}")


if df['prediction_prob'].nunique() == 1:
    print("\nWARNING: All predictions are identical!")
from collections import defaultdict
import matplotlib.pyplot as plt

df["tickers"] = df["tickers"].apply(lambda x: eval(x) if isinstance(x, str) else[])
df["impact"] = (
    (df["prediction_prob"] - 0.5) *  
    np.log1p(df["upvotes"]) *  
    (1 + 3*df["sentiment"].clip(-0.3, 0.3))  
)
if not df["impact"].empty:
    max_impact = df["impact"].abs().max() or 1 
    df["normalized_impact"] = 0.5 + 0.5 * (df["impact"] / max_impact)
ticker_preds = defaultdict(list)

ticker_impact = defaultdict(list)
for _, row in df.iterrows():
    for ticker in row["tickers"]:

        impact = row["prediction_prob"] * np.exp(row["log_upvotes"]/10) * (1+row["sentiment"])
        ticker_impact[ticker].append(impact)

ticker_avg_impact = {t: np.mean(impacts) for t, impacts in ticker_impact.items()}

if ticker_avg_impact:
    max_impact = max(abs(x) for x in ticker_avg_impact.values()) or 1  
    ticker_predictions = {
        ticker: 0.5 + 0.5 * (impact / max_impact)
        for ticker, impact in ticker_avg_impact.items()
    }

ticker_trends = {
    ticker: "UP" if sum(preds) > len(preds) / 2 else "DOWN"
    for ticker, preds in ticker_preds.items()
}

def parse_tickers(ticker_str):
    if isinstance(ticker_str, str):
        try:
            return eval(ticker_str)  
        except:
            return []
    elif isinstance(ticker_str, list):
        return ticker_str
    return []

df["tickers"] = df["tickers"].apply(parse_tickers)

predictions_df = pd.DataFrame(list(ticker_predictions.items()), columns=['Ticker', 'Probability'])
predictions_df['Direction'] = predictions_df['Probability'].apply(lambda x: 'UP' if x >= 0.5 else 'DOWN')

predictions_df = predictions_df.sort_values('Probability', ascending=False)

print("\nComplete Ticker Predictions:")
print(predictions_df.to_string(index=False, float_format="%.4f"))

if ticker_predictions:
    
    all_predictions = pd.DataFrame.from_dict(ticker_predictions, 
                                          orient='index',
                                          columns=['Probability'])
    
    
    all_predictions['Direction'] = all_predictions['Probability'].apply(
        lambda x: 'UP' if x >= 0.5 else 'DOWN'
    )
    
    
    all_predictions['Confidence'] = (all_predictions['Probability'] - 0.5).abs()
    all_predictions = all_predictions.sort_values('Confidence', ascending=False)
    
    print("\nCOMPLETE TICKER PREDICTIONS:")
    print(all_predictions[['Probability', 'Direction']].to_string(
        float_format="%.4f",
        formatters={'Probability': '{:,.4f}'.format}
    ))
    
    
    plt.figure(figsize=(14, 8))
    top_predictions = all_predictions.head(20) 

    colors = ['green' if p >= 0.5 else 'red' 
             for p in top_predictions['Probability']]
    bars = plt.bar(top_predictions.index, 
                 top_predictions['Probability'], 
                 color=colors)
    
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                f"{height:.4f}",
                ha='center', 
                va='bottom' if height >= 0.5 else 'top',
                fontsize=9)
    
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.title("Top 20 Most Confident Predictions", pad=20)
    plt.ylabel("Prediction Probability")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.show()
    
    
    all_predictions.to_csv('all_ticker_predictions.csv', float_format='%.4f')
    print("\nPredictions saved to 'all_ticker_predictions.csv'")
else:
    print("No ticker predictions available")

print("\nSample of tickers and predictions:")
for ticker, prob in list(ticker_predictions.items())[:5]:
    print(f"{ticker}: {prob:.4f}")