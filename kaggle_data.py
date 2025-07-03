import os
os.environ['KAGGLE_USERNAME'] = 'devavratapatel'
os.environ['KAGGLE_KEY'] = '0d895db5cc0edab4e45017a5bd9b9bc3'

import kagglehub
import pandas as pd
import re
from datetime import datetime, timedelta
from collections import Counter
import os


path = kagglehub.dataset_download("gpreda/reddit-wallstreetsbets-posts")
print("✅ Path to dataset files:", path)


csv_file = os.path.join(path, "reddit_wsb.csv")


df = pd.read_csv(csv_file)
print("✅ Columns:", df.columns.tolist())

df = df.rename(columns={
    'created_utc': 'date',
    'selftext': 'text',
    'link_flair_text': 'flair',
    'score': 'upvotes'
})
df['date'] = pd.to_datetime(df['timestamp'])


data_kaggle = []

for _, row in df.iterrows():
    post = {
        'date': row['date'],
        'title': str(row.get('title', '')),
        'text': str(row.get('text', '')),
        'upvotes': row.get('upvotes', 0),
        'flair': row.get('flair', None)
    }
    tickers = re.findall(r"\$[A-Z]{1,5}", post['text']) + re.findall(r"\$[A-Z]{1,5}", post['title'])
    post['tickers'] = [ticker[1:] for ticker in tickers]
    data_kaggle.append(post)

print(f"\n✅ Total posts in last 30 days: {len(data_kaggle)}")
print(f"✅ Posts with tickers: {len([d for d in data_kaggle if d['tickers']])}")

ticker_counts = Counter(ticker for post in data_kaggle for ticker in post['tickers'])
out_data_kaggle = ticker_counts.most_common(30)

print("\n📈 Top 30 Tickers over last 30 Days:")
print(out_data_kaggle)
