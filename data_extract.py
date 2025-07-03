from datetime import datetime, timedelta, timezone
import praw
import os
from dotenv import load_dotenv
import re
from collections import Counter
import time

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret"),
    user_agent=os.getenv("user_agent"),
    username=os.getenv("user"),
    password=os.getenv("password"),
)

subreddit = reddit.subreddit("wallstreetbets")

data = [] 


for day_offset in range(30, 0, -1):
    start_dt = datetime.now(timezone.utc) - timedelta(days=day_offset)
    start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = start_dt + timedelta(days=1)

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    print(f"\nScraping for {start_dt.date()}")

    for submission in subreddit.new(limit=1000):
        created_ts = int(submission.created_utc)
        if start_ts <= created_ts < end_ts:
            post = {
                'date': datetime.fromtimestamp(created_ts),
                'title': submission.title,
                'text': submission.selftext,
                'upvotes': submission.score,
                'flair': submission.link_flair_text,
            }
            tickers = re.findall(r"\$[A-Z]{1,5}", post['text']) + re.findall(r"\$[A-Z]{1,5}", post['title'])
            post['tickers'] = [ticker[1:] for ticker in tickers]
            data.append(post)

    time.sleep(1)  

print(f"\nTotal posts across 30 days: {len(data)}")
print(f"Posts with tickers: {len([d for d in data if d['tickers']])}")

ticker_counts = Counter(ticker for post in data for ticker in post['tickers'])
out_data = ticker_counts.most_common(30)  

print("\nTop 30 Tickers over 30 Days:")
print(out_data)
