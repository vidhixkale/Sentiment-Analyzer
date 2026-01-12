import pandas as pd

# Load datasets
tweets_df = pd.read_csv('./data/raw/tweets.csv')
news_df = pd.read_csv('./data/raw/news.csv')

# Standardize columns (keep only text + sentiment)
tweets_clean = tweets_df[['text', 'sentiment']]
news_clean = news_df[['text', 'sentiment']]

# Merge vertically
combined_df = pd.concat([tweets_clean, news_clean], ignore_index=True)
to_csv_path = './data/processed/combined_data.csv'
# Save to CSV
combined_df.to_csv(to_csv_path, index=False)
# Verify (optional)
print(f"Total entries: {len(combined_df)}")
print(combined_df.head())