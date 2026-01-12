import re
import pandas as pd

# Load the dataset
df = pd.read_csv("D:/misc/langs/sentiment_analyzer/data/processed/combined_data.csv")

# Define a cleaning function
def clean_text(text):
    text = re.sub(r"#\S*\s", "", text)  # Remove hashtags
    text = re.sub(r"\W+", " ", text)  # Remove non-word characters (punctuation etc.)
    text = re.sub(r"@\S*\s", "", text)  # Remove mentions
    text = re.sub(r"http\S*\s", "", text)  # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    return text.lower()

# Apply cleaning to the 'text' column
df['text'] = df['text'].astype(str).apply(clean_text)
# Save the cleaned dataset
df.to_csv("D:/misc/langs/sentiment_analyzer/data/processed/cleaned_data.csv", index=False)