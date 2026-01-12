import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def clean_text(text, minimal=True):
    """
    Clean and preprocess text data for sentiment analysis.
    
    Args:
        text (str): Raw text to clean
        minimal (bool): If True, apply minimal preprocessing suitable for BERT
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    if minimal:
        # Minimal preprocessing for BERT - just basic cleaning
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Basic contraction fixes (optional)
        contractions = {
            r'\bi\s+d\b': "i'd", r'\bi\s+ll\b': "i'll", r'\bi\s+m\b': "i'm",
            r'\bi\s+ve\b': "i've", r'\bi\s+re\b': "i're", r'\byou\s+re\b': "you're",
            r'\bdon\s+t\b': "don't", r'\bcan\s+t\b': "can't", r'\bwon\s+t\b': "won't",
            r'\bisn\s+t\b': "isn't", r'\baren\s+t\b': "aren't", r'\bwasn\s+t\b': "wasn't",
            r'\bweren\s+t\b': "weren't", r'\bhasn\s+t\b': "hasn't", r'\bhaven\s+t\b': "haven't",
            r'\bhadn\s+t\b': "hadn't", r'\bwouldn\s+t\b': "wouldn't", r'\bshouldn\s+t\b': "shouldn't",
            r'\bcouldn\s+t\b': "couldn't", r'\bmightn\s+t\b': "mightn't"
        }
        
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    else:
        # More aggressive preprocessing (use if minimal doesn't work well)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags for social media text
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation that might be important for sentiment
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

def encode_labels(labels):
    """
    Encode sentiment labels to numerical values.
    
    Args:
        labels: List or array of sentiment labels
        
    Returns:
        encoded_labels: Numerical labels
        label_encoder: LabelEncoder object for inverse transformation
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    return encoded_labels, label_encoder

def prepare_data(data_path, test_size=0.2, max_length=128, model_name='bert-base-uncased'):
    """
    Prepare sentiment analysis data for training.
    
    Args:
        data_path (str): Path to the CSV file with sentiment,text columns
        test_size (float): Proportion of data to use for testing
        max_length (int): Maximum sequence length for tokenization
        model_name (str): Name of the BERT model to use
        
    Returns:
        train_encodings: Tokenized training data
        test_encodings: Tokenized test data
        train_labels: Training labels
        test_labels: Test labels
        tokenizer: BERT tokenizer
        label_encoder: Label encoder for sentiment classes
    """
    
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Check if required columns exist
    if 'sentiment' not in df.columns or 'text' not in df.columns:
        raise ValueError("Dataset must contain 'sentiment' and 'text' columns")
    
    # Remove rows with missing values
    df = df.dropna(subset=['sentiment', 'text'])
    
    print(f"Dataset shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Clean text data
    print("Cleaning text data...")
    df['text'] = df['text'].apply(clean_text)
    
    # Remove empty texts after cleaning
    df = df[df['text'].str.len() > 0]
    
    # Encode labels
    encoded_labels, label_encoder = encode_labels(df['sentiment'])
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(),
        encoded_labels,
        test_size=test_size,
        random_state=42,
        stratify=encoded_labels
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize texts
    print("Tokenizing texts...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Convert labels to tensors
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    return train_encodings, test_encodings, train_labels, test_labels, tokenizer, label_encoder

def prepare_single_text(text, tokenizer, max_length=128):
    """
    Prepare a single text for prediction.
    
    Args:
        text (str): Text to analyze
        tokenizer: BERT tokenizer
        max_length (int): Maximum sequence length
        
    Returns:
        Dictionary with tokenized text
    """
    cleaned_text = clean_text(text)
    
    encoding = tokenizer(
        cleaned_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    return encoding