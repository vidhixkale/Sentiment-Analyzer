import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import pickle
import pandas as pd
from torch import nn
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Enhanced text cleaning with contraction handling"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # URLs
    text = re.sub(r'\@\w+|\#\w+', '', text)  # Mentions/hashtags
    text = re.sub(r'[^\w\s\-\']', '', text)  # Keep basic punctuation
    
    # Contraction expansion
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", 
        "'ll": " will", "'ve": " have", "'m": " am"
    }
    for cont, expanded in contractions.items():
        text = text.replace(cont, expanded)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class RobertaSentimentClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=768, num_layers=1, dropout_rate=0.1, model_name='roberta-base'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size*4,
                dropout=dropout_rate
            ),
            num_layers=num_layers
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)
        sequence = outputs.last_hidden_state
        
        # Transformer processing
        transformer_out = self.transformer(sequence)
        
        # Attention pooling
        attn_weights = self.attention(transformer_out)
        context = torch.sum(attn_weights * transformer_out, dim=1)
        
        return self.classifier(self.dropout(context))
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load_model(cls, path, num_classes, **kwargs):
        model = cls(num_classes=num_classes, **kwargs)
        model.load_state_dict(torch.load(path))
        return model

def prepare_data(data_path, max_length=128, test_size=0.2, model_name='roberta-base', device='cuda'):
    """Load and prepare dataset with class weights"""
    df = pd.read_csv(data_path)
    df['text'] = df['text'].apply(clean_text)
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['sentiment'])
    
    # Class distribution analysis
    logger.info("\nClass Distribution:")
    class_dist = pd.Series(labels).value_counts(normalize=True)
    logger.info(class_dist)
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'], labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(
        train_texts.tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    test_encodings = tokenizer(
        test_texts.tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    logger.info(f"Class weights: {class_weights}")
    
    return train_encodings, test_encodings, train_labels, test_labels, tokenizer, label_encoder, class_weights

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, class_weights, num_epochs=5, patience=3):
    """Training with class-weighted loss"""
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_val_accuracy = 0
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss, predictions, true_labels = 0, [], []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        avg_val_loss = val_loss / len(val_dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(accuracy)
        
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Val Accuracy: {accuracy:.4f}")
        
        # Early stopping
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                model.load_state_dict(best_model)
                break
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Optimized RoBERTa Sentiment Analysis")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)  # Reduced LR
    parser.add_argument("--hidden_size", type=int, default=768)  # Match RoBERTa
    parser.add_argument("--dropout_rate", type=float, default=0.1)  # Reduced dropout
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare data with class weights
    train_encodings, test_encodings, train_labels, test_labels, tokenizer, label_encoder, class_weights = prepare_data(
        args.data_path,
        max_length=args.max_length
    )

    # Datasets
    train_dataset = SentimentDataset(train_encodings, train_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model
    model = RobertaSentimentClassifier(
        num_classes=len(label_encoder.classes_),
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout_rate
    ).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * len(train_dataloader) * args.num_epochs),
        num_training_steps=len(train_dataloader) * args.num_epochs
    )

    # Train
    model, history = train_model(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        device,
        class_weights,
        num_epochs=args.num_epochs
    )

    # Save
    os.makedirs(args.model_dir, exist_ok=True)
    model.save_model(os.path.join(args.model_dir, 'model.pt'))
    tokenizer.save_pretrained(args.model_dir)
    with open(os.path.join(args.model_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Accuracy')
    plt.title('Validation Accuracy')
    plt.savefig(os.path.join(args.model_dir, 'training.png'))
    plt.close()

if __name__ == "__main__":
    main()