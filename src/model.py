import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig
import pickle
import os

class SentimentDataset(Dataset):
    """
    Custom Dataset class for sentiment analysis.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class BertSentimentClassifier(nn.Module):
    """
    BERT-based sentiment classifier with LSTM layer.
    """
    def __init__(self, num_classes=3, hidden_size=256, num_layers=2, dropout_rate=0.1, model_name='bert-base-uncased'):
        super(BertSentimentClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_config = AutoConfig.from_pretrained(model_name)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.bert_config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for classification
        """
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the last hidden state
        sequence_output = bert_outputs.last_hidden_state
        
        # Pass through LSTM
        lstm_output, (hidden, cell) = self.lstm(sequence_output)
        
        # Use the last output of LSTM
        # For bidirectional LSTM, concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Classification
        logits = self.classifier(hidden)
        
        return logits
    
    def predict_proba(self, input_ids, attention_mask):
        """
        Get prediction probabilities.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Probabilities for each class
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities
    
    def save_model(self, save_path):
        """
        Save the model state dict.
        
        Args:
            save_path: Path to save the model
        """
        # Save model state dict and configuration
        save_dict = {
            'state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'bert_config': self.bert_config
        }
        
        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, model_path, device='cpu'):
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded model
        """
        # Load the saved dictionary
        save_dict = torch.load(model_path, map_location=device)
        
        # Create model instance
        model = cls(
            num_classes=save_dict['num_classes'],
            model_name=save_dict['model_name']
        )
        
        # Load state dict
        model.load_state_dict(save_dict['state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        return model

class SentimentPredictor:
    """
    Wrapper class for making predictions with the trained model.
    """
    def __init__(self, model_path, label_encoder_path, tokenizer, device='cpu'):
        self.device = device
        self.tokenizer = tokenizer
        
        # Load model
        self.model = BertSentimentClassifier.load_model(model_path, device)
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
    
    def predict(self, text, max_length=128):
        """
        Predict sentiment for a single text.
        
        Args:
            text: Text to analyze
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with prediction results
        """
        from data_preprocessing import prepare_single_text
        
        # Prepare text
        encoding = prepare_single_text(text, self.tokenizer, max_length)
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            probabilities = self.model.predict_proba(input_ids, attention_mask)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Convert back to label
        predicted_sentiment = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Get all class probabilities
        all_probabilities = {}
        for i, label in enumerate(self.label_encoder.classes_):
            all_probabilities[label] = probabilities[0][i].item()
        
        return {
            'text': text,
            'predicted_sentiment': predicted_sentiment,
            'confidence': confidence,
            'all_probabilities': all_probabilities
        }
    
    def predict_batch(self, texts, max_length=128, batch_size=32):
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                result = self.predict(text, max_length)
                results.append(result)
        
        return results