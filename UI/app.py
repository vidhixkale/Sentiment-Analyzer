from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import sys
import torch
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import json
import logging
from transformers import RobertaTokenizer
import pickle
import re

# Add the src directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from model import SentimentPredictor, BertSentimentClassifier
    from data_preprocessing import clean_text, prepare_single_text
    from train_sentiment import RobertaSentimentClassifier
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the src directory contains the required modules")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model components
model = None
tokenizer = None
label_encoder = None
device = None

def load_model():
    """Load the trained sentiment analysis model"""
    global model, tokenizer, label_encoder, device

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

        # Load tokenizer (use the correct tokenizer directory or pretrained name)
        tokenizer = RobertaTokenizer.from_pretrained(os.path.join(model_dir, 'tokenizer'))

        # Load label encoder
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)

        num_classes = len(label_encoder.classes_)
        model = RobertaSentimentClassifier(
            num_classes=num_classes,
            hidden_size=768,
            dropout_rate=0.1
        )
        model_path = os.path.join(model_dir, 'roberta_sentiment_classifier.pt')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        logger.info("Model loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def predict_sentiment(text, max_length=128):
    """Predict sentiment for a single text"""
    global model, tokenizer, label_encoder, device
    
    if not all([model, tokenizer, label_encoder]):
        return None
    
    try:
        # Clean and prepare text
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return None
        
        # Tokenize
        encoding = tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Convert back to label
        predicted_sentiment = label_encoder.inverse_transform([predicted_class])[0]
        
        # Get all class probabilities
        all_probabilities = {}
        for i, label in enumerate(label_encoder.classes_):
            all_probabilities[label] = probabilities[0][i].item()
        
        return {
            'text': text,
            'predicted_sentiment': predicted_sentiment,
            'confidence': confidence,
            'all_probabilities': all_probabilities
        }
        
    except Exception as e:
        logger.error(f"Error predicting sentiment: {e}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'txt'}

def process_file(file_path):
    """Process uploaded file and analyze sentiment"""
    results = []
    
    try:
        if file_path.endswith('.csv'):
            # Process CSV file
            df = pd.read_csv(file_path)
            
            # Try to find text column
            text_column = None
            for col in ['text', 'review', 'comment', 'message', 'content']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                # Use the first column that contains strings
                for col in df.columns:
                    if df[col].dtype == 'object':
                        text_column = col
                        break
            
            if text_column:
                texts = df[text_column].dropna().tolist()
            else:
                return []
                
        else:  # .txt file
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Analyze each text
        for text in texts[:1000]:  # Limit to 1000 entries for performance
            if text and len(text.strip()) > 0:
                result = predict_sentiment(text)
                if result:
                    results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return []

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze single text input"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        result = predict_sentiment(text)
        
        if result:
            return jsonify({
                'success': True,
                'result': result
            })
        else:
            return jsonify({'error': 'Failed to analyze text'}), 500
            
    except Exception as e:
        logger.error(f"Error in analyze_text: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and batch analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV and TXT files are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process file
        results = process_file(file_path)
        
        # Clean up file
        os.remove(file_path)
        
        if not results:
            return jsonify({'error': 'No valid text found in file or processing failed'}), 400
        
        # Calculate statistics
        sentiment_counts = {}
        for result in results:
            sentiment = result['predicted_sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        return jsonify({
            'success': True,
            'results': results,
            'statistics': sentiment_counts,
            'total_analyzed': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_loaded = all([model, tokenizer, label_encoder])
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'device': str(device) if device else None
    })

if __name__ == '__main__':
    # Load model on startup
    model_loaded = load_model()
    
    if not model_loaded:
        logger.warning("Model not loaded. Please check model files in the models directory.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)