# Sentiment Analyzer

A web-based sentiment analysis tool built with Flask and PyTorch, using a RoBERTa-based model for advanced sentiment classification. The app allows users to analyze the sentiment of single texts or batch files (CSV/TXT) and visualize results interactively.

---

## Features

- **Single Text Analysis:** Enter any text and get instant sentiment prediction with confidence scores.
- **Batch File Analysis:** Upload CSV or TXT files for bulk sentiment analysis.
- **Interactive Visualizations:** Pie and bar charts, statistics cards, and detailed results table.
- **Modern UI:** Responsive design with Bootstrap and FontAwesome.
- **Model:** Uses a fine-tuned RoBERTa model for robust sentiment classification.

---

## Folder Structure

```
sentiment_analyzer/
│
├── UI/
│   ├── app.py                # Flask web app
│   ├── static/               # JS, CSS, images
│   │   ├── main.js
│   │   └── style.css
│   ├── templates/
│   │   └── index.html        # Main HTML template
│   └── uploads/              # Uploaded files
│
├── src/
│   ├── data_preprocessing.py # Text cleaning and preprocessing
│   ├── model.py              # Model classes and utilities
│   ├── train_sentiment.py    # Training script
│   └── models/
│       ├── roberta_sentiment_classifier.pt # Trained model weights
│       ├── tokenizer/        # Tokenizer files
│       └── label_encoder.pkl # Label encoder
│
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/erratus/sentiment_analyzer
cd sentiment_analyzer
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv env
env\Scripts\activate  # On Windows
# source env/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Model Files

Ensure the following files exist in `src/models/`:
- `roberta_sentiment_classifier.pt`
- `tokenizer/` (directory with tokenizer files)
- `label_encoder.pkl`

If you need to train your own model, run:
```bash
python src/train_sentiment.py --model_dir src/models/
```

### 5. Run the Web App

```bash
cd UI
python app.py
```

The app will be available at [http://localhost:5000](http://localhost:5000).

---

## Usage

- **Single Analysis:** Enter text in the input box and click "Analyze Sentiment".
- **Batch Analysis:** Upload a `.csv` or `.txt` file and click "Upload & Analyze File".
- **Results:** View sentiment, confidence, and probability breakdowns. Batch results include charts and tables.

---

## Troubleshooting

- **Model Not Loaded:** Ensure all model files are present in `src/models/` and match the expected architecture.
- **File Upload Issues:** Only `.csv` and `.txt` files are supported, max size 16MB.
- **Dependencies:** If you encounter import errors, double-check your Python environment and installed packages.

---

## Credits

- Built with [Flask](https://flask.palletsprojects.com/), [PyTorch](https://pytorch.org/), and [Transformers](https://huggingface.co/transformers/).
- UI powered by [Bootstrap](https://getbootstrap.com/) and [Chart.js](https://www.chartjs.org/).
