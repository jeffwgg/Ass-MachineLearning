# Emotion Classification Streamlit App

> **Quick Start for Teammates:**
> ```bash
> # 1. Check if you have Python 3.9+
> python3 --version
> # If you see Python 3.9+ or higher, skip to step 3
> 
> # 2. If no Python or version < 3.9, install from:
> # https://python.org/downloads/
> 
> # 3. Create virtual environment
> python3 -m venv .venv
> # Windows: .venv\Scripts\activate
> # macOS/Linux: source .venv/bin/activate
> pip install -r requirements.txt
> streamlit run app.py
> ```assification Streamlit App


This Streamlit web application provides an interactive interface for emotion classification using three pre-trained machine learning models: XGBoost, SVM, and Logistic Regression.

## Features

- **Model Selection**: Choose between XGBoost, SVM, or Logistic Regression models
- **Real-time Prediction**: Enter text and get instant emotion classification
- **Probability Visualization**: Interactive bar charts showing confidence scores for all emotions
- **Top-3 Predictions**: Display the three most likely emotions with percentages
- **Detailed Analysis**: Expandable sections showing raw probabilities and preprocessed text

## Emotion Classes

The app classifies text into six emotions:
- **0**: Sadness 😢
- **1**: Joy 😊
- **2**: Love ❤️
- **3**: Anger 😠
- **4**: Fear 😨
- **5**: Surprise 😲

## Setup and Installation

### Prerequisites
- **Python 3.9+** must be installed on your system
  - **Check first**: Run `python3 --version` in terminal
  - **Download if needed**: https://python.org/downloads/
  - Verify installation: `python3 --version`

### Requirements
- **Python**: 3.9+ (tested with 3.9.6)
- **Model files**: Must be present in `trained model/` directory

### Installation Steps

1. **Check Python version**:
   ```bash
   python3 --version
   ```
   - If you have Python 3.9+ or higher, proceed to step 2
   - If you don't have Python or version < 3.9, install from: https://python.org/downloads/

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the virtual environment**:
   ```bash
   # macOS/Linux
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**: `http://localhost:8501`

### Required Model Files
```
trained model/
├── xgb_best_model.pkl          # XGBoost classifier
├── svm_best_model.pkl          # SVM classifier  
├── logreg_best_model.pkl       # Logistic Regression classifier
└── tfidf_vectorizer.joblib     # TF-IDF vectorizer (80,000 features)
```

## How to Use

1. **Select a Model**: Use the sidebar dropdown to choose XGBoost, SVM, or Logistic Regression
2. **Enter Text**: Type or paste text in the main input area
3. **Predict**: Click "Predict Emotion" to analyze the text
4. **View Results**: 
   - See the predicted emotion with confidence score
   - Examine the probability distribution chart
   - Check the top-3 predictions
   - Expand sections for detailed analysis

## Model Information

### Architecture
- **Model-only approach**: Trained classifiers saved separately from preprocessing
- **Shared TF-IDF vectorizer**: All models use the same feature extraction (80,000 features)
- **Prediction pipeline**: Text preprocessing → TF-IDF → Model → Results

### Model Performance
| Model | Accuracy | Macro F1 | Description |
|-------|----------|----------|-------------|
| **XGBoost** | ~94.4% | ~93.0% | Gradient boosting with hyperparameter tuning |
| **SVM** | - | - | Linear SVM with balanced class weights |
| **Logistic Regression** | - | - | Fast linear classifier with regularization |

### TF-IDF Configuration
- **Max features**: 80,000
- **N-gram range**: (1, 2) - unigrams and bigrams  
- **Min document frequency**: 2
- **Sublinear TF**: True

## Text Preprocessing

The app applies these preprocessing steps to match the training pipeline:
- Convert to lowercase
- Remove URLs and mentions (@username)
- Extract hashtag words (remove # symbol)
- Replace numbers with `<num>` token
- Remove punctuation (except ! and ?)
- Normalize whitespace

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **"Model file not found"** | Ensure all files in `trained model/` directory are present |
| **"Vectorizer file not found"** | Verify `tfidf_vectorizer.joblib` exists in `trained model/` |
| **Import errors** | Activate virtual environment and run `pip install -r requirements.txt` |
| **Version warnings** | These are normal and won't affect functionality |
| **App won't start** | Check Python version (3.9+) and virtual environment activation |

### Getting Help
1. Check terminal output for specific error messages
2. Verify all model files are present and correctly named
3. Ensure virtual environment is activated
4. Contact the development team if issues persist

## Dependencies

See `requirements.txt` for the complete list of required packages:
- `streamlit` - Web interface
- `scikit-learn` - Machine learning models
- `xgboost` - XGBoost classifier
- `joblib` - Model serialization
- `numpy`, `pandas` - Data handling
- `plotly` - Interactive visualizations
- `scipy` - Scientific computing
