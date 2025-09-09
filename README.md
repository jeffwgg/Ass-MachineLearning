# Emotion Classification Streamlit App

This Streamlit web application provides an interactive interface for emotion classification using three pre-trained machine learning models: XGBoost, SVM, and Logistic Regression.

## Features

- **Model Selection**: Choose between XGBoost, SVM, or Logistic Regression models
- **Real-time Prediction**: Enter text and get instant emotion classification
- **Probability Visualization**: Interactive bar charts showing confidence scores for all emotions
- **Top-3 Predictions**: Display the three most likely emotions with percentages
- **Detailed Analysis**: Expandable sections showing raw probabilities and preprocessed text

## Emotion Classes

The app classifies text into six emotions:
- **0**: Sadness
- **1**: Joy
- **2**: Love
- **3**: Anger
- **4**: Fear
- **5**: Surprise

## Setup and Installation

0. **Activate Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model Files**:
   Make sure the following trained model files are present in the `trained model/` directory:
   - `xgb_best_pipeline.pkl`
   - `svm_best_pipeline.pkl`
   - `logreg_best_pipeline.pkl`

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the App**:
   Open your web browser and navigate to `http://localhost:8501`

## Usage

1. **Select a Model**: Use the sidebar dropdown to choose which model to use for predictions
2. **Enter Text**: Type or paste text in the main input area
3. **Predict**: Click the "Predict Emotion" button to analyze the text
4. **View Results**: 
   - See the top predicted emotion with confidence score
   - Examine the probability distribution chart
   - Check the top-3 predictions
   - Expand detailed sections for more information
5. **Clear**: Use the "Clear" button to reset the input

## Model Information

- **XGBoost**: Gradient Boosting classifier with hyperparameter tuning ✅ **Working** (Retrained with XGBoost 2.0.3)
- **SVM**: Support Vector Machine with linear kernel (✅ **Working**)
- **Logistic Regression**: Linear classifier with regularization (✅ **Working**)

## Current Status

✅ **All models working perfectly!**
- XGBoost: 96.16% accuracy, 95.65% macro F1 (retrained with XGBoost 2.0.3)
- SVM: Working with decision function probabilities
- Logistic Regression: Working with native probabilities

## Solutions for XGBoost

### Option 1: Use Working Models (Recommended)
Simply select **SVM** or **Logistic Regression** from the dropdown - both work perfectly!

### Option 2: Fix XGBoost (Advanced)
If you want to fix the XGBoost model:

1. **Retrain the model** with XGBoost >= 1.6.0
2. **Install compatible XGBoost version**:
   ```bash
   pip install xgboost>=1.6.0
   ```
3. **Use the patch script** (may not work for all cases):
   ```bash
   python fix_xgboost.py "trained model/xgb_best_pipeline.pkl"
   ```

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then select **SVM** or **Logistic Regression** for immediate emotion classification!

## Text Preprocessing

The app applies the following preprocessing steps to match the training pipeline:
- Convert to lowercase
- Remove URLs and mentions
- Extract hashtag words (remove #)
- Replace numbers with `<num>` token
- Remove punctuation except ! and ?
- Normalize whitespace

## Error Handling

The app includes comprehensive error handling for:
- Missing model files
- Empty or invalid text input
- Prediction errors
- Model loading issues

## Dependencies

See `requirements.txt` for the complete list of required packages.
