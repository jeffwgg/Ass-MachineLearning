import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Emotion Classification Demo",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Emotion label mapping (based on common emotion datasets)
EMOTION_LABELS = {
    0: "Sadness",
    1: "Joy", 
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

# Model information
MODEL_INFO = {
    "XGBoost": {
        "file": "trained model/xgb_best_pipeline.pkl",
        "description": "Gradient Boosting classifier with hyperparameter tuning"
    },
    "SVM": {
        "file": "trained model/svm_best_pipeline.pkl", 
        "description": "Support Vector Machine with linear kernel"
    },
    "Logistic Regression": {
        "file": "trained model/logreg_best_pipeline.pkl",
        "description": "Linear classifier with regularization"
    }
}

@st.cache_resource
def load_model(model_path):
    """Load and cache the selected model with compatibility handling"""
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        
        # Suppress warnings for version mismatches
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = joblib.load(model_path)
        
        # Handle XGBoost compatibility issues
        if hasattr(model, 'steps'):
            # It's a pipeline
            classifier = model.steps[-1][1]
            if hasattr(classifier, '__class__') and 'XGBClassifier' in str(classifier.__class__):
                # Clean up deprecated attributes that cause issues
                deprecated_attrs = ['use_label_encoder', '_le', '_label_encoder']
                for attr in deprecated_attrs:
                    if hasattr(classifier, attr):
                        try:
                            delattr(classifier, attr)
                        except:
                            pass
                
                # Try to patch the model for compatibility
                try:
                    # Force the model to work with newer XGBoost
                    if hasattr(classifier, 'get_booster'):
                        booster = classifier.get_booster()
                        # This helps with some compatibility issues
                        classifier._Booster = booster
                except:
                    pass
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_text(text):
    """Basic text preprocessing similar to the training pipeline"""
    if not text or not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = text.lower().strip()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', ' ', text)
    
    # Keep hashtag words (remove #)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Replace numbers with <num>
    text = re.sub(r'\b\d+\b', ' <num> ', text)
    
    # Keep only letters, numbers, and emotion punctuation
    text = re.sub(r'[^\w\s!?]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_predictions(model, text):
    """Get predictions and probabilities from the model"""
    try:
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        if not processed_text:
            return None, None
        
        # Get prediction
        prediction = model.predict([processed_text])[0]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            # Model supports probability prediction
            probabilities = model.predict_proba([processed_text])[0]
        else:
            # For SVM, use decision function + softmax
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function([processed_text])[0]
                probabilities = softmax(decision_scores)
            else:
                # Fallback: equal probabilities with max for predicted class
                probabilities = np.ones(6) * 0.1
                probabilities[prediction] = 0.5
                probabilities = probabilities / probabilities.sum()
        
        return prediction, probabilities
        
    except AttributeError as e:
        if 'use_label_encoder' in str(e):
            st.error("❌ XGBoost compatibility issue detected.")
            st.info("💡 **Solution**: Use SVM or Logistic Regression models, or retrain XGBoost with version >= 1.6.0")
            return None, None
        else:
            st.error(f"Model attribute error: {str(e)}")
            return None, None
    except Exception as e:
        error_msg = str(e).lower()
        if 'xgboost' in error_msg or 'xgb' in error_msg:
            st.error("❌ XGBoost prediction failed.")
            st.info("💡 **Try**: Switch to SVM or Logistic Regression models.")
        else:
            st.error(f"Error during prediction: {str(e)}")
        return None, None

def create_probability_chart(probabilities):
    """Create a bar chart showing emotion probabilities"""
    emotions = [EMOTION_LABELS[i] for i in range(6)]
    probs = probabilities * 100  # Convert to percentages
    
    # Create color map (highlight the highest probability)
    colors = ['lightblue'] * 6
    max_idx = np.argmax(probabilities)
    colors[max_idx] = 'darkblue'
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probs,
            marker_color=colors,
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Prediction Probabilities",
        xaxis_title="Emotions",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # App title and description
    st.title("😊 Emotion Classification Demo")
    st.markdown("---")
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        list(MODEL_INFO.keys()),
        help="Select which trained model to use for prediction"
    )
    
    # Display model information
    st.sidebar.markdown(f"**Selected:** {selected_model}")
    st.sidebar.markdown(f"*{MODEL_INFO[selected_model]['description']}*")
    
    # Show compatibility warning for XGBoost
    if selected_model == "XGBoost":
        st.sidebar.info("✅ **XGBoost Model**: Retrained with XGBoost 2.0.3")
    
    # Load the selected model
    model_path = MODEL_INFO[selected_model]['file']
    model = load_model(model_path)
    
    if model is None:
        st.error("Please ensure the model files are available in the 'trained model/' directory.")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Text Input")
        
        # Text input area
        user_text = st.text_area(
            "Enter a sentence or paragraph to classify its emotion:",
            height=150,
            placeholder="Example: I'm so excited about my vacation next week!"
        )
        
        # Buttons
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            predict_button = st.button("🔮 Predict Emotion", type="primary", use_container_width=True)
        
        with button_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()
    
    with col2:
        st.header("Model Info")
        st.info(f"**Current Model:** {selected_model}")
        
        # Emotion legend
        st.subheader("Emotion Classes")
        for i, emotion in EMOTION_LABELS.items():
            st.write(f"**{i}:** {emotion}")
    
    # Prediction results
    if predict_button:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text to classify.")
        else:
            with st.spinner("Analyzing emotion..."):
                prediction, probabilities = get_predictions(model, user_text)
            
            if prediction is not None and probabilities is not None:
                st.markdown("---")
                st.header("🎯 Prediction Results")
                
                # Main prediction
                predicted_emotion = EMOTION_LABELS[prediction]
                confidence = probabilities[prediction] * 100
                
                # Display main result
                st.success(f"**Predicted Emotion:** {predicted_emotion} ({confidence:.1f}% confidence)")
                
                # Create two columns for results
                result_col1, result_col2 = st.columns([3, 2])
                
                with result_col1:
                    # Probability chart
                    fig = create_probability_chart(probabilities)
                    st.plotly_chart(fig, use_container_width=True)
                
                with result_col2:
                    # Top 3 predictions
                    st.subheader("Top 3 Predictions")
                    
                    # Get top 3 indices
                    top3_indices = np.argsort(probabilities)[-3:][::-1]
                    
                    for i, idx in enumerate(top3_indices):
                        emotion = EMOTION_LABELS[idx]
                        prob = probabilities[idx] * 100
                        
                        if i == 0:
                            st.metric(f"🥇 {emotion}", f"{prob:.1f}%")
                        elif i == 1:
                            st.metric(f"🥈 {emotion}", f"{prob:.1f}%")
                        else:
                            st.metric(f"🥉 {emotion}", f"{prob:.1f}%")
                
                # Expandable detailed results
                with st.expander("📊 Detailed Probability Breakdown"):
                    prob_df = pd.DataFrame({
                        'Emotion': [EMOTION_LABELS[i] for i in range(6)],
                        'Probability': probabilities,
                        'Percentage': [f"{p*100:.2f}%" for p in probabilities]
                    })
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Display preprocessed text
                with st.expander("🔧 Preprocessed Text"):
                    processed = preprocess_text(user_text)
                    st.code(processed)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit • Powered by scikit-learn pipelines</p>
        <p>Models: XGBoost, SVM, Logistic Regression</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
