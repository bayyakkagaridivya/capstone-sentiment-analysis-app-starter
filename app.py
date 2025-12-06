# --- 1. CORE IMPORTS ---
from flask import Flask, render_template, request
import os # <-- Must be imported first to use os.environ
import numpy as np
import pickle

# --- 2. TENSORFLOW/VADER IMPORTS ---
# Keras Imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 3. TENSORFLOW WARNING SUPPRESSION (CRITICAL FIX) ---
# Suppress TensorFlow GPU warnings (non-critical in CPU environments)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define application metadata
APP_VERSION = "v1.0"

app = Flask(__name__)

# Get the base directory (where app.py is located)
basedir = os.path.abspath(os.path.dirname(__file__))

# --- VADER SETUP (from US-01) ---
vader_analyzer = SentimentIntensityAnalyzer()
# --- CUSTOM KERAS MODEL SETUP (New) ---
# NOTE: It is critical to load the model and tokenizer only ONCE when the app starts.
if os.environ.get('RUNNING_TESTS') != 'True':
    try:
        tokenizer_path = os.path.join(basedir, 'tokenizer.pickle')
        model_path = os.path.join(basedir, 'uci_sentimentanalysis.h5')
        # Load the tokenizer
        with open(tokenizer_path, 'rb') as handle:
            keras_tokenizer = pickle.load(handle)
        
        # Load the Keras model
        keras_model = load_model(model_path)
        
        # Determine the max sequence length
        MAX_SEQUENCE_LENGTH = keras_model.input_shape[1]
        
    except Exception as e:
        print(f"Error loading Keras model or tokenizer: {e}")
        keras_model = None
        keras_tokenizer = None
        MAX_SEQUENCE_LENGTH = 0
else:
    # If running tests, set mock objects to avoid the AttributeError
    print("Running tests: Skipping Keras model load.")
    keras_model = None # Will be replaced by MagicMock in tests.py
    keras_tokenizer = None # Will be replaced by MagicMock in tests.py
    MAX_SEQUENCE_LENGTH = 0

# --- Update the Keras prediction function in app.py ---
def predict_keras_sentiment(text):
    """Preprocesses text and returns a sentiment score using the Keras model."""
    if keras_model is None or keras_tokenizer is None:
        return {'error': 'Custom model not loaded.'}

    # 1. Tokenize the text
    tknz_text = keras_tokenizer.texts_to_sequences([text])
    
    # CRITICAL: Check if tokenization produced any sequences
    if not tknz_text or not tknz_text[0]:
        # If the input text contains only words outside of the max_features vocabulary, 
        # tknz_text might be empty, leading to an error later.
        # Handle this case by returning a neutral result or padding an empty list.
        # We will pad the sequence to ensure 'padded_text' is always defined.
        sequence_to_pad = []
    else:
        sequence_to_pad = tknz_text
    
    # 2. Pad the sequence
    # 'padded_text' must be defined here, regardless of the tokenization result
    padded_text = pad_sequences(sequence_to_pad, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    # CRITICAL FIX: Check if the padded input contains zero samples
    # We check the shape's first dimension (batch size). If it's 0, return a neutral result.
    if padded_text.shape[0] == 0:
        # This handles cases where input is purely unrecognized (e.g., just punctuation or numbers)
        return {
            'prediction': 0.5, # Neutral score
            'positive': 0.5,
            'negative': 0.5,
            'result': 'Neutral', 
            'overall': 'Neutral'
        }
    
    # 3. Make prediction
    prediction = keras_model.predict(padded_text)[0][0]
    
    # 4. Determine overall result
    overall_sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    
    return {
        'prediction': float(prediction),
        'positive': float(prediction),
        'negative': float(1.0 - prediction),
        'result': overall_sentiment,  # Use this for display
        'overall': overall_sentiment # Add the simple result for icon
    }

# --- Update the main index route in app.py ---

@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize variables to None or empty
    sentiment = None
    model_used = None
    error = None # New variable for error messages
    
    if request.method == "POST":
        text = request.form.get("user_text")
        selected_model = request.form.get("model_selector")

        if not text or text.strip() == "": # Check if text is empty or only whitespace
            error = "Please enter some text for sentiment analysis."
            
        else:
            # Existing analysis logic moves here (inside the 'else' block)
            if selected_model == "vader":
                # ... VADER analysis and 'overall' logic ...
                sentiment = vader_analyzer.polarity_scores(text)
                model_used = "VADER"
                
                # VADER Logic for overall sentiment
                compound_score = sentiment['compound']
                if compound_score >= 0.05:
                    sentiment['overall'] = 'Positive'
                elif compound_score <= -0.05:
                    sentiment['overall'] = 'Negative'
                else:
                    sentiment['overall'] = 'Neutral'
            
            elif selected_model == "keras":
                # ... Keras analysis and 'overall' logic ...
                sentiment = predict_keras_sentiment(text)
                model_used = "Custom Keras Model"

    # Pass the new 'error' variable to the template
    return render_template('form.html', sentiment=sentiment, model_used=model_used, error=error, app_version=APP_VERSION)


if __name__ == "__main__":
    app.run(debug=True)