from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import logging
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for frontend compatibility

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Load the trained model and vectorizer
MODEL_PATH = "model/decision_tree.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("✅ Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    logging.error(f"❌ File not found: {e}")
    raise
except Exception as e:
    logging.error(f"❌ Error loading model or vectorizer: {e}")
    raise

# ======== Serve Frontend Pages =========
@app.route('/')
def home():
    logging.info("Home route accessed. Attempting to render login.html.")
    return render_template('login.html')  # This renders the login.html file as the home page

@app.route('/data_demo')
def data_demo():
    return render_template('data_demo.html')

@app.route('/career_sugession')
def career_sugession():
    return render_template('career_sugession.html')

@app.route('/career_results')
def career_results():
    return render_template('career_results.html')

# ======== API Endpoint for Model Prediction =========
@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # Get JSON data from the request
        data = request.get_json()
        answers = data.get('answers', [])

        # Validate input data
        if not answers or len(answers) != 5:
            logging.warning("Invalid number of answers received.")
            return jsonify({'error': 'Invalid number of answers. Expected 5 answers.'}), 400

        # Vectorize the answers using the TF-IDF vectorizer
        input_vector = vectorizer.transform(answers)

        # Predict skill level
        prediction = model.predict(input_vector)[0]

        # Map prediction to skill level
        skill_mapping = {0: 'Beginner', 1: 'Intermediate', 2: 'Expert'}
        skill_level = skill_mapping.get(prediction, 'Unknown')

        logging.info(f"Prediction: {skill_level}")
        return jsonify({'skill_level': skill_level})
    
    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

# ======== Global Error Handler =========
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled Error: {e}")
    return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
