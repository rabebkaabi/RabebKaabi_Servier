from flask import Flask, request, render_template, jsonify
import json
import logging
from utils import fingerprint_features, preprocess_smiles, YOUR_INPUT_SHAPE, YOUR_VOCAB_SIZE
from tensorflow.keras.models import load_model 
from utils import np
import numpy as np  

app = Flask(__name__)

# Configure the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load your machine learning models
model1 = load_model('models/model1.h5')
model2 = load_model('models/model2.h5')
properties = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']
model3 = {}
for prop in properties:
    model3[prop] = load_model(f'models/model3_{prop}.h5')

# Define functions to make predictions using your models
def make_prediction1(smile):
    # You need to define how to preprocess and use model1 to make a prediction
    # Example:
    features = fingerprint_features(smile)
    prediction = model1.predict(np.array([features]))
    return float(prediction[0][0])  # Convert to float

def make_prediction2(smile):
    # You need to define how to preprocess and use model2 to make a prediction
    # Example:
    encoded_smiles = preprocess_smiles(smile, 50, 30)
    prediction = model2.predict(np.array([encoded_smiles]))
    return float(prediction[0][0])  # Convert to float

def make_prediction3(smile, property_name):
    features = fingerprint_features(smile)
    model = model3.get(property_name)
    if model is not None:
        prediction = model.predict(np.array([features]))
        return float(prediction[0][0])

# Add your 'serve' function here
def serve():
    app.run(debug=True)

# Add your 'train' function here
def train():
    # Implement your training logic here
    pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            logger.info(f"Received data: {data}")
            smile = data['smile']
            logger.info(f"SMILE: {smile}")

            # Add prediction code here
            prediction1 = make_prediction1(smile)  # Replace with your actual prediction code
            prediction2 = make_prediction2(smile)  # Replace with your actual prediction code
            predictions3 = {}
            for prop in properties:
                prediction3 = make_prediction3(smile, prop)
                predictions3[prop] = prediction3
            logger.info(f"Prediction (Model 1): {prediction1}")
            logger.info(f"Prediction (Model 2): {prediction2}")

            response = {
                'prediction_model1': prediction1,
                'prediction_model2': prediction2,
                'prediction_model3': predictions3
            }

            return jsonify(response)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
