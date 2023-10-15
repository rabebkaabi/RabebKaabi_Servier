from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
from tensorflow.keras.models import load_model
import tensorflow as tf
import json

app = Flask(__name__)

# Load your machine learning models
model1 = load_model('models/model1.h5')
model2 = load_model('models/model2.h5')

# Define the function to extract features from SMILES using RDKit
def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    features = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                            nBits=size,
                                                            useChirality=True,
                                                            useBondTypes=True,
                                                            useFeatures=False)
    return np.array(features)

# Define the function to preprocess SMILES strings (one-hot encoding)
def preprocess_smiles(smiles, input_shape, vocab_size):
    encoded_smiles = np.zeros((input_shape, vocab_size))
    for i, char in enumerate(smiles):
        if i >= input_shape:
            break
        char_index = ord(char) % vocab_size
        encoded_smiles[i, char_index] = 1
    return encoded_smiles.tolist()  # Convert to list

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
            app.logger.info(f"Received data: {data}")
            smile = data['smile']
            app.logger.info(f"SMILE: {smile}")

            # Add prediction code here
            prediction1 = make_prediction1(smile)  # Replace with your actual prediction code
            prediction2 = make_prediction2(smile)  # Replace with your actual prediction code
            app.logger.info(f"Prediction (Model 1): {prediction1}")
            app.logger.info(f"Prediction (Model 2): {prediction2}")

            return jsonify({'prediction_model1': prediction1, 'prediction_model2': prediction2})
        except Exception as e:
            app.logger.error(f"Error: {str(e)}")
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
