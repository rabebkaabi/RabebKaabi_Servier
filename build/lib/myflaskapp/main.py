#Modèle 1 
import numpy as np
import pandas as pd
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the function to extract features from SMILES using RDKit
def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                          nBits=size,
                                                          useChirality=True,
                                                          useBondTypes=True,
                                                          useFeatures=False
                                                          )

# Load the dataset
data = pd.read_csv('C:/Users/natan/Desktop/test/dataset_single.csv')

# Extract features for each molecule
data['features'] = data['smiles'].apply(fingerprint_features)

# Split the data into training, validation, and test sets
X = np.array(data['features'].tolist())
y = np.array(data['P1'])
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the Model1 architecture
model1 = keras.Sequential([
    layers.Input(shape=(2048,)),  # Replace with the actual feature vector size
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model1
model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate Model1
loss, accuracy = model1.evaluate(X_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")

# Save the trained Model1
model1.save('model1.h5')

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



# Fixed values for input shape and vocab size
YOUR_INPUT_SHAPE = 50
YOUR_VOCAB_SIZE = 30

# Split the data into training, validation, and test sets
X = data['smiles']
y = data['P1']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define a function to preprocess SMILES strings (one-hot encoding)
def preprocess_smiles(smiles):
    # Initialize an array for one-hot encoding
    encoded_smiles = np.zeros((YOUR_INPUT_SHAPE, YOUR_VOCAB_SIZE))
    # Convert each character to a one-hot encoded vector
    for i, char in enumerate(smiles):
        if i >= YOUR_INPUT_SHAPE:
            break  # Truncate or pad the sequence to the fixed length
        # Map the character to an index in the one-hot vector
        char_index = ord(char) % YOUR_VOCAB_SIZE
        encoded_smiles[i, char_index] = 1
    return encoded_smiles

# Apply preprocessing to the SMILES strings and convert to NumPy arrays
X_train = np.array([preprocess_smiles(smiles) for smiles in X_train])
X_val = np.array([preprocess_smiles(smiles) for smiles in X_val])
X_test = np.array([preprocess_smiles(smiles) for smiles in X_test])
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

# Define the Model2 architecture
model2 = keras.Sequential([
    layers.Input(shape=(YOUR_INPUT_SHAPE, YOUR_VOCAB_SIZE)),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model2
model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate Model2
loss, accuracy = model2.evaluate(X_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")

# Save the trained Model2
model2.save('model2.h5')

from app import app

if __name__ == '__main__':
    app.run(debug=True)
