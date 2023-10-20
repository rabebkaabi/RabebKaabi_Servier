#Modèle 1 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import fingerprint_features, preprocess_smiles, YOUR_INPUT_SHAPE, YOUR_VOCAB_SIZE
from sklearn.model_selection import train_test_split


# Load the dataset
data = pd.read_csv('data/dataset_single.csv')

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
model1.save('models/model1.h5')

# Split the data into training, validation, and test sets
X = data['smiles']
y = data['P1']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

INPUT_SHAPE = 50
VOCAB_SIZE = 30
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
model2.save('models/model2.h5')

# ... (previous code)

# Load the dataset_multi.csv dataset
multi_data = pd.read_csv('data/dataset_multi.csv')

# Extract features for each molecule
multi_data['features'] = multi_data['smiles'].apply(fingerprint_features)

# Split the data into training, validation, and test sets
X_multi = np.array(multi_data['features'].tolist())
properties = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']

for prop in properties:
    y_multi = np.array(multi_data[prop])
    X_train, X_temp, y_train, y_temp = train_test_split(X_multi, y_multi, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Define and compile Model3 for each property
    model3 = keras.Sequential([
        layers.Input(shape=(2048,)),  # Replace with the actual feature vector size
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train Model3 for each property
    model3.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    # Evaluate Model3 for each property
    loss, accuracy = model3.evaluate(X_test, y_test)
    print(f"Test loss for {prop}: {loss}, Test accuracy for {prop}: {accuracy}")

    # Save the trained Model3 for each property
    model3.save(f'models/model3_{prop}.h5')


from app import app

if __name__ == '__main__':
    app.run(debug=True)
