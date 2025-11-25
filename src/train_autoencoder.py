# src/train_autoencoder.py
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model
from data_prep import prepare_data

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def train_autoencoder():
    train_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'KDDTrain+.txt')
    test_file  = os.path.join(os.path.dirname(__file__), '..', 'data', 'KDDTest+.txt')

    # prepare but do not SMOTE; we need original labels to select 'normal'
    X_train, X_test, y_train, y_test = prepare_data(train_file, test_file, save_dir=MODEL_DIR, binary=True, apply_smote=False)

    # We need mapping from prepare_data: it returns numpy arrays for X,y; but we need original indices to filter normals
    # Quick hack: re-run loading to get y_train labels
    from utils import load_nslkdd
    df_train = load_nslkdd(train_file)
    y_train_labels = (df_train['label'] == 'normal').astype(int).values
    # pick only normal rows
    X_train_norm = X_train[y_train_labels == 1]

    input_dim = X_train.shape[1]
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(input_dim, activation='linear')(x)

    ae = Model(inputs, outputs)
    ae.compile(optimizer='adam', loss='mse')
    # Remove validation_split seed control for variation
    ae.fit(X_train_norm, X_train_norm, epochs=30, batch_size=256, validation_split=0.1, shuffle=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    ae.save(os.path.join(MODEL_DIR, 'ae_model.h5'))
    print("Saved autoencoder.")

if __name__ == "__main__":
    train_autoencoder()
