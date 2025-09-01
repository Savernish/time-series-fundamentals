import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from data_loader import LoadData

if __name__ == "__main__":
    # Generate and split data
    SEQUENCE_LENGTH = 60
    X_train, y_train, X_val, y_val = LoadData(SEQUENCE_LENGTH)

    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")

    # --- 4. Define and Compile the LSTM Model ---
    model = Sequential([
        LSTM(units=64, input_shape=(SEQUENCE_LENGTH, X_train.shape[2])),
        Dense(units=X_train.shape[2])
    ])
    
    optimizer = Adam(learning_rate=0.001) # We can try a slightly higher LR with scaled data
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()

    # --- 5. Train the Model ---
    print("\nTraining the LSTM forecaster on SCALED data...")
    history = model.fit( 
        X_train, 
        y_train, 
        epochs=30,
        validation_data=(X_val, y_val),
        batch_size=32,
        verbose=2
    )

    # --- 6. Evaluate and Visualize ---
    print("\nEvaluating and plotting results...")
    # Predict on the scaled validation data
    scaled_predictions = model.predict(X_val)
    
    # CRITICAL: Inverse transform the predictions to bring them back to the original scale
    predictions = scaler.inverse_transform(scaled_predictions)
    
    # Also, inverse transform the true validation labels for a fair comparison
    true_values = scaler.inverse_transform(y_val)

    # Create a plot to compare the predictions with the true values
    plt.figure(figsize=(15, 6))
    plt.title("LSTM Forecasting Results for 'Roll' Sensor (Corrected)")
    plt.plot(true_values[:, 0], label="True Values", color='blue')
    plt.plot(predictions[:, 0], label="Predictions", color='red', linestyle='--')
    plt.xlabel("Time Step (in validation set)")
    plt.ylabel("Roll Value (Original Scale)")
    plt.legend()
    plt.grid(True)
    plt.show()