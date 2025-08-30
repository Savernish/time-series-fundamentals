import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

from generate_telemetry import generate_telemetry, create_sliding_windows

if __name__ == "__main__":
    SEQUENCE_LENGTH = 20
    telemetry_data = generate_telemetry()
    X, Y = create_sliding_windows(telemetry_data, SEQUENCE_LENGTH, target_columns=telemetry_data.columns)

    # Perform a chronological train/test split.
    # Using the %80 for training and %20 for validation
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_val = X[:split_index], X[split_index:]
    Y_train, Y_val = Y[:split_index], Y[split_index:]

    print(f"Training data shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, Y={Y_val.shape}")

    # Define the LSTM model
    model = Sequential([
        LSTM(64, input_shape=(SEQUENCE_LENGTH, X_train.shape[2])),
        Dense(Y_train.shape[1])
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    model.summary()

    # Train the model
    print("\nTraining the LSTM forecaster...")
    history = model.fit( 
        X_train, 
        Y_train, 
        epochs=20, 
        validation_data=(X_val, Y_val)
    )

    # Evaluate the model and visualize results
    print("\nEvaluating and plotting results...")
    # Use the trained model to make predictions on the validation set (X_val).
    predictions = model.predict(X_val)
    # Create a plot to compare the predictions with the true values
    # We'll just visualize the 'roll' channel (index 0) for clarity.
    plt.figure(figsize=(15, 6))
    plt.title("LSTM Forecasting Results for 'Roll' Sensor")
    plt.plot(Y_val[:, 0], label="True Values", color='blue')
    plt.plot(predictions[:, 0], label="Predictions", color='red', linestyle='--')
    plt.xlabel("Time Step (in validation set)")
    plt.ylabel("Roll Value")
    plt.legend()
    plt.grid(True)
    plt.show()