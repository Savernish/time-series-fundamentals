import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 

from generate_telemetry import generate_telemetry, create_sliding_windows

if __name__ == "__main__":
    # Generate and split data
    SEQUENCE_LENGTH = 20
    telemetry_data = generate_telemetry()
    
    # Chronological split on the DataFrame BEFORE scaling
    split_ratio = 0.8
    split_index = int(len(telemetry_data) * split_ratio)
    train_df = telemetry_data.iloc[:split_index]
    val_df = telemetry_data.iloc[split_index:]

    # Scale the data
    # Create the scaler and fit it ONLY on the training data
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    
    # Transform both the training and validation data
    scaled_train_data = scaler.transform(train_df)
    scaled_val_data = scaler.transform(val_df)
    
    # --- 3. Create Sliding Windows from SCALED data ---
    X_train, y_train = create_sliding_windows(scaled_train_data, SEQUENCE_LENGTH, target_columns=range(scaled_train_data.shape[1]))
    X_val, y_val = create_sliding_windows(scaled_val_data, SEQUENCE_LENGTH, target_columns=range(scaled_val_data.shape[1]))

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