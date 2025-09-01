import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from generate_telemetry import generate_telemetry

def create_specialist_windows_autoencoder(data, sequence_length):
    """
    Correctly creates windows for a sequence-to-sequence autoencoder.
    - X: The input sequence with all features.
    - y: The target sequence with only the features to be reconstructed.
    """
    X, y = [], []
    target_indices = [0, 1] # Reconstruct only roll and pitch
    for i in range(len(data) - (sequence_length + 1)):
        X.append(data[i:(i + sequence_length)])
        # THE FIX: The target y is the SAME full window as X, but only the target columns
        y.append(data[i:(i + sequence_length), target_indices])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    SEQUENCE_LENGTH = 20
    telemetry_data = generate_telemetry()
    
    split_index = int(len(telemetry_data) * 0.8)
    train_df = telemetry_data.iloc[:split_index]
    val_df = telemetry_data.iloc[split_index:]

    scaler = MinMaxScaler()
    scaler.fit(train_df)
    scaled_train_data = scaler.transform(train_df)
    scaled_val_data = scaler.transform(val_df)
    
    X_train, y_train = create_specialist_windows_autoencoder(scaled_train_data, SEQUENCE_LENGTH)
    X_val, y_val = create_specialist_windows_autoencoder(scaled_val_data, SEQUENCE_LENGTH)

    print(f"Input shape (X_train): {X_train.shape}")
    print(f"Target shape (y_train): {y_train.shape}") # Should be (samples, 20, 2)

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQUENCE_LENGTH, X_train.shape[2])),
        RepeatVector(SEQUENCE_LENGTH),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(units=2)) # Output layer has 2 neurons for roll and pitch
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.summary()

    print("\nTraining the Specialist LSTM Autoencoder (Corrected)...")
    history = model.fit(
        X_train, y_train, # Shapes now match
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=2
    )
    
    print("\nSaving the trained specialist autoencoder model...")
    model.save('specialist_autoencoder.h5')
    print("Autoencoder training complete and model saved.")