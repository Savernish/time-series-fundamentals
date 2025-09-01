import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from generate_telemetry import generate_telemetry
# Import the corrected window function
from train_autoencoder import create_specialist_windows_autoencoder

if __name__ == "__main__":
    SEQUENCE_LENGTH = 20
    telemetry_data = generate_telemetry()
    split_index = int(len(telemetry_data) * 0.8)
    train_df = telemetry_data.iloc[:split_index]
    val_df = telemetry_data.iloc[split_index:]

    scaler = MinMaxScaler()
    scaler.fit(train_df)
    scaled_val_data = scaler.transform(val_df)
    
    model = tf.keras.models.load_model('specialist_autoencoder.h5')

    # --- Create an Anomalous Dataset ---
    # We create the anomaly in the raw scaled data first, THEN create windows
    scaled_val_anomalous = scaled_val_data.copy()
    print("Injecting anomaly into 'roll' channel...")
    anomaly_start_index = 100
    anomaly_end_index = 120
    scaled_val_anomalous[anomaly_start_index:anomaly_end_index, 0] += 2.0 
    
    # Create windows from both normal and anomalous data
    X_val_orig, _ = create_specialist_windows_autoencoder(scaled_val_data, SEQUENCE_LENGTH)
    X_val_anomalous, _ = create_specialist_windows_autoencoder(scaled_val_anomalous, SEQUENCE_LENGTH)

    # --- Calculate Reconstruction Error ---
    reconstructions = model.predict(X_val_anomalous)
    mae = np.mean(np.abs(X_val_anomalous[:, :, :2] - reconstructions), axis=(1, 2))

    # --- Determine Anomaly Threshold (using normal data) ---
    normal_reconstructions = model.predict(X_val_orig)
    normal_mae = np.mean(np.abs(X_val_orig[:, :, :2] - normal_reconstructions), axis=(1, 2))
    threshold = np.mean(normal_mae) + 4 * np.std(normal_mae)
    print(f"Anomaly detection threshold (MAE): {threshold:.4f}")

    # --- Visualize the Results ---
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(X_val_anomalous[:, -1, 0], label="Anomalous Roll Data (Input)")
    plt.plot(reconstructions[:, -1, 0], label="Reconstructed Roll Data")
    plt.title("Specialist Reconstruction of Anomalous 'Roll' Sensor Data")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(mae, label="Reconstruction Error (MAE)")
    plt.axhline(y=threshold, color='r', linestyle='--', label="Anomaly Threshold")
    plt.title("Reconstruction Error Over Time")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    anomalies_detected = np.where(mae > threshold)[0]
    print(f"\nDetected anomalies at time steps (indices): {anomalomedialies_detected}")