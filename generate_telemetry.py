import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_telemetry(timesteps=1000):
    """
    Generates a synthetic multi-variate drone telemetry dataset.
    Returns a pandas DataFrame.
    """
    time = np.linspace(0, 100, timesteps)
    
    # Generate each sensor signal using NumPy.
    # - roll: Should oscillate. A sine wave is a good choice.
    # - pitch: Should also oscillate. A cosine wave can be used to be out of phase with roll.
    # - yaw: A slower, wider sine wave to simulate gradual turning. So the frequency should be lower.
    # - altitude: A gradual climb. A linear function (like linspace) with some noise.
    # - battery_voltage: Should start high and slowly decay. A line with a small negative slope and noise.
    roll = 0.7 * np.sin(2 * np.pi * time + 0.5) + 0.4 * np.where(np.sin(np.random.uniform(0, 1, timesteps) * np.pi * time) <= 0, np.sin(2 * np.pi * time), 0) + np.random.normal(0, 0.05, timesteps)
    pitch = 0.8 * np.cos(2 * np.pi * time + 0.5) + 0.4 * np.where(np.sin(np.random.uniform(0, 1, timesteps) * np.pi * time) <= 0, np.sin(2 * np.pi * time), 0) + np.random.normal(0, 0.05, timesteps)
    yaw = 0.3 * np.sin(0.5 * np.pi * time + 0.5) + np.random.normal(0, 0.05, timesteps)
    altitude = np.linspace(0, 100, timesteps) + np.random.normal(0, 1, timesteps)
    battery_voltage = np.linspace(12, 10, timesteps) + np.random.normal(0, 0.1, timesteps)

    # Combine the signals into a single pandas DataFrame.
    telemetry_df = pd.DataFrame({
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        'altitude': altitude,
        'battery_voltage': battery_voltage
    })
    
    return telemetry_df

def create_sliding_windows(data, sequence_length, target_columns):
    """
    Creates input/output pairs from a multi-variate time series DataFrame.
    """
    # TODO: Implement the sliding window logic.
    # - The input (X) for each sample should be a window of the full DataFrame.
    # - The output (y) for each sample should be the values of the `target_columns`
    #   at the time step immediately following the input window.
    # - Return X and y as NumPy arrays.
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length].values)
        y.append(data.iloc[i + sequence_length][target_columns].values)
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Configuration
    SEQUENCE_LENGTH = 20 # Use the last 20 time steps to predict the next state

    # Generate data
    telemetry_data = generate_telemetry()
    print("Generated Telemetry Data Head:")
    print(telemetry_data.head())
    
    # Create windows
    # We will try to predict all sensor values in the next time step.
    X, y = create_sliding_windows(telemetry_data, SEQUENCE_LENGTH, target_columns=telemetry_data.columns)
    
    print(f"\nShape of input sequences (X): {X.shape}")
    # Expected X shape: (num_samples, sequence_length, num_sensors)
    
    print(f"Shape of target values (y): {y.shape}")
    # Expected y shape: (num_samples, num_sensors)

    # Visualize the data
    print("\nVisualizing generated telemetry data...")
    telemetry_data.plot(subplots=True, figsize=(15, 10), title="Synthetic Drone Telemetry")
    plt.tight_layout()
    plt.show()