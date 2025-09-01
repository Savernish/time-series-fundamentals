from generate_telemetry import generate_telemetry, create_sliding_windows
from sklearn.preprocessing import MinMaxScaler

def LoadData(SEQUENCE_LENGTH):
    # Generate telemetry data
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

    return X_train, y_train, X_val, y_val