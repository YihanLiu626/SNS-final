import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, \
    GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# --------------------------
# Custom Positional Encoding Layer
# --------------------------
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # Apply sine to even indices in the array; cosine to odd indices.
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]  # Shape (1, position, d_model)
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, d_model)
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]


# --------------------------
# Transformer Encoder Block
# --------------------------
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Multi-head self-attention layer
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # Feed-forward network
    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2


# --------------------------
# Build Transformer Model with Positional Encoding
# --------------------------
def build_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=2, dropout=0.1):
    inputs = Input(shape=input_shape)
    # Add positional encoding to the inputs
    x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(25, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# --------------------------
# 1. Data Retrieval
# --------------------------
def get_data(ticker, start_date="2020-01-01", end_date="2025-01-01"):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


tickers = ['AAPL', 'GOOG', 'AMZN', 'TSLA']
all_data = {ticker: get_data(ticker) for ticker in tickers}
data = all_data[tickers[0]]  # Choose one stock, e.g., AAPL


# --------------------------
# 2. Compute Technical Indicators
# --------------------------
def compute_features(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Return'] = df['Close'].pct_change()
    # Remove rows with NaN values caused by indicator calculation
    df.dropna(inplace=True)
    return df


data = compute_features(data)

# --------------------------
# 3. Data Preprocessing and Normalization
# --------------------------
features = ['Close', 'Open', 'High', 'Low', 'MA_10', 'MA_50', 'Volume', 'Return']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])


# --------------------------
# 4. Create Time Series Dataset
# --------------------------
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, :])  # Past time_step days of features
        y.append(data[i, 0])  # Target: Close price of day i
    return np.array(X), np.array(y)


time_step = 60
X, y = create_sequences(scaled_data, time_step)

# --------------------------
# 5. Split into Training and Testing Sets
# --------------------------
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --------------------------
# 6. Build the Transformer Model
# --------------------------
model = build_transformer_model((X_train.shape[1], X_train.shape[2]),
                                head_size=64,
                                num_heads=4,
                                ff_dim=128,
                                num_transformer_blocks=2,
                                dropout=0.1)
model.summary()

# --------------------------
# 7. Train the Model (with EarlyStopping and ModelCheckpoint)
# --------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_transformer_model.h5', monitor='val_loss', save_best_only=True)
]

history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=callbacks)

# Plot training and validation loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss (Transformer)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# --------------------------
# 8. Prediction and Inverse Normalization
# --------------------------
predictions = model.predict(X_test).reshape(-1, 1)


def inverse_transform(predictions, scaler, n_features):
    # Create a dummy array to match original feature dimensions (fill remaining features with 0)
    dummy = np.zeros((predictions.shape[0], n_features - 1))
    pred_full = np.concatenate((predictions, dummy), axis=1)
    return scaler.inverse_transform(pred_full)[:, 0]


predictions_actual = inverse_transform(predictions, scaler, len(features))
y_test_actual = inverse_transform(y_test.reshape(-1, 1), scaler, len(features))

# --------------------------
# 9. Visualize the Prediction Results
# --------------------------
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, color='blue', label='Actual Stock Price')
plt.plot(predictions_actual, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction using Transformer with Position Encoding')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# --------------------------
# 10. Evaluate the Model
# --------------------------
mse = mean_squared_error(y_test_actual, predictions_actual)
print(f'Mean Squared Error: {mse}')