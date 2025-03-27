#  Stock Price Prediction with LSTM + GRU

This project predicts future stock prices (AAPL, GOOG, AMZN) using a deep learning model (LSTM + GRU) with technical indicators.

##  Features
- Supports AAPL, GOOG, AMZN
- Predicts next N days of stock prices
- Uses LSTM + GRU hybrid model
- Includes Transformer model for comparison
- Provides chatbox-style client for user interaction
- Flask server handles model inference and chart generation

## Files
- `server.py`: Flask backend to serve predictions
- `client.py`: Chatbox client interface
- `StockPricePredict_latest.py`: Model training (LSTM + GRU)
- `StockPricePredict_transformer.py`: Alternative Transformer model
- `fixed_best_model_*.h5`: Trained models (AAPL, GOOG, AMZN)
- `scaler_*.pkl`: Corresponding scalers

## ▶️ How to Use

1. Start the Flask server:
   ```bash
   python server.py
2.	Run the chatbot client:
   python client.py
3.	Example queries:
What will AAPL stock price be in 10 days?
Predict GOOG stock price in 30 days.
