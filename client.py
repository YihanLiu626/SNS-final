import requests
import re
import base64
from PIL import Image
import io

SERVER_URL = "http://127.0.0.1:5001/predict"

def extract_stock_info(user_input):

    match = re.search(r"(GOOG|AAPL|AMZN)", user_input.upper())
    if not match:
        return None, None

    ticker = match.group(0)
    days_match = re.search(r"(\d+)", user_input)
    if not days_match:
        return ticker, None

    days = int(days_match.group(0))
    if days not in [1, 5, 10, 30]:
        return ticker, None

    return ticker, days

def chatbot():

    print("\n🔮 Welcome to the Oracle Stock Predictor 🔮")
    print("Ask me about stock price predictions!")
    print("Examples:")
    print("  - What will GOOG stock price be in 5 days?")
    print("  - Predict AAPL stock price in 30 days.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Oracle: Goodbye! 🔮")
            break

        ticker, days = extract_stock_info(user_input)
        if not ticker or not days:
            print("Oracle: Please enter a valid stock (GOOG, AMZN, AAPL) and period (1, 5, 10, 30 days).")
            continue

        response = requests.post(SERVER_URL, json={"ticker": ticker, "days": days})
        data = response.json()

        if "error" in data:
            print(f"Oracle: {data['error']}")
            continue

        print(f"Oracle: Predicted prices for {ticker}:")
        for date, price in data["predictions"].items():
            print(f"  📅 {date}: ${price:.2f}")

        chart_data = base64.b64decode(data["chart"])
        image = Image.open(io.BytesIO(chart_data))
        image.show()

if __name__ == "__main__":
    chatbot()