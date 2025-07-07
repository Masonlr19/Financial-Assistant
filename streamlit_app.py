import streamlit as st
from services.tradier_client import TradierClient
import os
from models.xgboost_classifier import WeeklyPriceIncreasePredictor
import pandas as pd

def main():
    st.set_page_config(page_title="Financial Assistant", layout="wide")
    st.title("📈 ML Financial Assistant")

    api_key = st.secrets["TRADIER_API_KEY"]
    tradier_client = TradierClient(api_key)

    symbol = st.text_input("Enter a stock symbol (e.g., AAPL)", value="AAPL")

    if st.button("Fetch Historical Data"):
        try:
            data = tradier_client.get_historical_data(symbol)
            st.json(data)
        except Exception as e:
            st.error(f"Error fetching data: {e}")

if __name__ == "__main__":
    main()

if st.button("Predict Weekly Price Increase"):
    try:
        predictor = WeeklyPriceIncreasePredictor()
        predictor.train(data)

        preds, confs = predictor.predict_next_5_weeks()

        st.write("### Predictions for next 5 Fridays:")
        for i, (p, c) in enumerate(zip(preds, confs), 1):
            direction = "Increase 📈" if p == 1 else "Decrease 📉"
            st.write(f"Week {i}: {direction} with confidence {c}/100")
    except Exception as e:
        st.error(f"Prediction error: {e}")
