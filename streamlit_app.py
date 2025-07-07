import streamlit as st
from services.tradier_client import TradierClient
import os

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
