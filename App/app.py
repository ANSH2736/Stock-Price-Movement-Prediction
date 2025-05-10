
import streamlit as st
import yfinance as yf
import pandas as pd
import pickle
from feature_engineering import compute_technical_indicators


st.title("Stock Price Movement Prediction")

# User input for ticker symbol
ticker = st.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")
model_option = st.selectbox("Choose model", ["Random Forest", "XGBoost", "Logistic Regression"])

if st.button("Predict Tomorrow's Movement"):
    # Load the selected model
    if model_option == "Random Forest":
        model = pickle.load(open("../models/random_forest_model.pkl", "rb"))
    elif model_option == "XGBoost":
        model = pickle.load(open("../models/xgboost_model.pkl", "rb"))
    else:
        model = pickle.load(open("../models/logistic_model.pkl", "rb"))

    # Fetch recent data for the ticker
    df = yf.download(ticker, period="1mo", interval="1d")
    df.reset_index(inplace=True)
    if df.empty:
        st.error("No data found for ticker.")
    else:
        # Prepare features for the latest available day
        df = compute_technical_indicators(df)
        latest = df.iloc[-1]
        features = pd.DataFrame([latest[['Close','Volume','MA20','MA50','RSI','MACD','Signal']]])
        pred = model.predict(features)[0]
        if pred == 1:
            st.success(f"The model predicts **UP** for tomorrow ({ticker}).")
        else:
            st.error(f"The model predicts **DOWN** for tomorrow ({ticker}).")
