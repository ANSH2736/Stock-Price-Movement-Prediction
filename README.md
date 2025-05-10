# 📈 Stock Price Movement Prediction using Machine Learning

This project predicts the movement of stock prices for the next day (up or down) using machine learning. The model analyzes historical stock data, including the Open, High, Low, Close prices, Volume, and various technical indicators like Moving Averages (MA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD) to forecast price direction.

A web interface, built with Streamlit, allows users to easily input a stock ticker and receive predictions about the next day's price movement.

---

## 🔍 Project Overview

- **Domain**: Financial Analytics, Machine Learning, Time Series Forecasting
- **Goal**: To predict whether a stock's price will rise or fall the next day using historical stock data and technical indicators.
- **Technology Stack**:
  - **Data Analysis**: Python, Pandas, Numpy
  - **Machine Learning**: Scikit-learn, XGBoost
  - **Web Interface**: Streamlit
- **Key Features**:
  - Predicts stock price movement (Up/Down) for the next trading day
  - Uses multiple machine learning models for comparison
  - Easy-to-use web interface for inputting stock tickers

---

## 🧠 Machine Learning Models

Three machine learning models have been trained and used for the prediction:

- **Random Forest**: A powerful ensemble method based on decision trees.
- **Logistic Regression**: A simple but effective model for binary classification.
- **XGBoost**: An optimized gradient boosting model known for its high performance.

These models have been trained using historical stock data and technical indicators, and they predict whether the stock price will go up or down the next day.

---

## 📊 Data

The model uses the following data sources:

- **Historical Stock Data**: Open, High, Low, Close, Volume (OHLCV data)
- **Technical Indicators**:
  - **Moving Averages (MA)**: Helps smooth out price data to identify trends.
  - **Relative Strength Index (RSI)**: Measures the speed and change of price movements to identify overbought or oversold conditions.
  - **Moving Average Convergence Divergence (MACD)**: A momentum indicator that shows the relationship between two moving averages of a stock's price.

The data is gathered using APIs such as Yahoo Finance or Alpha Vantage.

---

## 🛠️ Setup Instructions

### Prerequisites

Before running this project, you need to install the following libraries:

```bash
pip install -r requirements.txt