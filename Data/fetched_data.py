import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, period="1y", interval="1d"):
    """
    Fetches historical stock data for the given ticker and saves to CSV.
    """
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    filename = f"data/{ticker}_data.csv"
    df.to_csv(filename, index=False)
    print(f"Saved data for {ticker} to {filename}")
    return df

if __name__ == "__main__":
    # Example tickers
    tickers = ["AAPL", "MSFT", "GOOG"]
    for tk in tickers:
        fetch_stock_data(tk, period="1y")