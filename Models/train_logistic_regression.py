import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from feature_engineering import compute_technical_indicators

# Load and process data
df = pd.read_csv("data/AAPL_data.csv")
df = compute_technical_indicators(df)
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD', 'Signal']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Save model
with open("models/logistic_model.pkl", "wb") as f:
    pickle.dump(lr, f)

print("✅ Logistic Regression model saved as models/logistic_model.pkl")
