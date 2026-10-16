import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Collect historical stock data
data = yf.download("AAPL", period="5y", auto_adjust=True)

# Step 2: Select the closing price
data = data[["Close"]].dropna()

# Step 3: Create previous-day price features
data["Previous_Close"] = data["Close"].shift(1)
data = data.dropna()

X = data[["Previous_Close"]]
y = data["Close"]

# Step 4: Split data chronologically
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict stock prices
predictions = model.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("R-squared Score:", round(r2, 4))

# Step 8: Visualize actual and predicted prices
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values,
         label="Actual Price")
plt.plot(y_test.index, predictions,
         label="Predicted Price")

plt.title("Stock Price Prediction Using Linear Regression")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
