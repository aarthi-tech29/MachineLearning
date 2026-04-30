import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# pandas → used to handle dataset (tables)
# RandomForestClassifier → ML algorithm
# train_test_split → splits data into train & test
# accuracy_score → checks performance

# -------------------------
# 1. Load Data
# -------------------------
df = pd.read_csv("stock_data.csv")

# -------------------------
# 2. Create Moving Averages
# -------------------------
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()

# Using Moving Average
# MA5 → average of last 5 days
# MA10 → average of last 10 days
# Helps model understand trend

# -------------------------
# 3. Create Bullish/Bearish Feature
# -------------------------
# 1 = Bullish (MA5 > MA10)-uptrend
# 0 = Bearish (MA5 < MA10)-downtrend
df['Trend'] = (df['MA5'] > df['MA10']).astype(int)

# -------------------------
# 4. Create Target
# -------------------------
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# shift(-1) → looks at next day price
# If next day price > today → 1 (UP 📈)
# Else → 0 (DOWN 📉)
# This converts problem into classification

# -------------------------
# 5. Remove NaN rows
# -------------------------
df = df.dropna()

# Moving average creates empty values (NaN)
# Model cannot handle missing values

# -------------------------
# 6. Features & Labels
# -------------------------
X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'Trend']]
y = df['Target']

# X → input features (what model sees)
# y → output (what model predicts)

# -------------------------
# 7. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Splits data:
# 80% → training
# 20% → testing
# shuffle=False is important:
# Keeps time order (very important for stock data)

# -------------------------
# 8. Model
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Creates 100 decision trees
# Learns patterns from training data

# -------------------------
# 9. Prediction
# -------------------------
y_pred = model.predict(X_test)

# UP or DOWN for test data

# -------------------------
# 10. Accuracy
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compares:actual vs predicted

# -------------------------
# 11. Predict New Data
# -------------------------
# IMPORTANT: include MA values
new_data = pd.DataFrame(
    [[135, 140, 134, 138, 26000, 132, 125, 1]],
    columns=['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'Trend']
)

prediction = model.predict(new_data)
# You give new stock data:
# Open, High, Low, Close, Volume, MA5, MA10
# Model predicts next day trend
# Open - Opening price of the stock for the day
# High - Highest price of the stock for the day
# Low - Lowest price of the stock for the day
# Close - Closing price of the stock for the day
# Volume - Number of shares traded during the day
# MA5 (5-day Moving Average) - Average closing price of the last 5 days
# MA10 (10-day Moving Average) - Average closing price of the last 10 days

if prediction[0] == 1:
    print("Stock will go UP 📈")
else:
    print("Stock will go DOWN 📉")