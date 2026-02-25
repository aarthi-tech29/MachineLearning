import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Create dataset (Historical sales)
data = {
    "Month_Number": [1, 2, 3, 4, 5, 6, 7, 8],
    "Sales_Units": [200, 250, 300, 350, 400, 420, 480, 520]
}

df = pd.DataFrame(data)

# Step 2: Define input (X) and output (Y)
X = df[["Month_Number"]]
Y = df["Sales_Units"]

# Step 3: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, Y_train)

# Step 5: Predict future demand (Month 10)
future_month = pd.DataFrame([[10]], columns=["Month_Number"])
predicted_sales = model.predict(future_month)

print("Predicted Sales for Month 10:", predicted_sales[0])