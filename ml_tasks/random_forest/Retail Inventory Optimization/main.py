# =========================================================
# RETAIL INVENTORY OPTIMIZATION
# USING RANDOM FOREST
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# LOAD CSV DATASET
# =========================================================

df = pd.read_csv("retail_inventory_dataset.csv")

print("\nDATASET")
print("------------------------------------------------")
print(df.head())

# =========================================================
# LABEL ENCODING
# =========================================================

festival_encoder = LabelEncoder()
season_encoder = LabelEncoder()

df["Festival_Season"] = festival_encoder.fit_transform(
    df["Festival_Season"]
)

df["Season"] = season_encoder.fit_transform(
    df["Season"]
)

# =========================================================
# INPUT FEATURES AND TARGET
# =========================================================

X = df[[
    "Previous_Sales",
    "Festival_Season",
    "Season",
    "Current_Stock"
]]

y = df["Inventory_Need"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=5,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print("------------------------------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nCLASSIFICATION REPORT")
print("------------------------------------------------")
print(classification_report(y_test, y_pred))

# =========================================================
# USER INPUT
# =========================================================

print("\nENTER PRODUCT DETAILS")
print("------------------------------------------------")

sales = int(input("Enter Previous Sales: "))
festival = input("Festival Season? (Yes/No): ")
season = input("Enter Season (Summer/Winter/Rainy/Festival): ")
stock = int(input("Enter Current Stock: "))

# =========================================================
# ENCODE INPUT
# =========================================================

festival_encoded = festival_encoder.transform([festival])[0]
season_encoded = season_encoder.transform([season])[0]

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

product_data = pd.DataFrame({
    "Previous_Sales": [sales],
    "Festival_Season": [festival_encoded],
    "Season": [season_encoded],
    "Current_Stock": [stock]
})

# =========================================================
# INVENTORY PREDICTION
# =========================================================

prediction = model.predict(product_data)

print("\nINVENTORY OPTIMIZATION PREDICTION")
print("------------------------------------------------")
print("Predicted Inventory Need:", prediction[0])

# =========================================================
# AUTOMATED REORDER ALERTS
# =========================================================

print("\nAUTOMATED REORDER ALERT")
print("------------------------------------------------")

if prediction[0] == "High":
    print("Immediate stock reorder required.")
    print("Increase inventory levels.")
elif prediction[0] == "Medium":
    print("Maintain moderate stock levels.")
else:
    print("Current inventory is sufficient.")

# =========================================================
# FEATURE IMPORTANCE VISUALIZATION
# =========================================================

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(8, 5))

plt.bar(features, importance)

plt.title("Feature Importance in Inventory Optimization")

plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.show()
# ========================================================
# The model learns:
# - Previous sales data, festival seasons, and current stock levels are key factors in predicting inventory needs.
# - The model can help retailers optimize inventory levels, reduce stockouts, and minimize overstock situations.
# - Automated reorder alerts can assist in timely restocking, improving customer satisfaction and sales.
# - The feature importance visualization highlights which factors have the most influence on inventory predictions, guiding retailers in decision-making.

# The Input example:
# Inventory Need: High
# Enter Previous Sales: 600
# Festival Season? (Yes/No): Yes
# Enter Season (Summer/Winter/Rainy/Festival): Festival
# Enter Current Stock: 30

# Inventory Need: Low
# Enter Previous Sales: 140
# Festival Season? (Yes/No): No
# Enter Season (Summer/Winter/Rainy/Festival): Summer
# Enter Current Stock: 80