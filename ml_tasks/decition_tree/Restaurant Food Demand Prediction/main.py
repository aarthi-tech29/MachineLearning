# =========================================================
# RESTAURANT FOOD DEMAND PREDICTION
# USING DECISION TREE
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# LOAD CSV DATASET
# =========================================================

df = pd.read_csv("restaurant_food_demand_dataset.csv")

print("\nDATASET")
print("------------------------------------------------")
print(df.head())

# =========================================================
# LABEL ENCODING
# =========================================================

season_encoder = LabelEncoder()
festival_encoder = LabelEncoder()

df["Season"] = season_encoder.fit_transform(df["Season"])
df["Festival_Day"] = festival_encoder.fit_transform(df["Festival_Day"])

# =========================================================
# INPUT FEATURES AND TARGET
# =========================================================

X = df[[
    "Season",
    "Festival_Day",
    "Customer_Traffic", # customers are expected to visit the restaurant.
    "Average_Temperature" # It is used because weather affects food demand.
]]
# | Customer Traffic | Meaning      |
# | ---------------- | ------------ |
# | 100              | Low crowd    |
# | 250              | Medium crowd |
# | 500              | Heavy crowd  |

# | Temperature | Possible Demand        |
# | ----------- | ---------------------- |
# | 18°C        | Low cold-season demand |
# | 28°C        | Moderate demand        |
# | 39°C        | High summer demand     |


y = df["Food_Demand"]

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
# DECISION TREE MODEL
# =========================================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
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

print("\nENTER RESTAURANT DETAILS")
print("------------------------------------------------")

season = input("Enter Season (Summer/Winter/Rainy): ")
festival = input("Festival Day? (Yes/No): ")
traffic = int(input("Enter Customer Traffic: "))
temperature = float(input("Enter Average Temperature: "))

# =========================================================
# ENCODE INPUTS
# =========================================================

season_encoded = season_encoder.transform([season])[0]
festival_encoded = festival_encoder.transform([festival])[0]

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

restaurant_data = pd.DataFrame({
    "Season": [season_encoded],
    "Festival_Day": [festival_encoded],
    "Customer_Traffic": [traffic],
    "Average_Temperature": [temperature]
})

# =========================================================
# FOOD DEMAND PREDICTION
# =========================================================

prediction = model.predict(restaurant_data)

print("\nFOOD DEMAND PREDICTION")
print("------------------------------------------------")
print("Predicted Demand:", prediction[0])

# =========================================================
# INVENTORY PLANNING DASHBOARD
# =========================================================

print("\nINVENTORY PLANNING DASHBOARD")
print("------------------------------------------------")

if prediction[0] == "High":
    print("Increase food inventory and staff preparation.")
elif prediction[0] == "Medium":
    print("Maintain moderate inventory levels.")
else:
    print("Reduce inventory to avoid food wastage.")

# =========================================================
# DECISION TREE VISUALIZATION
# =========================================================

plt.figure(figsize=(12, 10))

plot_tree(
    model,
    feature_names=[
        "Season",
        "Festival Day",
        "Customer Traffic",
        "Average Temperature"
    ],
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Restaurant Food Demand Prediction Decision Tree")

plt.show()
# =========================================================

# The model learns:
# - The decision tree model can predict food demand based on season, festival days, customer traffic, and temperature.
# - The accuracy and classification report provide insights into the model's performance.
# - The user input section allows restaurant managers to get real-time demand predictions for better inventory planning.
# - The decision tree visualization helps understand the decision-making process and the key factors influencing food demand.
# - The inventory planning dashboard provides actionable insights based on the predicted demand, helping to optimize inventory levels and reduce wastage.

# The Input example:
# High
# Enter Season (Summer/Winter/Rainy): Summer
# Festival Day? (Yes/No): Yes
# Enter Customer Traffic: 480
# Enter Average Temperature: 39

# Low
# Enter Season (Summer/Winter/Rainy): Winter
# Festival Day? (Yes/No): No
# Enter Customer Traffic: 170
# Enter Average Temperature: 20