import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA
# =========================
customers = pd.read_csv("customer_history.csv")
products = pd.read_csv("products.csv")

# =========================
# CUSTOMER BEHAVIOR PROFILE
# =========================
customer_profile = customers.groupby("CustomerID").agg({
    "Quantity": "sum",
    "CartClicks": "sum",
    "TimeSpent": "sum",
    "PurchaseCount": "sum"
}).reset_index()

# =========================
# FEATURE MATRIX
# =========================
X = customer_profile[[
    "Quantity",
    "CartClicks",
    "TimeSpent",
    "PurchaseCount"
]]

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN KNN MODEL
# =========================
model = NearestNeighbors(n_neighbors=3, metric="cosine")
model.fit(X_scaled)

# =========================
# RECOMMENDATION SYSTEM
# =========================
def recommend(customer_id):

    idx = customer_profile[customer_profile["CustomerID"] == customer_id].index[0]

    user_vector = X_scaled[idx].reshape(1, -1)

    distances, indices = model.kneighbors(user_vector)

    similar_customers = customer_profile.iloc[indices[0]]["CustomerID"].values

    # products already bought
    seen_products = set(customers[customers["CustomerID"] == customer_id]["ProductID"])

    # candidate products from similar customers
    candidate_products = customers[customers["CustomerID"].isin(similar_customers)]["ProductID"].values

    print("\nSMART SHOPPING RECOMMENDATION")
    print("-----------------------------------")
    print("Customer ID:", customer_id)

    print("\nSimilar Customers:")
    for c in similar_customers:
        print("-", c)

    print("\nRecommended Products:")

    printed = set()

    for product in candidate_products:

        if product not in seen_products and product not in printed:

            product_info = products[products["ProductID"] == product].iloc[0]

            print("-", product_info["ProductName"], "| ₹", product_info["Price"])

            printed.add(product)

    # =========================
    # PERSONALIZED OFFERS
    # =========================
    print("\nPersonalized Offers:")

    total_purchase = customers[customers["CustomerID"] == customer_id]["PurchaseCount"].sum()

    if total_purchase >= 5:
        print("20% OFF on Electronics")
    elif total_purchase >= 3:
        print("10% OFF on Fashion Items")
    else:
        print("Free Delivery on next order")

# =========================
# RUN SYSTEM
# =========================
cid = input("Enter Customer ID: ")
recommend(cid)
# =====================================================

# The model learns:
# - How to find similar customers based on their shopping behavior.
# - How to recommend products that similar customers have bought but the current customer hasn't.
# - How to provide personalized offers based on purchase history.

# Input Example:
# CustomerID: C1
