import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
 
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
 
 
# ======================================================
# STEP 1: CREATE CUSTOMER DATASET
# ======================================================
 
np.random.seed(42)
 
data = {
    "Age": np.random.randint(18, 65, 200),
    "Annual_Income": np.random.randint(20000, 150000, 200),
    "Purchase_Frequency": np.random.randint(1, 30, 200),
    "Website_Visits": np.random.randint(5, 100, 200),
}
 
df = pd.DataFrame(data)
 
df["Spending_Score"] = (
    df["Annual_Income"] * 0.0003 +
    df["Purchase_Frequency"] * 2 +
    df["Website_Visits"] * 0.5 +
    np.random.randint(1, 30, 200)
)
 
df["High_Value_Customer"] = np.where(df["Spending_Score"] > df["Spending_Score"].median(), 1, 0)
 
print("Dataset:")
print(df.head())
 
 
# ======================================================
# STEP 2: FEATURE SCALING
# ======================================================
 
features = ["Age", "Annual_Income", "Purchase_Frequency", "Website_Visits"]
 
X = df[features]
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
 
# ======================================================
# STEP 3: K-MEANS CLUSTERING
# ======================================================
 
kmeans = KMeans(n_clusters=3, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
 
print("\nK-Means Cluster Output:")
print(df[["Age", "Annual_Income", "Spending_Score", "KMeans_Cluster"]].head())
 
 
# ======================================================
# STEP 4: ELBOW METHOD
# ======================================================
 
wcss = []
 
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    wcss.append(model.inertia_)
 
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
 
 
# ======================================================
# STEP 5: HIERARCHICAL CLUSTERING
# ======================================================
 
hierarchical = AgglomerativeClustering(n_clusters=3)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)
 
print("\nHierarchical Cluster Output:")
print(df[["Age", "Annual_Income", "Hierarchical_Cluster"]].head())
 
 
# Dendrogram
linked = linkage(X_scaled[:30], method="ward")
 
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()
 
 
# ======================================================
# STEP 6: DBSCAN CLUSTERING
# ======================================================
 
dbscan = DBSCAN(eps=1.5, min_samples=5)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)
 
print("\nDBSCAN Cluster Output:")
print(df[["Age", "Annual_Income", "DBSCAN_Cluster"]].head())
 
print("\nDBSCAN Cluster Counts:")
print(df["DBSCAN_Cluster"].value_counts())
 
 
# ======================================================
# STEP 7: REGRESSION MODEL
# Predict Spending Score
# ======================================================
 
X_reg = df[features]
y_reg = df["Spending_Score"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
 
# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
 
print("\nRidge Coefficients:")
print(ridge.coef_)
 
 
# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)
 
print("\nLasso Coefficients:")
print(lasso.coef_)
 
 
# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
elastic_pred = elastic.predict(X_test_scaled)
 
print("\nElastic Net Coefficients:")
print(elastic.coef_)
 
 
# ======================================================
# STEP 8: GRID SEARCH FOR RIDGE
# ======================================================
 
ridge_params = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}
 
grid = GridSearchCV(Ridge(), ridge_params, cv=5)
grid.fit(X_train_scaled, y_train)
 
print("\nBest Ridge Parameter using Grid Search:")
print(grid.best_params_)
 
 
# ======================================================
# STEP 9: CLASSIFICATION MODEL
# Predict High Value Customer
# ======================================================
 
X_cls = df[features]
y_cls = df["High_Value_Customer"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)
 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_scaled, y_train)
 
y_pred = classifier.predict(X_test_scaled)
 
 
# ======================================================
# STEP 10: MODEL EVALUATION
# ======================================================
 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
 
print("\nClassification Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
 
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
 
 
# ======================================================
# STEP 11: RANDOM SEARCH
# ======================================================
 
random_params = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10]
}
 
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    random_params,
    n_iter=5,
    cv=5,
    random_state=42
)
 
random_search.fit(X_train_scaled, y_train)
 
print("\nBest Random Forest Parameters using Random Search:")
print(random_search.best_params_)
 
 
# ======================================================
# STEP 12: FINAL CUSTOMER PREDICTION
# ======================================================
 
new_customer = pd.DataFrame({
    "Age": [28],
    "Annual_Income": [85000],
    "Purchase_Frequency": [18],
    "Website_Visits": [60]
})
 
new_customer_scaled = scaler.transform(new_customer)
 
prediction = random_search.best_estimator_.predict(new_customer_scaled)
 
if prediction[0] == 1:
    print("\nFinal Output: This is a High Value Customer")
else:
    print("\nFinal Output: This is a Normal Customer")