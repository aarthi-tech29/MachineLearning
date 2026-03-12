# ===========================Train the model pkl===================================
# train_aqi_model.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# -------------------------
# 1. Load historical AQI data
# -------------------------
df = pd.read_csv(
    r"C:\Users\ADMIN\Desktop\India_AQI_Bulletins_Master.csv",  # replace with your path
    low_memory=False
)

# -------------------------
# 2. Clean column names
# -------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
# This converts: "Index Value" -> "index_value"

# -------------------------
# 3. Convert numeric AQI
# -------------------------
df['index_value'] = pd.to_numeric(df['index_value'], errors='coerce')
df = df.dropna(subset=['index_value'])

# -------------------------
# 4. Define target labels
# -------------------------
def aqi_to_label(aqi):
    if aqi <= 50: return "Low"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "High"
    elif aqi <= 200: return "Very High"
    else: return "Hazardous"

df['risk_level'] = df['index_value'].apply(aqi_to_label)

# -------------------------
# 5. Select features and target
# -------------------------
# Using only numeric AQI as feature
X = df[['index_value']]
y = df['risk_level']

# -------------------------
# 6. Scale features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 7. Train Random Forest
# -------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

# -------------------------
# 8. Save model and scaler
# -------------------------
joblib.dump(model, "aqi_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Training completed. Model and scaler saved as .pkl files.")