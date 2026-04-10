import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load dataset
data = pd.read_csv("salary_data.csv")

# Define feature and target
X = data[['YearsExperience']]
y = data['Salary']
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
# Create model
model = LinearRegression()
# Train model
model.fit(X_train, y_train)
# Model parameters
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
# Predictions
y_pred = model.predict(X_test)
# Compare actual vs predicted
results = pd.DataFrame({
 'Actual': y_test.values,
 'Predicted': y_pred
})
print(results)
# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
# Visualization

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Fit")
plt.show()
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
# Load dataset
data = pd.read_csv("salary_data.csv")
 
# Define feature and target
X = data[['YearsExperience']]
y = data['Salary']
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# Create model
model = LinearRegression()
 
# Train model
model.fit(X_train, y_train)
 
# Model parameters
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
 
# Predictions on test data
y_pred = model.predict(X_test)
 
# Compare actual vs predicted
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
print(results)
 
# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
 
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
 
# User input
experience = float(input("Enter Years of Experience: "))
 
# Convert user input into 2D array
user_input = np.array([[experience]])
 
# Predict salary
predicted_salary = model.predict(user_input)
 
# Show result
print("Predicted Salary for", experience, "years experience is:", predicted_salary[0])
 
# Visualization
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Fit")
plt.show()

# ===============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
# Load dataset
data = pd.read_csv("salary_data.csv")
 
# -------------------------------
#  STEP 1: CHECK NULL VALUES
# -------------------------------
print("Null Values Before Cleaning:\n", data.isnull().sum())
 
# -------------------------------
#  STEP 2: REMOVE NULL VALUES
# -------------------------------
data = data.dropna()
 
# -------------------------------
#  STEP 3: VERIFY CLEANING
# -------------------------------
print("Null Values After Cleaning:\n", data.isnull().sum())
 
# Define feature and target
X = data[['YearsExperience']]
y = data['Salary']
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# Create model
model = LinearRegression()
 
# Train model
model.fit(X_train, y_train)
 
# Model parameters
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
 
# Predictions on test data
y_pred = model.predict(X_test)
 
# Compare actual vs predicted
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
print(results)
 
# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
 
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
 
# -------------------------------
#  USER INPUT
# -------------------------------
experience = float(input("Enter Years of Experience: "))
 
# Convert user input into 2D array
user_input = np.array([[experience]])
 
# Predict salary
predicted_salary = model.predict(user_input)
 
print("Predicted Salary for", experience, "years experience is:", predicted_salary[0])
 
# -------------------------------
#  VISUALIZATION
# -------------------------------
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Fit")
plt.show()
# ================================================================