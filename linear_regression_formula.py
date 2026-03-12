# # Import libraries
# import numpy as np
# from sklearn.linear_model import LinearRegression

# # Step 1: Prepare data
# experience = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
# salary = np.array([20000,26000,30000,32000,40000,45000,53000,55000,60000,62000])

# # Step 2: Train Linear Regression on all data (automatic calculation of m and c)
# model = LinearRegression()
# model.fit(experience, salary)

# # Step 3: Get automatically calculated slope (m) and intercept (c)
# m = model.coef_[0]
# c = model.intercept_
# print("Automatically calculated increment (m):", round(m,2))
# print("Automatically calculated starting salary (c):", round(c,2))

# # Step 4: Split data for demonstration (train/test) — no effect on m and c
# X_train = experience[:8]  # first 8 points as "train"
# X_test = experience[8:]   # last 2 points as "test"

# # Step 5: Predict salaries
# train_pred = model.predict(X_train)
# test_pred = model.predict(X_test)

# # Step 6: Show predictions
# print("\nTrain set predictions:")
# for exp, sal in zip(X_train, train_pred):
#     print(f"Experience {exp[0]} years -> Predicted Salary = {round(sal,2)}")

# print("\nTest set predictions:")
# for exp, sal in zip(X_test, test_pred):
#     print(f"Experience {exp[0]} years -> Predicted Salary = {round(sal,2)}")
# =============================================================================
import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([20000,26000,30000,32000,40000,45000,53000,55000,60000,62000])

# Step 1: Calculate slope (m) and intercept (c) from first and last points
m = (y[-1] - y[0]) / (X[-1][0] - X[0][0])
c = y[0] - m*X[0][0]

# Step 2: Create LinearRegression object (ML style)
model = LinearRegression()
model.fit(X, y)  # fit to use model object
model.coef_ = np.array([m])
model.intercept_ = c

# Step 3: Split for train/test demonstration
X_train, X_test = X[:8], X[8:]
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Step 4: Show predictions
print("\nTrain set predictions:")
for exp, sal in zip(X_train, train_pred):
    print(f"Experience {exp[0]} years -> Predicted Salary = {round(sal,2)}")

print("\nTest set predictions:")
for exp, sal in zip(X_test, test_pred):
    print(f"Experience {exp[0]} years -> Predicted Salary = {round(sal,2)}")