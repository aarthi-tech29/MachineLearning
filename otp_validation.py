
# import random
# import string
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# import time

# # -------------------------------
# # Step 1: Generate OTP
# # -------------------------------
# def generate_otp(length=6):
#     digits = string.digits
#     otp = "".join(random.choice(digits) for _ in range(length))
#     return otp

# # -------------------------------
# # Step 2: Feature Extraction for OTP Validation
# # -------------------------------
# def extract_features(otp_input, otp_generated, time_taken):
#     """
#     Converts OTP input into ML features
#     - Length matches?
#     - Only digits?
#     - Number of wrong attempts?
#     - Time taken to enter OTP
#     """
#     return [
#         int(len(otp_input) == len(otp_generated)),   # length correct
#         int(otp_input.isdigit()),                    # all digits?
#         int(otp_input == otp_generated),            # exact match
#         time_taken                                  # time taken to enter
#     ]

# # -------------------------------
# # Step 3: Prepare Training Data for ML Model
# # -------------------------------
# # Features: [length_correct, all_digits, match, time_taken]
# X = [
#     [1,1,1,5],   # valid OTP, entered quickly
#     [1,1,0,6],   # wrong OTP, but correct format
#     [0,1,0,10],  # wrong length
#     [1,0,0,7],   # has non-digit
# ]
# y = [1,0,0,0]  # 1 = valid OTP, 0 = invalid

# model = RandomForestClassifier(random_state=42)
# model.fit(X, y)

# # -------------------------------
# # Step 4: OTP Process with ML Validation
# # -------------------------------
# otp = generate_otp()
# print(f"Your OTP is: {otp} (for demo purposes)")

# start_time = time.time()
# user_input = input("Enter OTP: ")
# time_taken = int(time.time() - start_time)

# features = np.array([extract_features(user_input, otp, time_taken)])
# prediction = model.predict(features)

# if prediction[0] == 1:
#     print("OTP Verified Successfully!")
# else:
#     print("Invalid OTP. Try again!")

# ===================================================================
import random
import string
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Generate OTP
# -------------------------------
def generate_otp(length=6):
    digits = string.digits
    otp = ''.join(random.choice(digits) for _ in range(length))
    return otp

# -------------------------------
# Step 2: Extract features for ML
# -------------------------------
def extract_features(otp_input, otp_generated, time_taken):
    """
    Features used for ML analysis (not for actual OTP verification):
    - Length correct
    - All digits
    - Exact match
    - Time taken
    """
    return [
        int(len(otp_input) == len(otp_generated)),  # length correct
        int(otp_input.isdigit()),                   # all digits
        int(otp_input == otp_generated),           # exact match
        time_taken
    ]

# -------------------------------
# Step 3: Prepare ML dataset
# -------------------------------
data = {
    "length_correct": [1,1,0,1,1,0,1,1],
    "all_digits":     [1,1,1,0,1,0,1,1],
    "match":          [1,0,0,0,1,0,0,1],
    "time_taken":     [5,6,10,7,4,12,8,3],
    "valid":          [1,0,0,0,1,0,0,1]  # Only for ML monitoring
}

# created 8 different examples to show different OTP behaviors:
# Correct OTP, entered quickly
# Wrong OTP, correct format
# Wrong length
# Contains letters instead of digits
# Correct OTP, fast
# Wrong length and letters, slow
# Correct length/digits, wrong OTP, slower
# Correct OTP, very fast

# | Row | length_correct | all_digits | match | time_taken | valid |
# | --- | -------------- | ---------- | ----- | ---------- | ----- |
# | 1   | 1              | 1          | 1     | 5          | 1     |
# | 2   | 1              | 1          | 0     | 6          | 0     |
# | 3   | 0              | 1          | 0     | 10         | 0     |
# | 4   | 1              | 0          | 0     | 7          | 0     |
# | 5   | 1              | 1          | 1     | 4          | 1     |
# | 6   | 0              | 0          | 0     | 12         | 0     |
# | 7   | 1              | 1          | 0     | 8          | 0     |
# | 8   | 1              | 1          | 1     | 3          | 1     |


df = pd.DataFrame(data)
X = df[["length_correct", "all_digits", "match", "time_taken"]]
y = df["valid"]

# -------------------------------
# Step 4: Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# -------------------------------
# Step 5: Train ML model
# -------------------------------
ml_model = RandomForestClassifier(random_state=42)
ml_model.fit(X_train, y_train)
accuracy = ml_model.score(X_test, y_test)
print(f"ML model test accuracy (for monitoring patterns): {accuracy*100:.2f}%")

# -------------------------------
# Step 6: OTP process
# -------------------------------
otp = generate_otp()
print(f"Your OTP is: {otp} (for demo purposes)")

max_attempts = 3
attempt = 0

while attempt < max_attempts:
    start_time = time.time()
    user_input = input("Enter OTP: ")
    time_taken = int(time.time() - start_time)
    
    # -------------------------------
    # Step 6a: Direct OTP verification (100% reliable)
    # -------------------------------
    if user_input == otp:
        print(f" OTP Verified Successfully in {time_taken} seconds!")
        break
    else:
        attempt += 1
        print(f" Invalid OTP. Attempts left: {max_attempts - attempt}")
    
    # -------------------------------
    # Step 6b: ML Monitoring (optional)
    # -------------------------------
    features = pd.DataFrame(
        [extract_features(user_input, otp, time_taken)],
        columns=["length_correct", "all_digits", "match", "time_taken"]
    )
    ml_prediction = ml_model.predict(features)
    if ml_prediction[0] == 1 and user_input != otp:
        print("⚠ Warning: suspicious OTP pattern detected!")

if attempt == max_attempts:
    print("You have exceeded the maximum OTP attempts. Try again later!")