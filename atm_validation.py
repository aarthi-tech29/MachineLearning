
# import re
# import random
# import time
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier

# # -------------------------------
# # Step 1: Account Validation Function
# # -------------------------------
# def is_valid_account(account_number):
#     """
#     Check if account number is 10 digits
#     """
#     return bool(re.match(r'^\d{10}$', account_number))

# # -------------------------------
# # Step 2: PIN Validation Function
# # -------------------------------
# def is_valid_pin(pin):
#     """
#     Check if PIN is exactly 4 digits
#     """
#     return bool(re.match(r'^\d{4}$', pin))

# # -------------------------------
# # Step 3: ML OTP/PIN Validation (Optional)
# # -------------------------------
# def extract_features(pin_input, correct_pin, time_taken):
#     """
#     Features:
#     - Length correct
#     - All digits
#     - Match
#     - Time taken to enter
#     """
#     return [
#         int(len(pin_input) == len(correct_pin)),
#         int(pin_input.isdigit()),
#         int(pin_input == correct_pin),
#         time_taken
#     ]

# # Sample ML training data
# X = [
#     [1,1,1,5],   # valid PIN entered quickly
#     [1,1,0,6],   # wrong PIN
#     [0,1,0,10],  # wrong length
#     [1,0,0,7],   # has non-digit
# ]
# y = [1,0,0,0]  # 1 = valid, 0 = invalid

# ml_model = RandomForestClassifier(random_state=42)
# ml_model.fit(X, y)

# # -------------------------------
# # Step 4: ATM Validation Process
# # -------------------------------
# def atm_validation():
#     account_number = input("Enter your 10-digit account number: ")
#     if not is_valid_account(account_number):
#         print("Invalid account number.")
#         return
    
#     correct_pin = "1234"  # For demo; in real ATMs, stored securely
#     pin_input = input("Enter your 4-digit PIN: ")
#     start_time = time.time()
#     time_taken = int(time.time() - start_time)

#     if not is_valid_pin(pin_input):
#         print("Invalid PIN format.")
#         return

#     # ML validation
#     features = np.array([extract_features(pin_input, correct_pin, time_taken)])
#     prediction = ml_model.predict(features)
    
#     if prediction[0] == 1:
#         print("PIN Verified! Access granted.")
#     else:
#         print("Incorrect PIN. Access denied.")

# # -------------------------------
# # Step 5: Run ATM validation
# # -------------------------------
# if __name__ == "__main__":
#     atm_validation()

# =========================================================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Prepare dataset for ML
# -------------------------------
data = {
    "length_correct": [1, 1, 0, 1, 1, 0, 1, 1],
    "all_digits":     [1, 1, 1, 0, 1, 0, 1, 1],
    "match":          [1, 0, 0, 0, 1, 0, 0, 1],
    "valid":          [1, 0, 0, 0, 1, 0, 0, 1]
}

# length_correct → 1 if PIN has exactly 4 digits, 0 if not
# all_digits → 1 if PIN contains only numbers, 0 if letters or symbols are included
# match → 1 if PIN entered matches the actual ATM PIN, 0 if incorrect
# valid → 1 = valid PIN, 0 = invalid PIN

# | Row | length_correct | all_digits | match | valid | Meaning                                                                                           |
# | --- | -------------- | ---------- | ----- | ----- | ------------------------------------------------------------------------------------------------- |
# | 1   | 1              | 1          | 1     | 1     | Correct PIN: 4 digits, all numbers, matches actual PIN → valid                                  |
# | 2   | 1              | 1          | 0     | 0     | Wrong PIN: correct length & digits but does **not match** → invalid                             |
# | 3   | 0              | 1          | 0     | 0     | Wrong PIN: wrong length (less/more than 4 digits), digits correct, does **not match** → invalid |
# | 4   | 1              | 0          | 0     | 0     | Wrong PIN: 4 characters but contains non-digits, does **not match** → invalid                   |
# | 5   | 1              | 1          | 1     | 1     | Correct PIN: 4 digits, all numbers, matches actual PIN → valid                                  |
# | 6   | 0              | 0          | 0     | 0     | Wrong PIN: wrong length **and** contains non-digits → invalid                                   |
# | 7   | 1              | 1          | 0     | 0     | Wrong PIN: 4 digits, all numbers but does **not match** → invalid                               |
# | 8   | 1              | 1          | 1     | 1     | Correct PIN: 4 digits, all numbers, matches actual PIN → valid                                  |


df = pd.DataFrame(data)
X = df[["length_correct", "all_digits", "match"]]
y = df["valid"]

# -------------------------------
# Step 2: Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# -------------------------------
# Step 3: Train ML model
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test model accuracy
accuracy = model.score(X_test, y_test)
print(f"ML model test accuracy: {accuracy*100:.2f}%")

# -------------------------------
# Step 4: ATM PIN input process
# -------------------------------
correct_pin = "1234"
max_attempts = 3
attempt = 0

while attempt < max_attempts:
    user_pin = input("Enter your 4-digit ATM PIN: ")

    # Feature extraction
    length_correct = int(len(user_pin) == 4)
    all_digits = int(user_pin.isdigit())
    match = int(user_pin == correct_pin)

    features = pd.DataFrame([[length_correct, all_digits, match]],
                            columns=["length_correct","all_digits","match"])
    prediction = model.predict(features)

    if prediction[0] == 1:
        print("PIN Verified Successfully!")
        break
    else:
        print("Invalid PIN. Try again!")
        attempt += 1
        if attempt < max_attempts:
            print(f"Attempts left: {max_attempts - attempt}")
        else:
            print("⚠ Maximum attempts reached. Access blocked!")