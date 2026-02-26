

# import re
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier

# # Step 1: Sample dataset of emails (1 = valid, 0 = invalid)
# emails = [
#     "john.doe@gmail.com",      # valid
#     "alice123@company.org",    # valid
#     "invalid-email@",          # invalid
#     "bob@domain",              # invalid
#     "user!@gmail.com"          # invalid
# ]
# labels = [1, 1, 0, 0, 0]

# # Step 2: Feature extraction function
# def extract_features(email):
#     return [
#         len(email),                                 # email length
#         int(bool(re.search(r'\d', email))),         # contains number
#         int(bool(re.search(r'[!#$%^&*(),?":{}|<>]', email))),  # special char
#         int('@' in email),                          # has @
#         int('.' in email),                          # has dot
#         len(email.split('@')[-1].split('.')),      # number of domain parts
#     ]

# # Convert emails to feature array
# X = np.array([extract_features(email) for email in emails])
# y = np.array(labels)

# # Step 3: Train ML model
# model = RandomForestClassifier(random_state=42)
# model.fit(X, y)

# # Step 4: Validate new emails
# test_emails = [
#     "example.user@gmail.com",
#     "noatsign.com",
#     "hello_world@domain.org"
# ]

# for email in test_emails:
#     features = np.array([extract_features(email)])
#     prediction = model.predict(features)
#     if prediction[0] == 1:
#         print(f" '{email}' is a Valid Email")
#     else:
#         print(f" '{email}' is an Invalid Email")

# ===================================================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Prepare dataset (ML features)
# -------------------------------
# Features:
# length_correct: 1 if length > 5
# has_at: 1 if '@' present
# has_dot: 1 if '.' present after '@'
# valid: 1 = normal/correct, 0 = suspicious/wrong

data = {
    "length_correct": [1,1,0,1,1,0,1,1],
    "has_at":         [1,1,1,0,1,0,1,1],
    "has_dot":        [1,0,0,0,1,0,0,1],
    "valid":          [1,0,0,0,1,0,0,1]
}

df = pd.DataFrame(data)

X = df[["length_correct", "has_at", "has_dot"]]
y = df["valid"]

# length_correct → 1 if email is long enough (>5 characters), 0 if too short
# has_at → 1 if email contains @, 0 if missing
# has_dot → 1 if email contains . after @, 0 if missing
# valid → 1 = normal / correct email pattern, 0 = suspicious / wrong

# | Row | length_correct | has_at | has_dot | valid | Meaning                                                         |
# | --- | -------------- | ------ | ------- | ----- | --------------------------------------------------------------- |
# | 1   | 1              | 1      | 1       | 1     | Normal email: long enough, contains `@`, contains `.` → valid |
# | 2   | 1              | 1      | 0       | 0     | Suspicious: has `@` but missing `.` → invalid                 |
# | 3   | 0              | 1      | 0       | 0     | Suspicious: too short, has `@` but missing `.` → invalid      |
# | 4   | 1              | 0      | 0       | 0     | Suspicious: no `@`, no `.` → invalid                          |
# | 5   | 1              | 1      | 1       | 1     | Normal email: correct format → valid                          |
# | 6   | 0              | 0      | 0       | 0     | Suspicious: too short, no `@`, no `.` → invalid               |
# | 7   | 1              | 1      | 0       | 0     | Suspicious: has `@` but missing `.` → invalid                 |
# | 8   | 1              | 1      | 1       | 1     | Normal email: long enough, correct format → valid             |


# -------------------------------
# Step 2: Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# -------------------------------
# Step 3: Train ML model
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"ML model test accuracy: {accuracy*100:.2f}%")

# -------------------------------
# Step 4: Predict new email
# -------------------------------
def extract_features(email_input):
    length_correct = int(len(email_input) > 5)
    has_at = int('@' in email_input)
    has_dot = int('.' in email_input.split('@')[-1]) if has_at else 0
    return [length_correct, has_at, has_dot]

max_attempts = 3
attempt = 0

while attempt < max_attempts:
    email_input = input("Enter your email: ")
    features = pd.DataFrame([extract_features(email_input)], columns=["length_correct","has_at","has_dot"])
    prediction = model.predict(features)

    if prediction[0] == 1:
        print("Email looks valid (ML prediction)!")
        break
    else:
        print("Suspicious email! Check format.")
        attempt += 1
        if attempt < max_attempts:
            print(f"Attempts left: {max_attempts - attempt}")
        else:
            print("⚠ Maximum attempts reached. Try again later.")