import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Create transaction dataset
transactions = [
    ["Milk", "Bread", "Butter"],
    ["Cheese", "Bread"],
    ["Milk", "Bread", "Butter"],
    ["Milk", "Cheese"],
    ["Bread", "Butter"],
    ["Milk", "Bread"]
]

# Customer 1 bought → Milk, Bread, Butter
# Customer 2 bought → Cheese, Bread

# Step 2: Convert transactions into one-hot encoded format
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# Converts data into binary format

# Step 3: Apply Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# Apriori works only on True / False (1 / 0) data.

# Step 4: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Apriori does:Finds frequent product combinations
# min_support=0.3
# Means:
# Itemset must appear in at least 30% of transactions
# Example:
# {Milk, Bread} appears in 3 out of 6 transactions
# Support = 0.5 → kept

print("Frequent Itemsets:\n")
print(frequent_itemsets)

print("\nAssociation Rules:\n")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# Support	How often items appear together
# Confidence	How reliable the rule is
# Lift	Strength of the relationship (>1 is good)