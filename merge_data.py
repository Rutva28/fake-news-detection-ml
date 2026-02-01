import pandas as pd

# Load datasets
true_df = pd.read_csv("true.csv")
fake_df = pd.read_csv("fake.csv")

# Add labels
true_df["label"] = 0
fake_df["label"] = 1

# Combine datasets
data = pd.concat([true_df, fake_df])

# Save combined dataset
data.to_csv("data.csv", index=False)

print("data.csv created successfully!")
