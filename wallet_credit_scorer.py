import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
with open("C:/Users/Mr. Aditya/Downloads/user-wallet-transactions.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Step 1: Flatten nested fields
df['_id'] = df['_id'].apply(lambda x: x['$oid'])
df['createdAt'] = pd.to_datetime(df['createdAt'].apply(lambda x: x['$date']))
df['updatedAt'] = pd.to_datetime(df['updatedAt'].apply(lambda x: x['$date']))

# Step 2: Preview structure of the dataset
print("Available columns:", df.columns.tolist())
print("\nSample of 'actionData':")
print(df['actionData'].head(3))


# Step 3: Extract 'wallet', 'action', and 'amount' from appropriate fields

# Wallet is stored in 'userWallet'
df['wallet'] = df['userWallet']

# Extract 'action' and 'amount' from the 'actionData' field (which is a dict in string form)
df['action'] = df['actionData'].apply(lambda x: x.get('type') if isinstance(x, dict) else None)
df['amount'] = df['actionData'].apply(lambda x: float(x.get('amount', 0)) / 1e18 if isinstance(x, dict) else 0)

# Drop rows missing critical values
df = df.dropna(subset=['wallet', 'action', 'amount'])



# Step 4: Feature engineering per wallet
grouped = df.groupby('wallet')

features = pd.DataFrame()
features['tx_count'] = grouped.size()
features['deposit_count'] = grouped.apply(lambda x: (x['action'] == 'deposit').sum())
features['borrow_count'] = grouped.apply(lambda x: (x['action'] == 'borrow').sum())
features['repay_count'] = grouped.apply(lambda x: (x['action'] == 'repay').sum())
features['liquidation_count'] = grouped.apply(lambda x: (x['action'] == 'liquidationcall').sum())
features['total_amount'] = grouped['amount'].sum()
features['repay_to_borrow_ratio'] = features['repay_count'] / (features['borrow_count'] + 1)

# Step 5: Normalize features and compute score
scaler = MinMaxScaler()
normalized = scaler.fit_transform(features.fillna(0))
weights = [0.1, 0.2, 0.2, 0.3, -0.2, 0.1, 0.3]  # Example weights

import numpy as np
raw_score = normalized.dot(np.array(weights))
credit_score = MinMaxScaler((0, 1000)).fit_transform(raw_score.reshape(-1, 1)).flatten()
features['credit_score'] = credit_score

# Step 6: Output
features.reset_index(inplace=True)
features[['wallet', 'credit_score']].to_csv("credit_scores.csv", index=False)
print("✅ Credit scores saved to 'credit_scores.csv'")

# Task 5: Evaluate Model Performance

# 1. Summary statistics
print("\n📊 Credit Score Summary Statistics:")
print(features['credit_score'].describe())

# 2. Histogram of credit score distribution
import matplotlib.pyplot as plt

# Assuming 'features' is your DataFrame with a column 'credit_score'
plt.figure(figsize=(10, 6))
plt.hist(features['credit_score'], bins=50, edgecolor='black', color='skyblue')
plt.xlim(0, 200)  # Adjust this based on actual score range
plt.title('Improved Distribution of Credit Scores')
plt.xlabel('Credit Score')
plt.ylabel('Number of Wallets')
plt.grid(True)
plt.savefig('improved_score_distribution.png')
plt.show()


# 3. Top and bottom wallets
top_wallets = features.sort_values(by='credit_score', ascending=False).head(10)
bottom_wallets = features.sort_values(by='credit_score', ascending=True).head(10)

top_wallets.to_csv("top_10_wallets.csv", index=False)
bottom_wallets.to_csv("bottom_10_wallets.csv", index=False)

print("🏆 Top 10 wallets saved to 'top_10_wallets.csv'")
print("⚠️ Bottom 10 wallets saved to 'bottom_10_wallets.csv'")

