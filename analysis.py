import pandas as pd
import matplotlib.pyplot as plt

# Load the credit scores from CSV
features = pd.read_csv("credit_scores.csv")

# Plot the distribution of credit scores
plt.hist(features['credit_score'], bins=10, edgecolor='black')
plt.title('Wallet Credit Score Distribution')
plt.xlabel('Credit Score')
plt.ylabel('Number of Wallets')

# Save the plot
plt.savefig("score_distribution.png")
print("✅ Credit score distribution saved as 'score_distribution.png'")
