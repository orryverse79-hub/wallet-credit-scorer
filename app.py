import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# --- App Configuration ---
st.set_page_config(page_title="DeFi Credit Scorer", layout="wide")
st.title("🛡️ DeFi Wallet Credit Risk Scorer")
st.write("Upload your Aave/Compound transaction JSON logs to generate behavior-based credit scores (0-1000).")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload user-wallet-transactions.json", type="json")

if uploaded_file is not None:
    with st.spinner("Analyzing on-chain data and calculating scores..."):
        try:
            # 1. Load Data
            data = json.load(uploaded_file)
            df = pd.DataFrame(data)

            # 2. Flatten nested fields safely
            df['_id'] = df['_id'].apply(lambda x: x.get('$oid') if isinstance(x, dict) else x)
            
            # 3. Extract core fields
            df['wallet'] = df.get('userWallet', None)
            df['action'] = df['actionData'].apply(lambda x: x.get('type') if isinstance(x, dict) else None)
            df['amount'] = df['actionData'].apply(lambda x: float(x.get('amount', 0)) / 1e18 if isinstance(x, dict) else 0)
            
            # Drop missing values
            df = df.dropna(subset=['wallet', 'action', 'amount'])

            # 4. Feature Engineering (From your original script)
            grouped = df.groupby('wallet')
            features = pd.DataFrame()
            features['tx_count'] = grouped.size()
            features['deposit_count'] = grouped.apply(lambda x: (x['action'] == 'deposit').sum())
            features['borrow_count'] = grouped.apply(lambda x: (x['action'] == 'borrow').sum())
            features['repay_count'] = grouped.apply(lambda x: (x['action'] == 'repay').sum())
            features['liquidation_count'] = grouped.apply(lambda x: (x['action'] == 'liquidationcall').sum())
            features['total_amount'] = grouped['amount'].sum()
            features['repay_to_borrow_ratio'] = features['repay_count'] / (features['borrow_count'] + 1)

            # 5. Normalize features and compute score
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(features.fillna(0))
            weights = [0.1, 0.2, 0.2, 0.3, -0.2, 0.1, 0.3] 
            raw_score = normalized.dot(np.array(weights))
            
            # Scale to 0 - 1000
            credit_score = MinMaxScaler((0, 1000)).fit_transform(raw_score.reshape(-1, 1)).flatten()
            features['credit_score'] = np.round(credit_score, 2)
            features.reset_index(inplace=True)

            st.success("Analysis Complete!")

            # --- Dashboard UI ---
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🏆 Top 10 Safest Wallets")
                top_wallets = features.sort_values(by='credit_score', ascending=False).head(10)
                # Display as a clean, interactive dataframe
                st.dataframe(top_wallets[['wallet', 'credit_score', 'tx_count', 'repay_to_borrow_ratio']], use_container_width=True)

            with col2:
                st.subheader("📊 Score Distribution")
                # Using your matplotlib logic to render in the browser
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(features['credit_score'], bins=20, edgecolor='black', color='skyblue')
                ax.set_xlabel('Credit Score')
                ax.set_ylabel('Number of Wallets')
                ax.grid(axis='y', alpha=0.75)
                st.pyplot(fig)

            # Global Metrics
            st.divider()
            st.subheader("Global Dataset Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Wallets Analyzed", f"{len(features)}")
            m2.metric("Total Transactions", f"{features['tx_count'].sum()}")
            m3.metric("Average Credit Score", f"{int(features['credit_score'].mean())}")

        except Exception as e:
            st.error(f"Error processing data: {str(e)}. Please ensure the JSON file matches the expected schema.")
else:
    st.info("Awaiting file upload. Please upload your transaction JSON file to begin.")
