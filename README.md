# 💳 Wallet Credit Scorer – Zeru Finance

This project builds a **wallet-level credit scoring system** using DeFi transaction data from the Aave V2 protocol. Using Python and machine learning, we calculate behavior-based credit scores that can help assess a user's likelihood of default, even in decentralized finance ecosystems.

---

## 📂 Folder Structure

wallet-credit-scorer/
├── data/
│   └── user-wallet-transactions.json                                  # Raw DeFi transaction data (input)

├── outputs/

│   ├── credit_scores.csv                                              # Scores for all wallets

│   ├── top_wallets.csv                                                # Top 10 highest-scoring wallets

│   └── score_distribution.png/score_distribution2.png                 # Histogram of credit score distribution

├── wallet_credit_scorer.py                                            # Main credit score generator

├── analysis.py                                                        # Visualization and EDA

├── requirements.txt                                                   # Dependencies

└── README.md                                                          # This file


---

## 🧠 Problem Statement

> **Goal:** Predict and rank wallet addresses by their creditworthiness using behavioral indicators from DeFi transactions.  
> **Dataset:** On-chain JSON logs of lending, borrowing, collateral deposits, and repayments.

---

## 📈 Features Used for Scoring

The following features were engineered for scoring each wallet:

- `avg_borrow_amount`: Average amount borrowed  
- `total_transactions`: Total number of transactions  
- `repayment_ratio`: Ratio of repaid amount to borrowed amount  
- `collateral_deposit_ratio`: Share of total deposits marked as collateral  
- `risk_adjusted_activity`: Composite score of borrowing and repaying behavior  
- `borrow_frequency`: Normalized count of borrow events  
- `borrow_to_collateral_ratio`: Volume-based ratio to detect over-leveraging

---

## 🧮 Scoring Formula (Simplified)

A weighted combination of normalized features:
Credit Score = f(repayment_ratio, collateral_ratio, borrow_frequency, ...)

The exact weights are designed to reward consistent repayments, responsible collateral usage, and sustainable borrowing.

---

## ▶️ How to Run the Project

### 1. Clone the repository:

```bash
git clone https://github.com/orryverse79-hub/wallet-credit-scorer.git
cd wallet-credit-scorer
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the credit scoring script:

```bash
python wallet_credit_scorer.py
```

### 4. Optional: Visualize the score distribution:

```bash
python analysis.py
```



📊 Sample Visual Output

🔍 Evaluation (Step 7)
The scoring system was evaluated based on:

Logical consistency of score drivers (repayment behavior, collateral usage)

Distribution of scores across wallets

Top/bottom wallet profile analysis

Fairness Considerations:

No bias from wallet ID or user metadata

Transparent formula with interpretable metrics

Future potential for XAI (explainable AI) integration





🧑‍💻 Author
Aditya Vishwakarma
Built as part of the Zeru Finance Credit Scoring Challenge
July 2025




---



