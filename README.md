# Credit Card Default Prediction

## Overview
This project builds an end-to-end machine learning pipeline to predict credit card default risk using historical customer repayment and billing data. The objective is not leaderboard optimization, but to develop a **realistic, interpretable, and decision-oriented model**, aligned with how credit risk models are used in practice.

The project emphasizes:
- Structured exploratory data analysis (EDA)
- Behavior-driven feature engineering
- Comparison of multiple model families
- Transparent model selection
- Threshold tuning for business relevance

---

## Dataset
The dataset contains anonymized customer credit information, including:
- Demographics (age, sex, education, marriage)
- Credit limit
- Monthly repayment status (`PAY_x`)
- Monthly bill amounts (`BILL_AMT_x`)
- Monthly payment amounts (`PAY_AMT_x`)
- Binary default indicator (`DEFAULT`)

The target variable is **DEFAULT**, indicating whether a customer defaulted.

---

## Exploratory Data Analysis
EDA focused on understanding **repayment behavior over time**, rather than relying only on static attributes.

Key observations:
- Repayment status variables are the strongest predictors of default.
- Default rates vary across demographic groups but do not dominate predictive power.
- Billing and payment amounts are highly correlated with each other but weakly correlated with default in raw form.
- Class imbalance is present and explicitly handled during modeling.

EDA findings directly informed feature engineering decisions.

---

## Feature Engineering
Features were engineered to capture **behavioral patterns and trends**, including:
- Aggregated repayment behavior (number of late months, maximum delay)
- Weighted lateness scores emphasizing recent delinquency
- Payment-to-bill ratios and temporal changes
- Credit utilization indicators
- Consistency flags (usually late, usually minimum payment)
- Trend-based features capturing changes in bills and payments over time

All features were constructed using historical data only to avoid leakage.

---

## Modeling Approach
Three model families were evaluated:
1. Logistic Regression
2. Random Forest
3. XGBoost

Models were evaluated using stratified cross-validation with **ROC-AUC** and **F1 score**.

Although tree-based models achieved slightly higher performance, gains over Logistic Regression were marginal (~0.01–0.02 ROC-AUC). Given the importance of interpretability in credit risk settings, **Logistic Regression** was selected as the final model.

---

## Threshold Selection
The model outputs probabilities rather than direct decisions. To translate these probabilities into actionable classifications, multiple thresholds were evaluated.

- Thresholds tested: 0.3, 0.4, 0.5
- A threshold of **0.4** was selected as a balanced operating point

This threshold improves recall of defaulters while maintaining reasonable precision, reflecting real-world risk screening trade-offs.

---

## Final Performance (Test Set, Threshold = 0.4)
- **ROC-AUC:** ~0.77
- **Recall (Default):** ~67%
- **Precision (Default):** ~40%

These results are consistent with realistic credit risk models and demonstrate meaningful business value despite inherent uncertainty in default prediction.

---

## Key Takeaways
- Thoughtful feature engineering can extract strong signal without excessive model complexity.
- Increasing model complexity yields diminishing returns beyond a point.
- Interpretability and decision context are critical in regulated domains.
- Threshold tuning is essential when deploying probabilistic models.

---

## Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## Author
This project was developed as part of a personal machine learning portfolio, focusing on **practical modeling decisions and real-world applicability** rather than competition-style optimization.
