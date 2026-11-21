# Credit Card Fraud Detection

**Mir Fuad — TD Bank, Associate Banker II**

## Question

How do we flag fraudulent card transactions early enough to stop losses, without freezing so many legitimate cards that customers call the branch and lose trust? A missed fraud costs money; a false alarm costs a good customer their card, branch time, and trust. Accuracy hides that trade-off — predicting "never fraud" scores 99.83% and catches 0 frauds.

## Data — no Kaggle, no synthetic data

Zenodo record 21511985, v7: "A Deployment-Derived Online Banking Fraud Detection Inference-Log Dataset from a Live Cloud-Based Deep Learning System", Jan 2026, Open access, CC BY 4.0.
https://doi.org/10.5281/zenodo.20030064

56,962 transactions, 98 frauds (0.172%), Jan 1–31, 2026. Features used: `amount`, `time_value`, `v1`–`v28` only. Excluded as leakage (the old system's own outputs): `fraud_probability`, `risk_level`, `recommendation`, `confidence`, plus identifiers. No bank/customer data from work was used.

## Method

Stratified 80/20 train/test split (random_state=42; test = 11,393 rows, 20 frauds). Class weights on training data only — no resampling, so no synthetic rows. Models: Logistic Regression, Random Forest, XGBoost, LightGBM, plus two baselines (predict-never-fraud; flag everything ≥ $5,000). Judged on precision, recall, F1, and PR-AUC for the fraud class, not accuracy.

## Results (held-out test set)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Never fraud | 0.00 | 0.00 | 0.00 | 0.00 |
| Amount ≥ $5,000 rule | 0.43 | 0.65 | 0.52 | 0.28 |
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 |
| XGBoost | 1.00 | 1.00 | 1.00 | 1.00 |
| LightGBM | 1.00 | 1.00 | 1.00 | 1.00 |

**Read the perfect scores as a warning, not a win.** v12, v14, and v17 each have single-feature AUC ≈ 1.000, amount alone 0.996, and the best model's lowest fraud score (0.9999997) sits far above its highest legitimate score (0.0045). This demo dataset is near-perfectly separable — even without v12/v14/v17, Logistic Regression still scores 1.000. The realistic comparator is the amount rule: 13 of 20 test frauds caught, 17 legitimate cards frozen (76 frauds caught per 100 legitimate cards frozen). Thresholds were tuned on training out-of-fold scores; both reported operating points (0.10 high-recall, 0.50 balanced) give the same perfect confusion matrix on this 20-fraud test set, so the threshold choice cannot be learned here.

## Limitations

Proof-of-concept public demonstration, not a licensed bank. Only BLOCK transactions were manually reviewed (verification bias). 98 frauds, one month, one system. V1–V28 are anonymised. Do not deploy, and do not present these scores as production performance.

## Next step with real bank data

Re-run the same pipeline on a harder, larger labelled dataset (e.g. IEEE-CIS, 590,540 transactions, ~3.5% fraud); add card history, device, merchant, and velocity features; tune the threshold on expected cost (missed-fraud amount vs false-alarm cost), and validate over time (train January, test February).

## Reproduce

`Credit_Card_Fraud_Detection.ipynb` (executed, 15 cells) runs end-to-end from `data/fraud_tests_export.csv`. Environment in `.venv/` (pandas, scikit-learn, XGBoost, LightGBM). `model.py` is the standalone script; full metrics in `results/metrics.json`; EDA in `eda/EDA_REPORT.md`.
