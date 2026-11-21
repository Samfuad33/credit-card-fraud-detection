# ID Without a Card: Catching Cardless Withdrawal Fraud at the Teller Line

**Mir Fuad — TD Bank, Associate Banker II (teller line), moving into data analytics**
**Notebook:** `TELLER_FRAUD_NOTEBOOK.ipynb` | **Plan:** `TELLER_FRAUD_CAPSTONE_PLAN.md`

## Question

How can a bank spot a fraudulent ID-based cash withdrawal made without a card, before the money leaves the branch, without blocking so many legitimate customers that the teller line stops moving?

## Why this question

Every day, customers withdraw cash with ID only, no card — mostly older customers who lost or forgot a card. One case, mid-November 2026: a man in his 20s, dressed as an older man, tried to withdraw money with a fake-looking ID. I stopped him. An office analyst sees a completed transaction — amount, account, time. They never see the person, the ID in hand, or how someone acts when questioned. That is the gap this project works in. No TD customer data was used; the case is anonymised, one case, not a rate.

## Data

Zenodo record 21511985, v7 (deployment-derived online banking fraud inference log, CC BY 4.0): **56,962 rows, 98 frauds (0.172%)**, January 2026. No Kaggle, no synthetic data. Usable features only: `amount`, `time_value`, `v1`–`v28`. Excluded as leakage: `fraud_probability`, `risk_level`, `recommendation`, `confidence` (the old system's own outputs), plus identifiers. Split: stratified 80/20, `random_state=42` — test set 11,393 rows, 20 frauds, 11,373 legitimate. EDA, re-used from the earlier project: median fraud $15,122 vs legitimate $33; 78 of 98 frauds are $5,000+ (35.8% fraud in that bin, 0% under $100).

## Method

No new ML model — the four existing models all score a perfect 1.000 precision/recall/PR-AUC on this 20-fraud test set, which is a dataset artefact (v12/v14/v17 each separate fraud almost perfectly), not production performance. The honest comparator is a simple amount rule, run at three thresholds on the same held-out test set, plus a teller verification checklist built from what I actually check at the counter.

## Results — amount rule, held-out test set

| Rule | Frauds caught | Frauds missed | Legit frozen | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Amount ≥ $1,000 | 19 / 20 | 1 | 310 | 0.058 | 0.95 | 0.109 |
| Amount ≥ $5,000 | 13 / 20 | 7 | 17 | 0.433 | 0.65 | 0.520 |
| Amount ≥ $10,000 | 11 / 20 | 9 | 5 | 0.688 | 0.55 | 0.611 |

$1,000 catches nearly all fraud but freezes 310 legitimate customers — the line stops moving. $10,000 freezes only 5 but lets 9 of 20 frauds walk out. $5,000 is the middle. Each test fraud is 5 points of recall, so the shape of the trade-off is the finding, not the exact F1. Chart: `results/charts/amount_rule_tradeoff.png`. Full table: `results/amount_rule_thresholds.csv`.

## The teller checklist

1. Face vs photo on the ID — catches impersonation. 2. ID details (feel, print, expiry, photo quality) — catches a fake ID. 3. Account knowledge questions — catches someone with an ID who does not know the account. 4. Amount vs the customer's usual pattern, as the teller knows it. 5. Behaviour when questioned. 6. Escalate to a supervisor before cash leaves. No catch rate is claimed for the checklist — there is no labelled data to measure one.

## Idea to test: a photo on the customer profile

When I pull up a customer's profile, there should be a photo of them on the screen. Right now the only face I can compare is the photo on the ID they hand me — and that ID can be fake. A profile photo, taken when the account was opened, would give me a second face to compare against the person standing in front of me. It would not stop someone who is disguised well, but it would catch most impersonation attempts, because a fake ID is easy and a matching face is not. This is a proposal to test, not a proven fix: it raises real privacy questions (whose photo, who can see it, how it is stored, whether customers can opt out), and it cannot be tested with this dataset, which has no photo field. It belongs in the project as the teller's idea, with the privacy trade-off stated openly.

## Recommendation

Two steps, for flagged withdrawals only: an amount flag from the system, then the checklist before cash leaves. Do not add friction to small, routine cardless withdrawals — under $100 there were 0 frauds in 41,528 transactions.

## Limitations

One case; one public demo dataset; 98 frauds, 20 in test; no ID, card-present, branch, or face/photo field in the data, so the checklist cannot be tested with numbers here; no TD data; perfect ML scores are an artefact. What would test it properly: labelled cardless-withdrawal data with ID and card-present fields, and a harder, larger dataset (IEEE-CIS via OpenML is still untested as a route).

## What next

Test on a harder, larger labelled dataset; add velocity and customer-history features a branch system would actually have; tune the threshold on cost (mean fraud here is $24,195) rather than F1; validate over time, not on a random split.
