# Interview Cheat Sheet — Teller Fraud Project
Mir Fuad — for data analyst interviews. Read this before the interview, not during it.
The rule: explain WHAT each block does and WHY, in one sentence. Nobody expects you to recite syntax.

## SQL vs Python — the honest answer

**Lead with SQL. It is easier to explain, and it is not useless.**
- SQL is short, readable, and every line says what it does in plain words: SELECT, GROUP BY, COUNT.
- Python is better at cleaning data and running models, but its code is longer and harder to talk through line by line.
- Newer teams do want Python, so keep Python in the project — but in the interview, walk them through the SQL first, then say "the same logic in Python is in the notebook."
- You are not hiding Python. You are leading with the part you can explain cold.

## The 5 blocks they are most likely to ask about

### 1. The SQL bin query (lead with this one)
```sql
SELECT CASE WHEN amount < 100 THEN '<$100' ... END AS amt_bin,
       COUNT(*) AS n, SUM(is_fraud) AS fraud,
       CAST(SUM(is_fraud) AS FLOAT) / COUNT(*) AS fraud_rate
FROM fraud_tests_export
GROUP BY amt_bin ORDER BY MIN(amount);
```
Say: "I grouped every transaction into amount bins, counted how many were in each bin, counted how many were fraud, and divided to get the fraud rate per bin."
Why it matters: this is the evidence the whole amount rule rests on. Under $100: 0% fraud. $5,000+: 35.8% fraud.

### 2. The same thing in pandas
```python
df["amt_bin"] = pd.cut(df["amount"], bins=bins, labels=labels)
ab = df.groupby("amt_bin").agg(n=("is_fraud", "size"), fraud=("is_fraud", "sum"))
ab["rate"] = ab["fraud"] / ab["n"]
```
Say: "pd.cut puts each transaction into an amount bin, groupby counts them the same way the SQL does, and the rate is fraud divided by total. Same table, two languages."
If they ask why both: "SQL is what I would run on a bank database. Python is what I used here because the data came as a CSV."

### 3. Loading the data
```python
df = pd.read_csv(DATA)
```
Say: "I loaded the Zenodo CSV into a dataframe. 56,962 rows, 98 frauds, 0.17% fraud rate. No Kaggle, no synthetic data."

### 4. The train/test split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)
```
Say: "I held out 20% of the data to test on, and I stratified it, so the test set has the same fraud rate as the full data — 20 frauds in 11,393 rows. Random state 42 means anyone can rerun it and get the same split."
If they ask what stratify means: "It keeps the fraud share the same in train and test, so the test is not accidentally all fraud or no fraud."

### 5. The amount rule (the finding)
```python
pred = (df["amount"] >= threshold).astype(int)   # threshold = 1000, 5000, or 10000
```
Say: "I flagged every transaction over a dollar amount and counted how many frauds I caught versus how many real customers I froze."
Results to know by heart: $1,000 catches 19 of 20 frauds but freezes 310 real customers. $5,000 catches 13 of 20, freezes 17. $10,000 catches 11 of 20, freezes 5.

## If you freeze

Do not guess syntax. Say one of these, then go back to the story:
- "The exact syntax I would look up, but what it does is..."
- "That line groups the data and counts fraud per group — the number it produced is..."
- "I built it in the notebook and I can walk you through what it does, even if I do not remember the exact function name."

Interviewers are testing whether you understand what the code does, not whether you memorised it. The numbers are your anchor: if you can say what the code produced, you have answered the question.

## The 30-second story (open with this)

"I work the teller line. A man in his 20s dressed as an older man tried to withdraw cash with a fake ID, and I stopped him. I wanted to know if data could catch that before the money leaves. The public data says fraud lives in big transactions — median fraud $15,122 versus $33 for a normal transaction — but a simple amount rule is a trade-off: at $1,000 you catch 19 of 20 frauds and freeze 310 real customers. So my answer is not a lower threshold, it is a flag plus a teller checklist, only on the flagged ones."

## Things NOT to claim

- Do not present the models' perfect 1.000 scores as real. Say: "That is a dataset artefact — three features separate fraud almost perfectly in this demo data. The amount rule is the honest comparator."
- Do not claim the checklist has a catch rate. Say: "There is no labelled data to measure one."
- Do not claim any TD data was used. Say: "One anonymised case from my own shift, no customer data."
