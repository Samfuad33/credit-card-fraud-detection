"""Credit Card Fraud Detection — modelling (Zenodo only, no Kaggle, no synthetic).
Features ONLY: amount, time_value, v1-v28.
Leakage columns EXCLUDED: fraud_probability, risk_level, recommendation,
confidence, response_time_ms, transaction_id, timestamp, ip_address.
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = "/home/hatch/workspace/projects/credit-card-fraud-detection"
OUT = f"{BASE}/results"
os.makedirs(f"{OUT}/charts", exist_ok=True)

FEATURES = ["amount", "time_value"] + [f"v{i}" for i in range(1, 29)]
LEAKAGE = ["fraud_probability", "risk_level", "recommendation", "confidence",
           "response_time_ms", "transaction_id", "timestamp", "ip_address"]
assert not set(FEATURES) & set(LEAKAGE)

df = pd.read_csv(f"{BASE}/data/fraud_tests_export.csv")
assert len(df) == 56962 and int(df["is_fraud"].sum()) == 98, (len(df), df["is_fraud"].sum())
X, y = df[FEATURES], df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)
print(f"train {X_train.shape} fraud={int(y_train.sum())} rate={y_train.mean():.4%}")
print(f"test  {X_test.shape} fraud={int(y_test.sum())} rate={y_test.mean():.4%}")

def evaluate(name, y_true, y_score, threshold=0.5):
    pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    m = {
        "model": name, "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float((pred == y_true).mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "frauds_caught_per_100_legit_frozen": float(tp / fp * 100) if fp else (float("inf") if tp else 0.0),
    }
    return m

results, scores = {}, {}

# Baseline: predict never fraud
results["Never fraud (baseline)"] = evaluate("Never fraud (baseline)", y_test, np.zeros(len(y_test)))
scores["Never fraud (baseline)"] = np.zeros(len(y_test))

# Simple amount rule: fraud if amount >= 5000 (no ML, for comparison)
results["Amount >= $5,000 rule"] = evaluate("Amount >= $5,000 rule", y_test,
    (X_test["amount"] >= 5000).astype(float))
scores["Amount >= $5,000 rule"] = (X_test["amount"] >= 5000).astype(float).values

# Logistic regression: log-amount + scaled features
Xtr_lr = X_train.copy(); Xte_lr = X_test.copy()
for d in (Xtr_lr, Xte_lr):
    d["amount"] = np.log1p(d["amount"])
    d["time_value"] = d["time_value"] - d["time_value"].min()
lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0))
lr.fit(Xtr_lr, y_train)
scores["Logistic Regression"] = lr.predict_proba(Xte_lr)[:, 1]
results["Logistic Regression"] = evaluate("Logistic Regression", y_test, scores["Logistic Regression"])

# Random Forest
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                            n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
scores["Random Forest"] = rf.predict_proba(X_test)[:, 1]
results["Random Forest"] = evaluate("Random Forest", y_test, scores["Random Forest"])

# XGBoost
spw = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                    eval_metric="logloss", n_jobs=-1, random_state=42)
xgb.fit(X_train, y_train)
scores["XGBoost"] = xgb.predict_proba(X_test)[:, 1]
results["XGBoost"] = evaluate("XGBoost", y_test, scores["XGBoost"])

# LightGBM
lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced",
                      verbose=-1, n_jobs=-1, random_state=42)
lgbm.fit(X_train, y_train)
scores["LightGBM"] = lgbm.predict_proba(X_test)[:, 1]
results["LightGBM"] = evaluate("LightGBM", y_test, scores["LightGBM"])

for name, m in results.items():
    print(f"{name:28s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
          f"PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f} "
          f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")

# Best model by PR-AUC (excluding rule/baseline)
ml = {k: v for k, v in results.items() if k not in ("Never fraud (baseline)", "Amount >= $5,000 rule")}
best_name = max(ml, key=lambda k: ml[k]["pr_auc"])
best_score = scores[best_name]
print("Best model by PR-AUC:", best_name, ml[best_name]["pr_auc"])

# Threshold tuning for best model — tuned on TRAIN out-of-fold scores, NOT on test.
# With perfect test separation, tuning on test gives degenerate thresholds (0 / 1).
from sklearn.model_selection import StratifiedKFold
best_cls = {"Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)),
            "Random Forest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
            "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, random_state=42),
            "LightGBM": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced", verbose=-1, n_jobs=-1, random_state=42)}[best_name]
oof = np.zeros(len(y_train))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, va_idx in skf.split(X_train, y_train):
    m = best_cls()
    if best_name == "Logistic Regression":
        Xtr, Xva = X_train.iloc[tr_idx].copy(), X_train.iloc[va_idx].copy()
        for d in (Xtr, Xva):
            d["amount"] = np.log1p(d["amount"]); d["time_value"] = d["time_value"] - d["time_value"].min()
        m.fit(Xtr, y_train.iloc[tr_idx]); oof[va_idx] = m.predict_proba(Xva)[:, 1]
    else:
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx]); oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)
f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
# thresholds array aligns with prec/rec[:-1]; use only finite, in (0,1) thresholds
valid = [(float(t), i) for i, t in enumerate(thr) if 0 < float(t) < 1]
best_f1_idx = max(valid, key=lambda ti: f1s[ti[1]])[1]
# Balanced: max F1 on OOF, ties broken by closeness to 0.5 (many thresholds tie at F1=1.0 here)
max_f1 = float(np.max(f1s[:-1]))
f1_ties = [i for i in range(len(thr)) if abs(float(f1s[i]) - max_f1) < 1e-9 and 0 < float(thr[i]) < 1]
balanced_thr = float(thr[min(f1_ties, key=lambda i: abs(float(thr[i]) - 0.5))]) if f1_ties else float(thr[best_f1_idx])
# High-recall: HIGHEST threshold that still keeps OOF recall >= 0.95
# (the lowest such threshold is degenerate when scores are 0/1-separated).
hi_candidates = [i for i, r in enumerate(rec[:-1]) if r >= 0.95 and 0 < float(thr[i]) < 1]
high_thr = float(thr[max(hi_candidates, key=lambda i: float(thr[i]))]) if hi_candidates else balanced_thr
print(f"OOF tuning ({best_name}): balanced thr={balanced_thr:.4f}, high-recall thr={high_thr:.4f}")
# For charting, use OOF precision/recall vs threshold
best_score = scores[best_name]
_gap = {"min_fraud_score_test": float(best_score[y_test.values == 1].min()),
        "max_legit_score_test": float(best_score[y_test.values == 0].max())}
print("Test score gap (LightGBM):", _gap)
# Operating points for reporting: the OOF-tuned thresholds collapse to ~1.0 because
# scores are 0/1-separated, so they are not meaningful choices. Instead use the gap:
# any threshold in (0.005, 0.999) is perfect on test. Choose 0.10 as high-recall
# (well below the gap — catches fraud even if scores shift down) and 0.50 as balanced
# (standard midpoint). Both are reported with their identical test confusion matrices,
# and the note in metrics.json states they cannot be distinguished on this test set.
high_thr, balanced_thr = 0.10, 0.50
print(f"Reporting operating points: high-recall thr={high_thr}, balanced thr={balanced_thr}")

operating = {}
for label, t in [("high_recall", high_thr), ("balanced_max_f1", balanced_thr)]:
    m = evaluate(f"{best_name} @ {label}", y_test, best_score, threshold=t)
    operating[label] = m
    print(f"{label}: thr={t:.4f} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
          f"TP={m['tp']} FP={m['fp']} FN={m['fn']} | {m['frauds_caught_per_100_legit_frozen']:.1f} frauds caught per 100 legit frozen")

# Ablation: same models WITHOUT v12, v14, v17 (the three near-perfectly separating
# features, each single-feature AUC ~1.000). This shows what performance looks like
# without the demo-dataset artefact, and is the honest comparison.
ABLATE = ["v12", "v14", "v17"]
FEATS_ABL = [f for f in FEATURES if f not in ABLATE]
ablation = {}
Xtr_a, Xte_a = X_train[FEATS_ABL], X_test[FEATS_ABL]
# LR ablation (log amount)
Xtr_l, Xte_l = Xtr_a.copy(), Xte_a.copy()
for d in (Xtr_l, Xte_l):
    d["amount"] = np.log1p(d["amount"]); d["time_value"] = d["time_value"] - d["time_value"].min()
lr_a = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0))
lr_a.fit(Xtr_l, y_train)
ablation["Logistic Regression (no v12/v14/v17)"] = evaluate("LR ablation", y_test, lr_a.predict_proba(Xte_l)[:, 1])
rf_a = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=42)
rf_a.fit(Xtr_a, y_train)
ablation["Random Forest (no v12/v14/v17)"] = evaluate("RF ablation", y_test, rf_a.predict_proba(Xte_a)[:, 1])
xgb_a = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, random_state=42)
xgb_a.fit(Xtr_a, y_train)
ablation["XGBoost (no v12/v14/v17)"] = evaluate("XGB ablation", y_test, xgb_a.predict_proba(Xte_a)[:, 1])
lgbm_a = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced", verbose=-1, n_jobs=-1, random_state=42)
lgbm_a.fit(Xtr_a, y_train)
ablation["LightGBM (no v12/v14/v17)"] = evaluate("LGBM ablation", y_test, lgbm_a.predict_proba(Xte_a)[:, 1])
for name, m in ablation.items():
    print(f"ABLATION {name:42s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.3f} TP={m['tp']} FP={m['fp']} FN={m['fn']}")

# Single-feature AUCs
single_auc = {}
for c in FEATURES:
    a = roc_auc_score(y, df[c])
    single_auc[c] = float(max(a, 1 - a))

# Save metrics
out = {
    "dataset": "Zenodo 21511985 v7, https://doi.org/10.5281/zenodo.20030064, Jan 2026, 56962 rows, 98 frauds (0.172%)",
    "features": FEATURES,
    "leakage_excluded": LEAKAGE,
    "split": {"test_size": 0.20, "stratify": True, "random_state": 42,
              "train_rows": int(len(X_train)), "train_fraud": int(y_train.sum()),
              "test_rows": int(len(X_test)), "test_fraud": int(y_test.sum())},
    "test_metrics_at_0.5": results,
    "ablation_no_v12_v14_v17_test_metrics_at_0.5": ablation,
    "single_feature_auc_max_direction": single_auc,
    "best_model": best_name,
    "test_score_gap_best_model": _gap,
    "operating_points_note": "On this 20-fraud test set both operating points give the same confusion matrix (perfect separation). The threshold choice cannot be learned from this test set; a bank would need a harder, larger dataset to choose between them.",
    "operating_points": operating,
    "warning": "All four ML models score a perfect 1.000 precision/recall/F1/PR-AUC on the 20-fraud test set. This is NOT production performance: v12, v14 and v17 each have single-feature AUC ~1.000 in this demo dataset, so fraud is near-perfectly separable. The ablation without those three features is the honest comparison. Test set has only 20 frauds; each fraud is 5pp of recall. Do not present perfect scores as production-ready.",
    "perfect_score_explanation": "v12/v14/v17 fraud means are 5-8 standard deviations from legitimate means; a single threshold on any of them separates almost all fraud. Combined with amount (AUC 0.996), the test set is trivially separable.",
}
json.dump(out, open(f"{OUT}/metrics.json", "w"), indent=2)
pd.DataFrame(results).T.to_csv(f"{OUT}/evaluation_table.csv")

# --- Charts ---
plt.rcParams.update({"font.size": 10})
# PR curves
fig, ax = plt.subplots(figsize=(8, 5))
for name, s in scores.items():
    if name.startswith("Never"):
        continue
    p, r, _ = precision_recall_curve(y_test, s)
    ap = average_precision_score(y_test, s)
    ax.plot(r, p, label=f"{name} (AP={ap:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall curves — held-out test set (n={len(y_test)}, {int(y_test.sum())} frauds)")
ax.legend(fontsize=8); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/charts/pr_curves.png", dpi=200); plt.close()

# Model comparison: PR-AUC and F1
names = list(results.keys())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].barh(names, [results[n]["pr_auc"] for n in names], color="#1f4e79")
axes[0].set_title("PR-AUC (test set)"); axes[0].set_xlim(0, 1.05)
axes[1].barh(names, [results[n]["f1"] for n in names], color="#548235")
axes[1].set_title("F1 at threshold 0.5 (test set)"); axes[1].set_xlim(0, 1.05)
for ax in axes:
    for i, n in enumerate(names):
        v = results[n]["pr_auc"] if ax is axes[0] else results[n]["f1"]
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/charts/model_comparison.png", dpi=200); plt.close()

# Precision-recall vs threshold (best model)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(thr, prec[:-1], label="Precision", color="#1f4e79")
ax.plot(thr, rec[:-1], label="Recall", color="#c00000")
ax.plot(thr, f1s[:-1], label="F1", color="#548235", linestyle="--")
ax.axvline(high_thr, color="#c00000", linestyle=":", label=f"High-recall thr={high_thr:.3f}")
ax.axvline(balanced_thr, color="#548235", linestyle=":", label=f"Balanced thr={balanced_thr:.3f}")
ax.set_xlabel("Threshold"); ax.set_ylabel("Score"); ax.set_xlim(0, 1)
ax.set_title(f"Precision / Recall / F1 vs threshold — {best_name}")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT}/charts/threshold_tuning.png", dpi=200); plt.close()

# Confusion matrices at operating points (as annotated tables)
for label, m in operating.items():
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted legit", "Predicted fraud"])
    ax.set_yticklabels(["Actual legit", "Actual fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
    ax.set_title(f"{best_name} — {label}\n(thr={m['threshold']:.3f}, P={m['precision']:.2f}, R={m['recall']:.2f})")
    plt.tight_layout(); plt.savefig(f"{OUT}/charts/confusion_{label}.png", dpi=200); plt.close()

print("saved metrics.json, evaluation_table.csv, charts")
