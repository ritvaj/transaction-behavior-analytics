import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# LOAD DATA
df = pd.read_csv("Transaction_FE_plots.csv")

# Ensure numeric types
df['fraud_signal_score'] = pd.to_numeric(df['fraud_signal_score'], errors='coerce').fillna(0)
df['isFraud'] = pd.to_numeric(df['isFraud'], errors='coerce').fillna(0).astype(int)

print(df.head(10))

print(df['fraud_signal_score'].min())

print(df['fraud_signal_score'].max())

print(df['fraud_signal_score'].describe())

print(f"Loaded {len(df)} rows. Baseline fraud rate: {df['isFraud'].mean():.4%}\n")

# Only using direct thresholds now
THRESHOLDS = [3, 4, 5, 6]

# Evaluate a single threshold
def evaluate_threshold_np(df, threshold):
    y_true = df['isFraud'].values
    y_pred = (df['fraud_signal_score'] >= threshold).astype(int).values

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    baseline  = y_true.mean()
    lift      = (recall / baseline) if baseline > 0 else np.nan

    flagged = int(y_pred.sum())

    return {
        "threshold": float(threshold),
        "flagged_count": flagged,
        "fraud_in_flagged": TP,
        "fraud_rate_in_flagged": TP / flagged if flagged > 0 else 0.0,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "baseline_fraud_rate": baseline,
        "lift": lift
    }


# Results print
def print_summary(result):
    thr = result["threshold"]
    print(f"\nThreshold = {thr}")
    print(f"  Flagged: {result['flagged_count']}")
    print(f"  Fraud in flagged: {result['fraud_in_flagged']} "
          f"(rate {result['fraud_rate_in_flagged']:.3%})")
    print(f"  TP={result['TP']} FP={result['FP']} FN={result['FN']} TN={result['TN']}")
    print(f"  Precision={result['precision']:.3%} | Recall={result['recall']:.3%} | FPR={result['fpr']:.3%}")
    print(f"  Baseline fraud={result['baseline_fraud_rate']:.4%} | Lift={result['lift']:.2f}x")


# RUN ALL THRESHOLDS
results = []

print("=== Fixed thresholds ===")
for t in THRESHOLDS:
    res = evaluate_threshold_np(df, t)
    results.append(res)
    print_summary(res)


# SAVE SUMMARY CSV
res_df = pd.DataFrame(results)
res_df.to_csv("ab_test_output/ab_test_results.csv", index=False)
print("\nSaved: ab_test_results.csv")


# QUICK TABLE with all thresholds
print("\n=== Results Table ===")
cols = ["threshold","flagged_count","fraud_in_flagged","precision","recall","fpr","lift"]
print(res_df[cols].to_string(index=False))



# Evaluation Plot: Precision Recall Curve

y_true = df["isFraud"].values
scores = df["fraud_signal_score"].values


# PRECISION–RECALL CURVE
thresholds = np.sort(df["fraud_signal_score"].unique())

precisions = []
recalls = []

for t in thresholds:
    y_pred = (scores >= t).astype(int)

    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)

plt.figure(figsize=(7,5))
plt.plot(recalls, precisions, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Fraud Signal Score)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/precision_recall_curve.png", dpi=300)
plt.savefig("plots/precision_recall_curve.svg", dpi=300)
plt.close()


print("Saved: precision_recall_curve.png / .svg")


# THRESHOLD METRICS TABLE
THRESHOLDS = [3, 4, 5, 6]
rows = []

for t in THRESHOLDS:
    y_pred = (scores >= t).astype(int)

    TP = int(((y_true==1) & (y_pred==1)).sum())
    FP = int(((y_true==0) & (y_pred==1)).sum())
    FN = int(((y_true==1) & (y_pred==0)).sum())
    TN = int(((y_true==0) & (y_pred==0)).sum())

    precision = TP/(TP+FP) if (TP+FP)>0 else 0
    recall = TP/(TP+FN) if (TP+FN)>0 else 0

    rows.append({
        "threshold": t,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision,
        "recall": recall,
        "flagged": int(y_pred.sum())
    })

tp_fp_table = pd.DataFrame(rows)
tp_fp_table.to_csv("threshold_metrics.csv", index=False)

print("Saved: threshold_metrics.csv")


# TOP 10 HIGH-RISK ACCOUNTS
top10 = (
    df.groupby("nameDest")
      .agg(
          max_score=("fraud_signal_score", "max"),
          avg_score=("fraud_signal_score", "mean"),
          total_score=('fraud_signal_score', 'sum'),
          tx_count=("fraud_signal_score", "count"),
          fraud_tx_count=("isFraud", "sum")
      )
      .sort_values(["max_score", "total_score"], ascending=False)
      .head(10)
      .reset_index()
)

print("\nTop 10 High-Risk Accounts:")
print(top10.to_string(index=False))


# SAVE CSV
top10.to_csv("ab_test_output/top10_high_risk_accounts.csv", index=False)
print("\nSaved: top10_high_risk_accounts.csv")
print("\nEvaluation complete.")
