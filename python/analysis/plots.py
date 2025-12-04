import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Transaction_FE_final.csv")
print(df.shape)
print(df.head(5))
print(df['type'].value_counts())

# Plot I
# Create log amount (already created a column)
# df["log_amount"] = np.log1p(df["amount"])

plt.figure(figsize=(10,5))
sns.histplot(df["log_amount"], bins=100, kde=True)
plt.title("Transaction Amount Distribution (Log Scale)")
plt.xlabel("log(amount + 1)")
plt.ylabel("Frequency")


median_val = df['log_amount'].median()
p90 = df['log_amount'].quantile(0.90)
p99 = df['log_amount'].quantile(0.99)

plt.axvline(median_val, color = 'red', linestyle = '--', linewidth = 2, label = 'Median')
plt.axvline(p90, color = 'orange', linestyle = '--', linewidth = 2, label = '90th %ile')
plt.axvline(p99, color = 'purple', linestyle = '--', linewidth = 2, label = '99th %ile')

plt.legend()

plt.tight_layout()
plt.savefig("plots/01_log_amount_distribution.png", dpi=300)
plt.savefig("plots/01_log_amount_distribution.svg", bbox_inches='tight')
plt.show()


# Plot II
# Map labels for x-axis readability

df["fraud_label"] = df["isFraud"].map({0: "Non-fraud", 1: "Fraud"})

plt.figure(figsize=(10,6))

# violin for distribution shape + box for medians/IQR
sns.violinplot(x="fraud_label", y="log_amount", data=df, inner=None, linewidth=0.8)
sns.boxplot(x="fraud_label", y="log_amount", data=df,
            width=0.18, showcaps=True, boxprops={'zorder':2}, showfliers=False)

plt.title("Fraud vs Non-Fraud — Transaction Amount (log scale)")
plt.ylabel("log(amount + 1)")
plt.xlabel("")
plt.tight_layout()


plt.savefig("plots/02_fraud_vs_nonfraud_amount.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/02_fraud_vs_nonfraud_amount.svg", bbox_inches="tight")
plt.show()


# Plot III
# Compute mismatch rate per type

type_mismatch = (df.groupby("type")["orig_delta_mismatch_dir"].mean().sort_values(ascending=False)) 

plt.figure(figsize=(10,5))
sns.barplot(
    x=type_mismatch.index,
    y=type_mismatch.values,
    palette="Reds_r")

plt.title("Origin Ledger Mismatch Rate by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Mismatch Rate")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("plots/03_origin_mismatch_by_type.png", dpi=300)
plt.savefig("plots/03_origin_mismatch_by_type.svg", bbox_inches='tight')

plt.show()


# Plot IV
# Grouping by dest_tx_count_step (destination activity level)

plot_df = (
    df.groupby('dest_tx_count_step')
      .agg(
          dest_mismatch_rate=('dest_mismatch', 'mean'),
          count=('dest_mismatch', 'size')   # <-- add count for annotation
      )
      .reset_index())

# Sort by step number (cleaner than sorting by rate)
plot_df = plot_df.sort_values('dest_tx_count_step')

plt.figure(figsize=(8,6))
ax = sns.barplot(
    data=plot_df,
    x='dest_tx_count_step',
    y='dest_mismatch_rate',
    palette='Blues')


# ADD ANNOTATIONS (n = count)
for i, row in plot_df.iterrows():
    ax.text(
        i,                                       # x-position (index of bar)
        row['dest_mismatch_rate'] + 0.02,        # y-position above the bar
        f"n={row['count']}",                     # annotation text
        ha='center', va='bottom', fontsize=10)    # formatting
    

plt.title("Mismatch Rate vs Destination Activity Level")
plt.xlabel("Destination Transaction Count Step")
plt.ylabel("Destination Mismatch Rate")
plt.ylim(0, plot_df['dest_mismatch_rate'].max() + 0.1)  # prevent text cutoff
plt.tight_layout()

# SAVE
plt.savefig("plots/04_Mismatch Rate vs Destination Activity Level.png", dpi=300)
plt.savefig("plots/04_Mismatch Rate vs Destination Activity Level.svg", bbox_inches='tight')

plt.show()

# Plot V - Either vs Both Mismatch Bars

# Compute rates and counts (clear, explicit)
rates = {
    "Either": df["either_mismatch"].mean(),
    "Both": df["both_mismatch"].mean(),
    "Origin only": ((df["origin_mismatch"] == 1) & (df["dest_mismatch"] == 0)).mean(),
    "Dest only": ((df["dest_mismatch"] == 1) & (df["origin_mismatch"] == 0)).mean()
}

counts = {
    "Either": int(df["either_mismatch"].sum()),
    "Both": int(df["both_mismatch"].sum()),
    "Origin only": int(((df["origin_mismatch"] == 1) & (df["dest_mismatch"] == 0)).sum()),
    "Dest only": int(((df["dest_mismatch"] == 1) & (df["origin_mismatch"] == 0)).sum())
}

plot_df = (
    pd.DataFrame({
        "label": list(rates.keys()),
        "rate": list(rates.values()),
        "count": [counts[k] for k in rates.keys()]
    })
    .sort_values("rate", ascending=True)   # for horizontal bars, smallest->bottom, largest->top
    .reset_index(drop=True)
)

print(plot_df.head())

# Plot
plt.figure(figsize=(8,5))
ax = sns.barplot(data=plot_df, x="rate", y="label", palette="Greens", edgecolor="k")

# Annotate percent and n (clean)
for i, row in plot_df.iterrows():
    ax.text(row["rate"] + 0.01, i, f"{row['rate']:.1%}", va="center", fontsize=10)      # percent to right
    ax.text(0.01, i, f"n={row['count']:,}", va="center", fontsize=9, color="black")     # n at left inside bar

ax.set_xlabel("Proportion of Transactions")
ax.set_xlim(0, plot_df["rate"].max() + 0.15)
ax.set_title("Either vs Both vs Single-Side Mismatch")
ax.invert_yaxis()  # largest on top
plt.tight_layout()
plt.show()


# Plot VI - Mule Score Distribution
# Purpose: show where mule-like accounts concentrate and separation vs fraud label

# --- Print percentiles to help pick thresholds
percentiles = [0.50, 0.75, 0.90, 0.95, 0.99]
pvals = df['mule_score_w'].quantile(percentiles)
print("Mule score percentiles:")
for pct, val in pvals.items():
    print(f"  {int(pct*100)}th percentile: {val:.3f}")

# --- KDE plot
plt.figure(figsize=(8,5))
ax = df.loc[df['isFraud']==0, 'mule_score_w'].plot.kde(label='Non-Fraud Density', linewidth=2)
df.loc[df['isFraud']==1, 'mule_score_w'].plot.kde(ax=ax, label='Fraud Density', linewidth=2)
plt.xlim(0, df['mule_score_w'].max() + 1)
ax.set_title("Mule Score — Fraud vs Non-Fraud")
ax.set_xlabel("Mule Score")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()

plt.savefig("plots/06_Mule Score — Fraud vs Non-Fraud.png", dpi=300, bbox_inches='tight')
plt.savefig("plots/06_Mule Score — Fraud vs Non-Fraud.svg", bbox_inches='tight')
plt.show()


# --- Quick high-risk tagging

# Choose threshold (95th percentile) for "high-risk sample
threshold = df['mule_score_w'].quantile(0.95)

high_risk_df = df.loc[df['mule_score_w'] >= threshold]

print(high_risk_df.head())

# counts
high_risk_count = len(high_risk_df)
fraud_in_high_risk = high_risk_df['isFraud'].sum()

print("Threshold:", threshold)
print("High-risk transactions:", high_risk_count)
print("Frauds within high-risk:", fraud_in_high_risk)



# Plot VII
# Fraud Signal Score — Additive, interpretable risk metric

# Feature: Lifetime destination activity.
# Mule accounts often appear only once so flagging first-time receivers.
df['dest_tx_count_lifetime'] = df.groupby('nameDest')['nameDest'].transform('count')
df['is_new_dest'] = (df['dest_tx_count_lifetime'] == 1).astype(int)

# Final HARD anomalies with weights
df['mule_score_w_high'] = 2.5 * (df['mule_score_w'] >= 2).astype(int)
df['both_mismatch'] = 2.5 * df['both_mismatch'].astype(int)
df['is_dest_velo'] = 2 * df['is_dest_high_velocity'].astype(int)
df['is_pass_through'] = 1 * df['is_pass_through'].astype(int)
df['is_many_senders'] = 1 * df['is_many_senders'].astype(int)
df['dest_burst'] = 1 * (df['dest_tx_count_step'] >= 3).astype(int)
df['is_new_dest'] = 1 * df['is_new_dest'].astype(int)

# Final Fraud Signal Score (sum of all weighted flags)
fraud_components = ['mule_score_w_high', 'both_mismatch', 
    'is_dest_velo', 'is_pass_through', 'is_many_senders', 'dest_burst', 'is_new_dest']

df['fraud_signal_score'] = df[fraud_components].sum(axis=1)


plt.figure(figsize=(8,5))
sns.kdeplot(df[df['isFraud']==0]['fraud_signal_score'], label='Non-Fraud Density', linewidth=2, color="#1E95075F")
sns.kdeplot(df[df['isFraud']==1]['fraud_signal_score'], label='Fraud Density', linewidth=2, color='#d62728')
plt.xlim(0, df['fraud_signal_score'].max() + 1)
plt.title("Fraud Signal Score — Fraud vs Non-Fraud")
plt.xlabel("Fraud Signal Score")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("plots/07_FraudScore_fraud_vs_nonfraud.png", dpi=300)
plt.savefig("plots/07_FraudScore_fraud_vs_nonfraud.svg", bbox_inches='tight')
plt.show()

# Save
df.to_csv('Transaction_FE_plots.csv', index = False)