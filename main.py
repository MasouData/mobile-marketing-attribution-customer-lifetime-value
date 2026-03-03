# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

events_df = pd.read_csv("/Volumes/workspace/peaks/project/app_events.csv")
spend_df = pd.read_csv("/Volumes/workspace/peaks/project/marketing_spend.csv")
users_df = pd.read_csv("/Volumes/workspace/peaks/project/user_profiles.csv")
transactions_df = pd.read_csv("/Volumes/workspace/peaks/project/user_transactions.csv")
skan_df = pd.read_csv("/Volumes/workspace/peaks/project/skan_attribution_ios.csv")

# COMMAND ----------

print(events_df.info())
print(spend_df.info())
print(users_df.info())
print(transactions_df.info())
print(skan_df.info())

# COMMAND ----------

events_df['event_timestamp'] = pd.to_datetime(events_df['event_timestamp'], errors='coerce')
spend_df['date'] = pd.to_datetime(spend_df['date'], errors='coerce')
users_df['registration_date'] = pd.to_datetime(users_df['registration_date'], errors='coerce')
transactions_df['transaction_timestamp'] = pd.to_datetime(transactions_df['transaction_timestamp'], errors='coerce')
skan_df['postback_date'] = pd.to_datetime(skan_df['postback_date'], errors='coerce')

# COMMAND ----------

events_df["event_date"] = events_df["event_timestamp"].dt.floor("D")
events_df

# COMMAND ----------

# Filter event types
installs_daily = (
    events_df[events_df["event_name"] == "install"]
    .groupby(["event_date", "source", "campaign_id"], dropna=False)["user_id"]
    .nunique()
    .reset_index(name="installs")
)
regs_daily = (
    events_df[events_df["event_name"] == "registration"]
    .groupby(["event_date", "source", "campaign_id"], dropna=False)["user_id"]
    .nunique()
    .reset_index(name="registrations")
)

# COMMAND ----------

installs_daily.head(19)

# COMMAND ----------

spend_daily = (
    spend_df.assign(event_date=spend_df["date"].dt.floor("D"))
    .groupby(["event_date", "channel", "campaign_id"], dropna=False)
    .agg(
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        campaign_name=("campaign_name", "first"),
    )
    .reset_index()
)

# COMMAND ----------

#a union of keys
keys = pd.concat([
    spend_daily[["event_date","channel","campaign_id"]],
    installs_daily.rename(columns={"source":"channel"})[["event_date","channel","campaign_id"]],
    regs_daily.rename(columns={"source":"channel"})[["event_date","channel","campaign_id"]],
], ignore_index=True).drop_duplicates()

keys.head(19)

# COMMAND ----------

# Merge everything onto keys
panel = (keys
    .merge(spend_daily, on=["event_date","channel","campaign_id"], how="left")
    .merge(installs_daily.rename(columns={"source":"channel"}), on=["event_date","channel","campaign_id"], how="left")
    .merge(regs_daily.rename(columns={"source":"channel"}), on=["event_date","channel","campaign_id"], how="left")
)

panel[["spend","impressions","clicks","installs","registrations"]] = panel[
    ["spend","impressions","clicks","installs","registrations"]
].fillna(0)

# COMMAND ----------

panel

# COMMAND ----------

# Cost per Click (CPC): total_spend / clicks 
# Cost per Mille (CPM): (total_spend / impressions) * 1000
# Install rate per impression: installs / impressions * 1000 (installs per thousand impressions)
# prevent prevent inf/NaN 

panel["CPI"] = np.where(panel["installs"] > 0, panel["spend"] / panel["installs"], np.nan)
panel["CAC"] = np.where(panel["registrations"] > 0, panel["spend"] / panel["registrations"], np.nan)

panel["CTR"] = np.where(panel["impressions"] > 0, 100 * panel["clicks"] / panel["impressions"], np.nan)
panel["CPC"] = np.where(panel["clicks"] > 0, panel["spend"] / panel["clicks"], np.nan)
panel["CPM"] = np.where(panel["impressions"] > 0, 1000 * panel["spend"] / panel["impressions"], np.nan)
panel

# COMMAND ----------

reg_users = (
    events_df[events_df["event_name"] == "registration"]
    .sort_values("event_timestamp")
    .drop_duplicates("user_id", keep="first")
    .loc[:, ["user_id", "event_timestamp", "source", "campaign_id"]]
    .rename(columns={"event_timestamp": "reg_date", "source": "channel"})
)
reg_users

# COMMAND ----------

import numpy as np
import pandas as pd

# ----------------------------
# Part 1 — Channel Performance
# ----------------------------

fee_rate = 0.005
window_days = 30
CAMPAIGN_MISSING = "__NO_CAMPAIGN__"   # critical so merges work (NaN keys don't match)

# 0) Ensure panel has a non-null campaign_id key (otherwise merges won't match Organic/Referral)
panel = panel.copy()
panel["campaign_id"] = panel["campaign_id"].fillna(CAMPAIGN_MISSING)

# 1) Build 1-row-per-user registration mapping (earliest registration)
reg_users = (
    events_df[events_df["event_name"] == "registration"]
    .sort_values("event_timestamp")
    .drop_duplicates("user_id", keep="first")
    .loc[:, ["user_id", "event_timestamp", "source", "campaign_id"]]
    .rename(columns={"event_timestamp": "reg_date", "source": "channel"})
)
reg_users["campaign_id"] = reg_users["campaign_id"].fillna(CAMPAIGN_MISSING)

# 2) Interval method: 30-day AUM fee revenue per registered user
tx = (transactions_df
      .merge(reg_users[["user_id", "reg_date", "channel", "campaign_id"]], on="user_id", how="inner")
      .sort_values(["user_id", "transaction_timestamp"])
)

tx["window_end"] = tx["reg_date"] + pd.Timedelta(days=window_days)

tx = tx[(tx["transaction_timestamp"] >= tx["reg_date"]) &
        (tx["transaction_timestamp"] <= tx["window_end"])].copy()

tx["next_ts"] = tx.groupby("user_id")["transaction_timestamp"].shift(-1)
tx["next_ts"] = tx["next_ts"].fillna(tx["window_end"])
tx["next_ts"] = tx["next_ts"].where(tx["next_ts"] <= tx["window_end"], tx["window_end"])

tx["delta_days"] = (tx["next_ts"] - tx["transaction_timestamp"]).dt.total_seconds() / 86400.0

# fee revenue over each interval
tx["interval_fee"] = tx["running_balance_usd"] * tx["delta_days"] * (fee_rate / 365.0)

# revenue in first 30 days per user
user_rev_30d = (
    tx.groupby("user_id", as_index=False)["interval_fee"]
      .sum()
      .rename(columns={"interval_fee": "revenue_30d"})
)

# 3) Attach revenue back onto ALL registered users (so non-transactors count as 0)
reg_with_rev = (
    reg_users[["user_id", "channel", "campaign_id"]]
    .merge(user_rev_30d, on="user_id", how="left")
    .fillna({"revenue_30d": 0.0})
)

# avg revenue per registered user (campaign-grain)
avg_rev_campaign = (
    reg_with_rev.groupby(["channel", "campaign_id"], dropna=False)["revenue_30d"]
    .mean()
    .reset_index(name="avg_rev_30d_per_reg")
)

# total revenue in 30d (campaign-grain) — useful for ROAS-like metric
rev_campaign_total = (
    reg_with_rev.groupby(["channel", "campaign_id"], dropna=False)["revenue_30d"]
    .sum()
    .reset_index(name="revenue_30d_total")
)

# 4) Campaign-level KPI table over FULL period (stable, less noisy than daily)
campaign_stats = (
    panel.groupby(["channel", "campaign_id"], dropna=False)[
        ["spend", "impressions", "clicks", "installs", "registrations"]
    ].sum().reset_index()
)

# Core performance metrics
campaign_stats["CPI"] = np.where(campaign_stats["installs"] > 0,
                                 campaign_stats["spend"] / campaign_stats["installs"], np.nan)

campaign_stats["CAC"] = np.where(campaign_stats["registrations"] > 0,
                                 campaign_stats["spend"] / campaign_stats["registrations"], np.nan)

campaign_stats["install_to_reg_rate"] = np.where(campaign_stats["installs"] > 0,
                                                 campaign_stats["registrations"] / campaign_stats["installs"], np.nan)

# Media metrics (only when denominators exist)
campaign_stats["CTR"] = np.where(campaign_stats["impressions"] > 0,
                                 campaign_stats["clicks"] / campaign_stats["impressions"], np.nan)

campaign_stats["CPC"] = np.where(campaign_stats["clicks"] > 0,
                                 campaign_stats["spend"] / campaign_stats["clicks"], np.nan)

campaign_stats["CPM"] = np.where(campaign_stats["impressions"] > 0,
                                 1000 * campaign_stats["spend"] / campaign_stats["impressions"], np.nan)

# Optional but useful: click→install and click→registration efficiency
campaign_stats["click_to_install"] = np.where(campaign_stats["clicks"] > 0,
                                             campaign_stats["installs"] / campaign_stats["clicks"], np.nan)

campaign_stats["click_to_reg"] = np.where(campaign_stats["clicks"] > 0,
                                         campaign_stats["registrations"] / campaign_stats["clicks"], np.nan)

# Merge value metrics
campaign_stats = (campaign_stats
                  .merge(avg_rev_campaign, on=["channel", "campaign_id"], how="left")
                  .merge(rev_campaign_total, on=["channel", "campaign_id"], how="left"))

campaign_stats["avg_rev_30d_per_reg"] = campaign_stats["avg_rev_30d_per_reg"].fillna(0.0)
campaign_stats["revenue_30d_total"] = campaign_stats["revenue_30d_total"].fillna(0.0)

# Payback: CAC divided by 30-day revenue per registered user
# (Interpretation: "how many ~30-day periods to earn back CAC")
campaign_stats["payback_months"] = np.where(
    (campaign_stats["CAC"].notna()) & (campaign_stats["avg_rev_30d_per_reg"] > 0),
    campaign_stats["CAC"] / campaign_stats["avg_rev_30d_per_reg"],
    np.nan
)

# ROAS-like proxy (30-day fee revenue vs spend)
campaign_stats["roas_30d"] = np.where(
    campaign_stats["spend"] > 0,
    campaign_stats["revenue_30d_total"] / campaign_stats["spend"],
    np.nan
)

# 5) Channel-level "big picture" table (what CMO uses for budget shifts)
channel_stats = (
    campaign_stats.groupby("channel", dropna=False)[
        ["spend", "impressions", "clicks", "installs", "registrations", "revenue_30d_total"]
    ].sum().reset_index()
)

channel_stats["CPI"] = np.where(channel_stats["installs"] > 0,
                                channel_stats["spend"] / channel_stats["installs"], np.nan)
channel_stats["CAC"] = np.where(channel_stats["registrations"] > 0,
                                channel_stats["spend"] / channel_stats["registrations"], np.nan)
channel_stats["install_to_reg_rate"] = np.where(channel_stats["installs"] > 0,
                                                channel_stats["registrations"] / channel_stats["installs"], np.nan)

channel_stats["CTR"] = np.where(channel_stats["impressions"] > 0,
                                channel_stats["clicks"] / channel_stats["impressions"], np.nan)
channel_stats["CPC"] = np.where(channel_stats["clicks"] > 0,
                                channel_stats["spend"] / channel_stats["clicks"], np.nan)
channel_stats["CPM"] = np.where(channel_stats["impressions"] > 0,
                                1000 * channel_stats["spend"] / channel_stats["impressions"], np.nan)

# Avg revenue per registered user at channel level (includes 0-revenue users)
avg_rev_channel = (
    reg_with_rev.groupby("channel", dropna=False)["revenue_30d"]
    .mean()
    .reset_index(name="avg_rev_30d_per_reg")
)

channel_stats = channel_stats.merge(avg_rev_channel, on="channel", how="left").fillna({"avg_rev_30d_per_reg": 0.0})

channel_stats["payback_months"] = np.where(
    (channel_stats["CAC"].notna()) & (channel_stats["avg_rev_30d_per_reg"] > 0),
    channel_stats["CAC"] / channel_stats["avg_rev_30d_per_reg"],
    np.nan
)

channel_stats["roas_30d"] = np.where(
    channel_stats["spend"] > 0,
    channel_stats["revenue_30d_total"] / channel_stats["spend"],
    np.nan
)

# Nice-to-have: share metrics (helps "over/undervalued" narrative)
total_spend = channel_stats["spend"].sum()
total_regs = channel_stats["registrations"].sum()
channel_stats["spend_share"] = np.where(total_spend > 0, channel_stats["spend"] / total_spend, np.nan)
channel_stats["reg_share"] = np.where(total_regs > 0, channel_stats["registrations"] / total_regs, np.nan)
channel_stats["reg_minus_spend_share"] = channel_stats["reg_share"] - channel_stats["spend_share"]

# COMMAND ----------

campaign_stats

# COMMAND ----------

channel_stats

# COMMAND ----------

panel

# COMMAND ----------

# campaign_stats = panel.groupby(["channel", "campaign_id"])[["spend", "registrations"]].sum().reset_index().rename(columns={"spend": "total_spend","registrations": "total_regs" })
# campaign_stats["CAC_campaign"] = campaign_stats["total_spend"]/ campaign_stats["total_regs"]


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Filter out Organic to keep the scale meaningful for paid channels
paid_channels = channel_stats[channel_stats['channel'] != 'Organic']

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=paid_channels.sort_values('CAC'), x='CAC', y='channel', palette='viridis')

plt.title('CAC by Channel (Lower is Better)', fontsize=14)
plt.xlabel('Cost Per Acquisition ($)')
plt.ylabel('Channel')

# Add data labels
ax.bar_label(ax.containers[0], fmt='%.2f', padding=5)
plt.show()


# COMMAND ----------

plt.figure(figsize=(10, 6))
# Sorting by payback_months to show most efficient at the top
ax = sns.barplot(data=paid_channels.sort_values('payback_months'), x='payback_months', y='channel', palette='magma')

plt.title('Payback Period by Channel (Lower is Faster ROI)', fontsize=14)
plt.xlabel('Months to Payback')
plt.ylabel('Channel')

# Add data labels
ax.bar_label(ax.containers[0], fmt='%.1f', padding=5)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Attribution Gaps
# MAGIC - if Organic share spikes when tracking is disabled (especially on iOS), then “Organic” includes unattributed paid → paid channels look worse than reality.
# MAGIC - compare Android vs. iOS :  Android: Higher percentage of "Paid" installs vs iOS: significantly larger "Organic" count.
# MAGIC
# MAGIC ### SKAN (StoreKit Ad Network) is Apple's private way of reporting installs without identifying the specific user.
# MAGIC - comparing app_events.csv (which misses the "Ask Not to Track" users) against the skan_attribution_ios.csv (which counts them anonymously), actually calculate exactly how many paid users were wrongly labeled as Organic.
# MAGIC - If skan_attribution_ios.csv says TikTok got 500 installs on iOS, but app_events.csv only shows 100,=> 400 of "Organic" iOS users actually came from TikTok.

# COMMAND ----------

# --- Base installs table (unique users) ---
installs = events_df[events_df["event_name"] == "install"].copy()
installs["event_date"] = installs["event_timestamp"].dt.floor("D")

# Join tracking_enabled (may be NaN for some; keep them as Unknown)
installs_u = installs.merge(
    users_df[["user_id", "platform", "tracking_enabled"]],
    on=["user_id", "platform"],
    how="left"
)
installs_u["tracking_bucket"] = installs_u["tracking_enabled"].map({True: "Tracking ON", False: "Tracking OFF"})
installs_u["tracking_bucket"] = installs_u["tracking_bucket"].fillna("Unknown")

installs_u["is_organic"] = installs_u["source"].eq("Organic")
installs_u["is_paid"] = ~installs_u["is_organic"]  # includes Referral as paid here

# 1) Organic-to-paid ratio by platform
plat_totals = installs_u.groupby("platform")["user_id"].nunique().rename("installs_total")
plat_org = installs_u[installs_u["is_organic"]].groupby("platform")["user_id"].nunique().rename("installs_organic")
plat_paid = installs_u[installs_u["is_paid"]].groupby("platform")["user_id"].nunique().rename("installs_paid")

organic_paid_by_platform = pd.concat([plat_totals, plat_org, plat_paid], axis=1).fillna(0)
organic_paid_by_platform["organic_share"] = organic_paid_by_platform["installs_organic"] / organic_paid_by_platform["installs_total"]
organic_paid_by_platform["organic_to_paid_ratio"] = np.where(
    organic_paid_by_platform["installs_paid"] > 0,
    organic_paid_by_platform["installs_organic"] / organic_paid_by_platform["installs_paid"],
    np.nan
)

print("Organic vs Paid by platform:")
display(organic_paid_by_platform.reset_index())

# 2) Organic share by platform x tracking_enabled
tmp = (installs_u
       .groupby(["platform", "tracking_bucket"])
       .agg(
           installs=("user_id","nunique"),
           organic_installs=("is_organic","sum"),
       )
       .reset_index()
)
tmp["organic_share"] = tmp["organic_installs"] / tmp["installs"]

print("\nOrganic share by platform & tracking:")
display(tmp)

# --- Simple visualization: Organic share by platform and tracking bucket ---
pivot = tmp.pivot(index="platform", columns="tracking_bucket", values="organic_share").fillna(0)

ax = pivot.plot(kind="bar", figsize=(9,4))
ax.set_title("Organic Share of Installs by Platform and Tracking Status")
ax.set_ylabel("Organic share")
ax.set_xlabel("Platform")
ax.legend(title="")
plt.tight_layout()
plt.show()

# 3) Paid-channel mix by platform & tracking (to see which channels lose attribution)
paid_mix = (installs_u[installs_u["is_paid"]]
            .groupby(["platform", "tracking_bucket", "source"])["user_id"]
            .nunique()
            .reset_index(name="installs_paid")
)

# share within each platform+tracking group
paid_mix["share_within_group"] = paid_mix["installs_paid"] / paid_mix.groupby(["platform","tracking_bucket"])["installs_paid"].transform("sum")
display(paid_mix.sort_values(["platform","tracking_bucket","installs_paid"], ascending=[True,True,False]).head(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1.3 — Anomaly Detection
# MAGIC
# MAGIC - Test A: If Organic installs rise on the same day (or 1–7 days later) when paid spend spikes → last-click likely misallocates credit.
# MAGIC - Test B: paid spend steady but attributed paid installs drop
# MAGIC - Test C: Compare SKAN install_count trends per network to iOS last-click paid installs. SKAN is aggregated and privacy-preserving, so divergence can highlight measurement gaps.
# MAGIC
# MAGIC Example: On March 1st, for the TikTok channel:
# MAGIC - SKAN Data (iOS only): Total TikTok installs for iOS. (Let's say 500)
# MAGIC - Last-Click Data (Filtered): Total TikTok installs for iOS only. (Let's say 100)
# MAGIC - The Gap: 400 iOS users who came from TikTok but were hidden by privacy settings.

# COMMAND ----------

# Daily installs by platform x source
daily_installs = (installs
    .groupby(["event_date","platform","source"])["user_id"]
    .nunique()
    .reset_index(name="installs")
)

# Daily spend by channel
spend_daily = (spend_df.assign(event_date=spend_df["date"].dt.floor("D"))
    .groupby(["event_date","channel"])[["spend","impressions","clicks"]]
    .sum()
    .reset_index()
)

# Create daily totals: organic installs and paid installs (per platform)
daily_pivot = daily_installs.pivot_table(
    index=["event_date","platform"],
    columns="source",
    values="installs",
    aggfunc="sum",
    fill_value=0
).reset_index()

# Organic + paid totals
daily_pivot["organic_installs"] = daily_pivot.get("Organic", 0)
daily_pivot["paid_installs"] = daily_pivot.drop(columns=["event_date","platform","Organic"], errors="ignore").sum(axis=1)
daily_pivot["organic_share"] = daily_pivot["organic_installs"] / (daily_pivot["organic_installs"] + daily_pivot["paid_installs"]).replace(0, np.nan)

display(daily_pivot.head())

# COMMAND ----------

# Test A — Correlation of paid spend with organic installs (same-day + lags)
# Total paid spend per day (sum across channels) - you can also do this per channel
total_spend_daily = spend_daily.groupby("event_date")["spend"].sum().reset_index(name="paid_spend_total")

# Merge spend into daily_pivot (per platform)
ts = daily_pivot.merge(total_spend_daily, on="event_date", how="left").fillna({"paid_spend_total": 0})

def lag_corr(df, x_col, y_col, lags=range(0, 15)):
    out = []
    for lag in lags:
        out.append({
            "lag_days": lag,
            "corr": df[x_col].corr(df[y_col].shift(lag))
        })
    return pd.DataFrame(out)

for plat in ts["platform"].unique():
    d = ts[ts["platform"] == plat].sort_values("event_date")
    corr_df = lag_corr(d, "paid_spend_total", "organic_installs", lags=range(0, 15))
    print(f"\nLag correlation paid spend -> organic installs ({plat})")
    display(corr_df)

    plt.figure(figsize=(8,3))
    plt.plot(corr_df["lag_days"], corr_df["corr"], marker="o")
    plt.title(f"Correlation: Paid Spend vs Organic Installs (lagged) — {plat}")
    plt.xlabel("Lag (days)")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# Test B — Spike/Drop anomaly detection on Organic share (rolling z-score)
def rolling_zscore(s, window=14):
    mu = s.rolling(window, min_periods=max(3, window//2)).mean()
    sig = s.rolling(window, min_periods=max(3, window//2)).std()
    z = (s - mu) / sig
    z = z.replace([np.inf, -np.inf], np.nan)
    return z

threshold = 3.0  # try 2.5 if nothing shows up

anomaly_rows = []

for plat in ts["platform"].dropna().unique():
    d = ts[ts["platform"] == plat].sort_values("event_date").copy()

    # Ensure organic_share has valid numeric values
    d["organic_share"] = d["organic_share"].replace([np.inf, -np.inf], np.nan)

    d["z_organic_share"] = rolling_zscore(d["organic_share"], window=14)

    flagged = d.loc[d["z_organic_share"].abs() >= threshold,
                    ["event_date","platform","organic_share","z_organic_share","paid_spend_total"]].copy()
    if not flagged.empty:
        anomaly_rows.append(flagged)

    # Plot organic share; mark any flagged points (if none, scatter is empty and that's fine)
    plt.figure(figsize=(10,3))
    plt.plot(d["event_date"], d["organic_share"])
    plt.scatter(
        d.loc[d["z_organic_share"].abs()>=threshold, "event_date"],
        d.loc[d["z_organic_share"].abs()>=threshold, "organic_share"]
    )
    plt.title(f"Organic Share Over Time (|z|≥{threshold} marked) — {plat}")
    plt.xlabel("Date")
    plt.ylabel("Organic share")
    plt.tight_layout()
    plt.show()

# Safe concat
if anomaly_rows:
    anomalies = pd.concat(anomaly_rows, ignore_index=True)
else:
    anomalies = pd.DataFrame(columns=["event_date","platform","organic_share","z_organic_share","paid_spend_total"])

try:
    display(anomalies.sort_values("z_organic_share", ascending=False).head(20))
except:
    print("Anomalies table is empty")


# Even if no anomalies pass threshold, show top 10 extreme days per platform
top_extremes = []
for plat in ts["platform"].dropna().unique():
    d = ts[ts["platform"] == plat].sort_values("event_date").copy()
    d["z_organic_share"] = rolling_zscore(d["organic_share"], window=14)
    top_extremes.append(
        d.loc[d["z_organic_share"].notna(), ["event_date","platform","organic_share","z_organic_share","paid_spend_total"]]
         .assign(abs_z=lambda x: x["z_organic_share"].abs())
         .sort_values("abs_z", ascending=False)
         .head(10)
    )
top_extremes = pd.concat(top_extremes, ignore_index=True) if top_extremes else pd.DataFrame()
print("Top 10 most extreme days (even if none cross threshold):")
display(top_extremes.sort_values("abs_z", ascending=False).head(20))

# COMMAND ----------

#Test C — SKAN vs last-click iOS paid installs (trend comparison)
# SKAN daily installs (aggregated by network)
skan_daily = (skan_df
    .assign(event_date=skan_df["postback_date"].dt.floor("D"))
    .groupby(["event_date","network"])["install_count"]
    .sum()
    .reset_index()
)

# Last-click paid installs on iOS (exclude Organic)
ios_paid = (installs
    .query("platform == 'ios' and source != 'Organic'")
    .groupby(["event_date","source"])["user_id"]
    .nunique()
    .reset_index(name="lastclick_paid_installs")
)

# Compare totals over time (SKAN total vs last-click attributed paid installs)
skan_total = skan_daily.groupby("event_date")["install_count"].sum().reset_index(name="skan_installs_total")
ios_paid_total = ios_paid.groupby("event_date")["lastclick_paid_installs"].sum().reset_index(name="lastclick_paid_installs_total")

cmp = skan_total.merge(ios_paid_total, on="event_date", how="outer").fillna(0).sort_values("event_date")

plt.figure(figsize=(10,3))
plt.plot(cmp["event_date"], cmp["skan_installs_total"], label="SKAN installs (total)")
plt.plot(cmp["event_date"], cmp["lastclick_paid_installs_total"], label="Last-click paid installs (iOS)")
plt.title("iOS: SKAN Install Trend vs Last-click Paid Installs")
plt.xlabel("Date")
plt.ylabel("Installs")
plt.legend()
plt.tight_layout()
plt.show()

cmp["skan_minus_lastclick"] = cmp["skan_installs_total"] - cmp["lastclick_paid_installs_total"]
display(cmp.sort_values("skan_minus_lastclick", ascending=False).head(15))

# COMMAND ----------

cmp2 = cmp.copy()

for shift in [0,1,2,3,4]:
    shifted = cmp2.copy()
    shifted["skan_shifted"] = shifted["skan_installs_total"].shift(shift)
    corr = shifted["skan_shifted"].corr(shifted["lastclick_paid_installs_total"])
    mae = (shifted["skan_shifted"] - shifted["lastclick_paid_installs_total"]).abs().mean()
    print(f"Shift SKAN by {shift} day(s): corr={corr:.3f}, MAE={mae:.2f}")

# COMMAND ----------

# Test D : measure the statistical relationship between Paid Spend and Organic Installs. If Organic users are truly organic, they shouldn't care if paid spend $0 or $10,000 on TikTok today. However, if there is a high correlation, it proves that "Organic" is actually just "Untracked Paid."

# daily organic installs (all platforms OR do per platform)
daily_org = (installs[installs["source"]=="Organic"]
             .groupby("event_date")["user_id"].nunique()
             .reset_index(name="organic_installs"))

daily_spend = (spend_df.assign(event_date=spend_df["date"].dt.floor("D"))
               .groupby("event_date")["spend"].sum()
               .reset_index(name="paid_spend_total"))

df = (daily_spend.merge(daily_org, on="event_date", how="left")
      .fillna({"organic_installs": 0})
      .sort_values("event_date"))

# Lag correlation (raw)
for lag in [0,1,3,7]:
    corr = df["paid_spend_total"].corr(df["organic_installs"].shift(-lag))
    print(f"corr(spend, organic with +{lag}d lag) = {corr:.3f}")

# Lag correlation (differenced)
df["d_spend"] = df["paid_spend_total"].diff()
df["d_org"] = df["organic_installs"].diff()

for lag in [0,1,3,7]:
    corr = df["d_spend"].corr(df["d_org"].shift(-lag))
    print(f"corr(diff spend, diff organic with +{lag}d lag) = {corr:.3f}")

# Quick plot
plt.figure(figsize=(10,3))
plt.plot(df["event_date"], df["paid_spend_total"], label="Paid spend")
plt.plot(df["event_date"], df["organic_installs"], label="Organic installs")
plt.title("Paid Spend and Organic Installs Over Time")
plt.xlabel("Date"); plt.ylabel("Value")
plt.legend(); plt.tight_layout(); plt.show()

# COMMAND ----------

plt.figure(figsize=(5,4))
plt.scatter(df["paid_spend_total"], df["organic_installs"], s=10)
plt.title("Paid Spend vs Organic Installs (Daily)")
plt.xlabel("Paid spend (total)")
plt.ylabel("Organic installs")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Test E: Share of Voice (SoV) vs Share of Installs
# If a channel’s spend share increases, but its attributed paid install share doesn’t, and Organic share rises, that’s evidence of measurement reallocation to Organic.

# Weekly spend share by channel
spend_w = (spend_df.assign(week=spend_df["date"].dt.to_period("W").dt.start_time)
           .groupby(["week","channel"])["spend"].sum()
           .reset_index())

spend_w["spend_share"] = spend_w["spend"] / spend_w.groupby("week")["spend"].transform("sum")

# Weekly installs share by source (using install events)
inst_w = (installs.assign(week=installs["event_date"].dt.to_period("W").dt.start_time)
          .groupby(["week","source"])["user_id"].nunique()
          .reset_index(name="installs"))

# Paid installs share among paid sources
paid_sources = inst_w[inst_w["source"]!="Organic"].copy()
paid_sources["paid_install_share"] = paid_sources["installs"] / paid_sources.groupby("week")["installs"].transform("sum")

# Organic share of all installs (to see jumps)
tot_w = inst_w.groupby("week")["installs"].sum().reset_index(name="total_installs")
org_w = inst_w[inst_w["source"]=="Organic"][["week","installs"]].rename(columns={"installs":"organic_installs"})
org_w = tot_w.merge(org_w, on="week", how="left").fillna({"organic_installs":0})
org_w["organic_share_all"] = org_w["organic_installs"] / org_w["total_installs"]

# Example plot: TikTok spend share vs TikTok paid install share + organic share
channel_name = "TikTok"

tik_spend = spend_w[spend_w["channel"]==channel_name][["week","spend_share"]].rename(columns={"spend_share":"tiktok_spend_share"})
tik_inst = paid_sources[paid_sources["source"]==channel_name][["week","paid_install_share"]].rename(columns={"paid_install_share":"tiktok_paid_install_share"})

sov = tik_spend.merge(tik_inst, on="week", how="outer").merge(org_w[["week","organic_share_all"]], on="week", how="left").fillna(0).sort_values("week")

plt.figure(figsize=(10,3))
plt.plot(sov["week"], sov["tiktok_spend_share"], label="TikTok spend share")
plt.plot(sov["week"], sov["tiktok_paid_install_share"], label="TikTok paid install share (last-click)")
plt.plot(sov["week"], sov["organic_share_all"], label="Organic share (all installs)")
plt.title("SoV vs Attributed Install Share (TikTok) + Organic Share")
plt.xlabel("Week"); plt.ylabel("Share")
plt.legend(); plt.tight_layout(); plt.show()

# COMMAND ----------

sov2 = sov.sort_values("week").copy()
sov2["d_tiktok_spend_share"] = sov2["tiktok_spend_share"].diff()
sov2["d_organic_share"] = sov2["organic_share_all"].diff()
sov2["d_tiktok_paid_install_share"] = sov2["tiktok_paid_install_share"].diff()

print("corr(Δ TikTok spend share, Δ Organic share) =",
      sov2["d_tiktok_spend_share"].corr(sov2["d_organic_share"]))

print("corr(Δ TikTok spend share, Δ TikTok install share) =",
      sov2["d_tiktok_spend_share"].corr(sov2["d_tiktok_paid_install_share"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1.5 funnel metrics

# COMMAND ----------

import numpy as np
import pandas as pd

# --- Campaign-level media metrics from spend_df ---
spend_campaign = (
    spend_df.groupby(["channel","campaign_id","campaign_name"], dropna=False)
    .agg(
        spend=("spend","sum"),
        impressions=("impressions","sum"),
        clicks=("clicks","sum")
    )
    .reset_index()
)

spend_campaign["CTR"] = np.where(spend_campaign["impressions"]>0,
                                 spend_campaign["clicks"]/spend_campaign["impressions"],
                                 np.nan)
spend_campaign["CPM"] = np.where(spend_campaign["impressions"]>0,
                                 1000*spend_campaign["spend"]/spend_campaign["impressions"],
                                 np.nan)
spend_campaign["CPC"] = np.where(spend_campaign["clicks"]>0,
                                 spend_campaign["spend"]/spend_campaign["clicks"],
                                 np.nan)

# --- Join with conversion/value metrics from your campaign_stats (Part 1.1) ---
# campaign_stats should already include installs, registrations, CPI, CAC, payback_months, roas_30d, etc.
role_df = campaign_stats.merge(
    spend_campaign[["channel", "campaign_id", "campaign_name"]], 
    on=["channel", "campaign_id"],
    how="left"
)


# Derived funnel metrics
role_df["install_rate_per_click"] = np.where(role_df["clicks"]>0, role_df["installs"]/role_df["clicks"], np.nan)
role_df["reg_rate_per_click"] = np.where(role_df["clicks"]>0, role_df["registrations"]/role_df["clicks"], np.nan)
role_df["install_to_reg_rate"] = np.where(role_df["installs"]>0, role_df["registrations"]/role_df["installs"], np.nan)

# Optional: filter out tiny campaigns for stable interpretation
role_df_stable = role_df[role_df["registrations"] >= 200].copy()

# COMMAND ----------

import re

def classify_campaign(name: str, channel: str) -> str:
    name = str(name or "").lower()

    # strong keywords
    if re.search(r"\bbrand\b|branded|peaks", name) and "google" in channel.lower():
        return "Brand / demand-capture"
    if re.search(r"retarget|remarket|re-engage|reengage|crm", name):
        return "Retargeting / re-engagement"
    if re.search(r"prospect|acq|acquisition|upper|awareness|reach|video|view", name):
        return "Prospecting / upper-funnel"
    if "non-brand" in channel.lower() or re.search(r"non[- ]brand|generic|competitor", name):
        return "Non-brand intent"
    if re.search(r"app campaign|uac|app|install", name) and "google" in channel.lower():
        return "Automated app acquisition"
    return "Unclear / mixed"

role_df_stable["campaign_role"] = role_df_stable.apply(
    lambda r: classify_campaign(r.get("campaign_name",""), r.get("channel","")),
    axis=1
)

# COMMAND ----------

# Spend-weighted role distribution within each channel
role_mix = (
    role_df_stable.groupby(["channel","campaign_role"])["spend"]
    .sum()
    .reset_index(name="role_spend")
)
role_mix["role_spend_share"] = role_mix["role_spend"] / role_mix.groupby("channel")["role_spend"].transform("sum")
role_mix = role_mix.sort_values(["channel","role_spend_share"], ascending=[True,False])

# KPI profile per role (within each channel)
role_profile = (
    role_df_stable.groupby(["channel","campaign_role"])
    .agg(
        campaigns=("campaign_id","nunique"),
        spend=("spend","sum"),
        CAC=("CAC","mean"),
        payback=("payback_months","mean"),
        CPM=("CPM","mean"),
        CTR=("CTR","mean"),
        install_to_reg=("install_to_reg_rate","mean"),
        roas_30d=("roas_30d","mean")
    )
    .reset_index()
)

display(role_mix)
display(role_profile.sort_values(["channel","spend"], ascending=[True,False]))

# COMMAND ----------

import matplotlib.pyplot as plt

d = role_df_stable.dropna(subset=["CPM","CTR"]).copy()

plt.figure(figsize=(8,4))
plt.scatter(d["CPM"], d["CTR"], s=np.clip(d["spend"]/2000, 10, 200))
plt.title("Campaign Media Profile: CPM vs CTR (bubble size ~ spend)")
plt.xlabel("CPM ($ per 1,000 impressions)")
plt.ylabel("CTR (clicks / impressions)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2
# MAGIC
# MAGIC Compute fee revenue from AUM (running balance) using your interval method:
# MAGIC - rev_0_H: revenue from day 0 to day H - H=180 days 
# MAGIC - rev_0_14: revenue from day 0 to day 14
# MAGIC - Label to predict: y = rev_14_H = rev_0_H − rev_0_14
# MAGIC
# MAGIC Build features from first 14 days: 
# MAGIC - user profile: age, gender, platform, tracking_enabled
# MAGIC - acquisition: source, campaign_id, country
# MAGIC - early funnel: did register within 14 days? (binary) + days to register
# MAGIC - early transactions (within 14 days): deposit sum/count, withdrawal sum/count, net flow, max/mean/end balance, etc.
# MAGIC - early engagement: counts of event types in 14 days
# MAGIC
# MAGIC Train / evaluate: 
# MAGIC - Train: earlier cohorts
# MAGIC - Test: later cohorts
# MAGIC - Exclude users whose install_date > data_end - H days (no full label).

# COMMAND ----------

# install date per user
fee_rate = 0.005
H_DAYS = 180
EARLY_DAYS = 14

# install timestamp per user (first install)
installs = events_df[events_df["event_name"] == "install"].copy()
installs = installs.sort_values("event_timestamp").drop_duplicates("user_id", keep="first")
installs = installs.rename(columns={"event_timestamp": "install_date",
                                    "source": "channel"})
installs = installs[["user_id", "install_date", "platform", "channel", "campaign_id", "country"]]

data_end = transactions_df["transaction_timestamp"].max()
cutoff_for_labels = data_end - pd.Timedelta(days=H_DAYS)

# keep only users with full H-day future observed
eligible = installs[installs["install_date"] <= cutoff_for_labels].copy()

# COMMAND ----------

# Fee revenue function (interval method) for any window [start, end]
def fee_revenue_interval(transactions, user_starts, start_col, end_col, fee_rate=0.005):
    """
    transactions: transactions_df (user_id, transaction_timestamp, running_balance_usd)
    user_starts: df with user_id, start_col, end_col (timestamps per user)
    Returns: df user_id, revenue_window
    """
    tx = (transactions
          .merge(user_starts[["user_id", start_col, end_col]], on="user_id", how="inner")
          .sort_values(["user_id", "transaction_timestamp"])
          .copy())

    # keep transactions inside the window
    tx = tx[(tx["transaction_timestamp"] >= tx[start_col]) &
            (tx["transaction_timestamp"] <= tx[end_col])].copy()

    # next transaction time per user
    tx["next_ts"] = tx.groupby("user_id")["transaction_timestamp"].shift(-1)
    tx["next_ts"] = tx["next_ts"].fillna(tx[end_col])
    tx["next_ts"] = tx["next_ts"].where(tx["next_ts"] <= tx[end_col], tx[end_col])

    tx["delta_days"] = (tx["next_ts"] - tx["transaction_timestamp"]).dt.total_seconds() / 86400.0
    tx["interval_fee"] = tx["running_balance_usd"] * tx["delta_days"] * (fee_rate / 365.0)

    out = (tx.groupby("user_id", as_index=False)["interval_fee"]
             .sum()
             .rename(columns={"interval_fee": "revenue_window"}))
    return out

# Build labels: rev_14_H, define per-user window endpoints
windows = eligible[["user_id", "install_date"]].copy()
windows["end_14"] = windows["install_date"] + pd.Timedelta(days=EARLY_DAYS)
windows["end_H"]  = windows["install_date"] + pd.Timedelta(days=H_DAYS)

# revenue 0->14
rev_0_14 = fee_revenue_interval(
    transactions_df, windows.rename(columns={"install_date":"start_0"}),
    start_col="start_0", end_col="end_14", fee_rate=fee_rate
).rename(columns={"revenue_window":"rev_0_14"})

# revenue 0->H
rev_0_H = fee_revenue_interval(
    transactions_df, windows.rename(columns={"install_date":"start_0"}),
    start_col="start_0", end_col="end_H", fee_rate=fee_rate
).rename(columns={"revenue_window":"rev_0_H"})

labels = (windows[["user_id"]]
          .merge(rev_0_14, on="user_id", how="left")
          .merge(rev_0_H,  on="user_id", how="left")
          .fillna({"rev_0_14":0.0, "rev_0_H":0.0}))

labels["y_rev_14_H"] = (labels["rev_0_H"] - labels["rev_0_14"]).clip(lower=0.0)

# COMMAND ----------

# Campaign-level aggregates from spend_df
camp = spend_df.groupby(["channel", "campaign_id"], dropna=False).agg(
    camp_spend=("spend", "sum"),
    camp_impressions=("impressions", "sum"),
    camp_clicks=("clicks", "sum")
).reset_index()

camp["camp_CTR"] = np.where(camp["camp_impressions"] > 0,
                            camp["camp_clicks"] / camp["camp_impressions"], np.nan)
camp["camp_CPM"] = np.where(camp["camp_impressions"] > 0,
                            1000 * camp["camp_spend"] / camp["camp_impressions"], np.nan)
camp["camp_CPC"] = np.where(camp["camp_clicks"] > 0,
                            camp["camp_spend"] / camp["camp_clicks"], np.nan)


# COMMAND ----------

# Registration within 14 days
regs = events_df[events_df["event_name"] == "registration"].copy()
regs = regs.sort_values("event_timestamp").drop_duplicates("user_id", keep="first")
regs = regs.rename(columns={"event_timestamp":"reg_date"})[["user_id","reg_date"]]

feat = eligible.merge(regs, on="user_id", how="left")
feat["reg_within_14d"] = (feat["reg_date"].notna()) & (feat["reg_date"] <= feat["install_date"] + pd.Timedelta(days=EARLY_DAYS))
feat["days_to_reg"] = (feat["reg_date"] - feat["install_date"]).dt.days
feat.loc[~feat["reg_within_14d"], "days_to_reg"] = np.nan

feat = feat.merge(camp[["channel", "campaign_id", "camp_CTR", "camp_CPM", "camp_CPC"]],   #Add campaign-level CPM/CTR/CPC
                  on=["channel", "campaign_id"], how="left")

feat["platform_norm"] = feat["platform"].astype(str).str.lower().str.strip()
feat["channel_platform"] = feat["channel"].astype(str) + "__" + feat["platform_norm"]     # Add platform×channel interaction

# Transaction features within first 14 days
tx14 = (transactions_df
        .merge(eligible[["user_id","install_date"]], on="user_id", how="inner")
        .copy())
tx14["end_14"] = tx14["install_date"] + pd.Timedelta(days=EARLY_DAYS)
tx14 = tx14[(tx14["transaction_timestamp"] >= tx14["install_date"]) &
            (tx14["transaction_timestamp"] <= tx14["end_14"])].copy()

# aggregates
tx_feat = (tx14.groupby("user_id")
           .agg(
               tx_count=("transaction_timestamp","count"),
               dep_sum=("amount_usd", lambda s: s[tx14.loc[s.index,"transaction_type"].eq("deposit")].sum()),
               dep_cnt=("transaction_type", lambda s: (s=="deposit").sum()),
               wdr_sum=("amount_usd", lambda s: s[tx14.loc[s.index,"transaction_type"].eq("withdrawal")].sum()),
               wdr_cnt=("transaction_type", lambda s: (s=="withdrawal").sum()),
               bal_max=("running_balance_usd","max"),
               bal_mean=("running_balance_usd","mean"),
               bal_last=("running_balance_usd","last"),
           )
           .reset_index())

tx_feat["net_flow"] = tx_feat["dep_sum"] - tx_feat["wdr_sum"]

feat = feat.merge(tx_feat, on="user_id", how="left").fillna({
    "tx_count":0, "dep_sum":0, "dep_cnt":0, "wdr_sum":0, "wdr_cnt":0,
    "bal_max":0, "bal_mean":0, "bal_last":0, "net_flow":0
})

feat = feat.merge(users_df[["user_id","age","gender","tracking_enabled"]], on="user_id", how="left")

# COMMAND ----------

# Assemble modeling dataset + time split
# Assemble modeling dataset + time split
dataset = (feat
           .merge(labels[["user_id", "y_rev_14_H", "rev_0_14"]], on="user_id", how="inner")  #Add rev_0_14 as a feature - Important
           .drop(columns=["reg_date"])  # keep install_date for splitting
          )

# time-based split (e.g., last 20% time as test)
dataset = dataset.sort_values("install_date")
split_idx = int(len(dataset) * 0.8)

train_df = dataset.iloc[:split_idx].copy()
test_df  = dataset.iloc[split_idx:].copy()

# Train a model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

target = "y_rev_14_H"

# features
drop_cols = ["user_id", "install_date", target]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target].values
X_test  = test_df.drop(columns=drop_cols)
y_test  = test_df[target].values

cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object" or str(X_train[c].dtype)=="bool"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ]
)

# log1p target to stabilize heavy tails
model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, random_state=42)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

pipe.fit(X_train, np.log1p(y_train))
pred = np.expm1(pipe.predict(X_test))

r2_log = r2_score(np.log1p(y_test), np.log1p(pred))
mae = mean_absolute_error(y_test, pred)
rmse = root_mean_squared_error(y_test, pred)
print(f"Test MAE={mae:.4f}, RMSE={rmse:.4f}, R2_log={r2_log:.4f}")

# COMMAND ----------


X_test_sample = X_test.copy()
y_test_sample = np.log1p(y_test)

result = permutation_importance(
    pipe,
    X_test_sample,
    y_test_sample,
    n_repeats=5,
    random_state=42,
    scoring="r2"
)

imp = (pd.DataFrame({
    "feature": X_test_sample.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
})
.sort_values("importance_mean", ascending=False))

display(imp.head(20))

# COMMAND ----------

# predictions
test_out = test_df[["user_id","channel","campaign_id"]].copy()
test_out["y_true"] = y_test
test_out["y_pred"] = pred
test_out["residual"] = test_out["y_true"] - test_out["y_pred"]

# residual check by channel (systematic over/under prediction)
resid_by_channel = (test_out.groupby("channel")
                    .agg(mean_resid=("residual","mean"),
                         median_resid=("residual","median"),
                         n=("residual","size"))
                    .reset_index()
                    .sort_values("mean_resid"))
print("residual check by channel")                    
print(resid_by_channel)

print("future revenue per install")
# predicted future revenue per install (or per registered user; you can filter reg_within_14d)
channel_value = (test_out.groupby("channel")
                 .agg(
                     pred_mean=("y_pred","mean"),
                     pred_median=("y_pred","median"),
                     pred_p75=("y_pred", lambda s: np.quantile(s, 0.75)),
                     pred_p90=("y_pred", lambda s: np.quantile(s, 0.90)),
                     share_zero_pred=("y_pred", lambda s: np.mean(s <= 1e-6)),
                     n=("y_pred","size")
                 )
                 .reset_index()
                 .sort_values("pred_mean", ascending=False))
display(channel_value)

# COMMAND ----------

# Merge with CAC (exclude Organic / missing CAC)
ltv_cac = channel_value.merge(channel_stats[["channel","CAC"]], on="channel", how="left")
ltv_cac = ltv_cac[ltv_cac["CAC"].notna()].copy()

# LTV proxy = predicted future revenue (rev_14_H) per user
ltv_cac["LTV_over_CAC"] = ltv_cac["pred_mean"] / ltv_cac["CAC"]

display(ltv_cac.sort_values("LTV_over_CAC", ascending=False))

# Bar chart (no custom colors)
import matplotlib.pyplot as plt

d = ltv_cac.sort_values("LTV_over_CAC", ascending=True)
plt.figure(figsize=(9,4))
plt.barh(d["channel"], d["LTV_over_CAC"])
plt.title("Predicted LTV / CAC by Channel (higher is better)")
plt.xlabel("LTV/CAC")
plt.tight_layout()
plt.show()

# COMMAND ----------

from sklearn.ensemble import HistGradientBoostingRegressor

median_model = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.5,
    max_depth=6, learning_rate=0.08, random_state=42
)

pipe_med = Pipeline([("prep", preprocess), ("model", median_model)])
pipe_med.fit(X_train, np.log1p(y_train))
pred_med = np.expm1(pipe_med.predict(X_test))

print("Median-model MAE:", mean_absolute_error(y_test, pred_med))
