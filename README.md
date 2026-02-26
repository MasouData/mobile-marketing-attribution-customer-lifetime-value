# Mobile App Marketing Attribution & LTV Prediction

This repository contains a comprehensive data science project completed as part of a take-home assignment for a **mobile investment app** company. The goal was to evaluate marketing channel performance under last‑click attribution, uncover hidden patterns (attribution gaps, halo effects), and build a predictive model for customer lifetime value (LTV) to guide budget allocation.

## Project Overview

The company runs paid marketing campaigns across multiple channels (Google, Meta, TikTok, Referral) to acquire users. The current measurement relies on last‑click attribution, which may undervalue upper‑funnel channels (e.g., TikTok, Meta) due to privacy restrictions (iOS ATT) and delayed conversion effects. This project:

- Analyzes channel performance using last‑click metrics (CPI, CAC, conversion rates, payback period).
- Quantifies attribution gaps by platform and tracking consent.
- Detects anomalies and demonstrates that paid spend drives future organic installs (halo effect).
- Characterizes each channel’s role in the acquisition funnel (top/mid/bottom).
- Defines LTV as 180‑day fee revenue (0.5% annual fee on AUM) and builds a model to predict LTV using only first‑14‑day user behavior.
- Re‑evaluates channels using predicted LTV/CAC and payback.
- Proposes a method to estimate iOS user value from SKAdNetwork (SKAN) aggregated data.

## Data Sources

The analysis uses five simulated CSV files (not included in this repo for confidentiality):

- `marketing_spend.csv` – daily ad spend, impressions, clicks per campaign.
- `app_events.csv` – user‑level attribution events (impression, click, install, registration) with channel, platform, country.
- `user_profiles.csv` – demographics for registered users (age, gender, tracking consent).
- `user_transactions.csv` – deposit/withdrawal history with running balance.
- `skan_attribution_ios.csv` – SKAN postbacks (aggregated iOS installs by conversion value).

## Methodology

### Part 1 – Exploratory Analysis & Attribution

- **Channel Performance:** Aggregated spend, installs, registrations; computed CPI, CAC, conversion rate, CTR, CPM, and payback period (using 30‑day fee revenue proxy).
- **Attribution Gaps:** Compared organic/paid ratios by platform and tracking status; showed that iOS organic share is inflated due to untracked paid traffic.
- **Anomaly Detection:** Time‑series analysis of organic share, lagged correlations between paid spend and organic installs, and SKAN vs. last‑click installs on iOS revealed persistent bias.
- **Organic‑Paid Relationship:** Cross‑correlation and regression proved that paid spend significantly predicts future organic installs (halo effect).
- **Channel Roles:** Using CPM, CTR, and conversion rates, classified channels into top‑funnel (TikTok, Meta, Google App), mid‑funnel (Google Non‑Brand), and bottom‑funnel (Google Brand, Referral).

### Part 2 – Predictive Modeling

- **LTV Definition:** For each registered user, computed fee revenue over 180 days using an interval method on running balances (0.5% annual fee).
- **Early LTV Prediction:** Built a HistGradientBoosting model to predict LTV (after day 14) using features available within 14 days post‑install: early transaction activity, registration flag, demographics, channel, platform, country. Achieved high accuracy (R²=0.98 on log scale).
- **Channel Quality:** Aggregated predicted LTV by channel and computed LTV/CAC ratio and payback. Referral and Google Brand Search emerged as strongest; TikTok weakest but with caveats.
- **iOS Estimation:** Proposed a method to estimate iOS user value from SKAN data by mapping conversion values to average LTV using Android users as a calibration set.

### Part 3 – Strategic Recommendations

- **Budget Reallocation:** Increase Referral, maintain Brand Search, optimise Non‑Brand Search, reallocate within Meta/App Campaigns, and test TikTok incrementality.
- **Measurement Framework:** Move from last‑click to a blended approach: predicted LTV/CAC, platform‑splits, incrementality tests, SKAN reconciliation.
- **Quick Wins:** Dashboard fixes, campaign triage by LTV/CAC, and one immediate incrementality test.

## Key Findings

- Last‑click undervalues upper‑funnel channels (TikTok, Meta) because:
  - Privacy restrictions on iOS push paid installs into "Organic".
  - Paid spend drives future organic installs (halo effect) not captured.
- Referral and Google Brand Search are the most efficient in both last‑click and LTV views.
- Early transaction behavior (especially balance and deposits) is the strongest predictor of long‑term value.
- Campaign‑level heterogeneity within channels offers immediate optimisation opportunities.

## How to Run

1. Clone the repository.
2. Place the five CSV files in a `data/` folder (structure as described in the notebook).
3. Run the Jupyter notebook `notebooks/main_analysis.ipynb` to reproduce all analyses and plots.
4. The notebook requires Python 3.8+ with libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels.

## License

This project is for educational/demonstration purposes only. The data is simulated and not for commercial use.

## Author

**Masoud Aghayan**
