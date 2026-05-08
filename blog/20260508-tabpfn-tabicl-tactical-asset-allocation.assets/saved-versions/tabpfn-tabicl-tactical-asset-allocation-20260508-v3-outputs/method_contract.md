# Tactical Asset-Allocation Workflow Contract

Universe: SPY, QQQ, IWM, TLT, IEF, GLD, HYG, EEM, VNQ
Task: score each asset at monthly signal date t for whether it will be in the top 3 assets by next-month total return inside the configured universe.
Target construction: next-month asset return is computed from adjusted monthly closes after the signal date; the label is assigned only within the same signal month.
Feature availability policy: features use data available at or before the monthly signal date; forward returns, forward excess returns, and top-k labels are excluded from model features. FRED series are included in model features only when they have enough non-missing observations in the model-selection window; unavailable optional credit-spread histories are recorded but not imputed into the training set.
Splits: context through 2011-12-31; model-selection history through 2017-12-31; calibration through 2019-12-31; holdout from 2020-01-01.
Validation: rolling-origin chronological cross-validation on monthly groups with positive-count checks.
Default model families: deterministic allocation rules, GPU XGBoost, direct TabPFN, and direct TabICL. CPU-only Logistic Regression is available but disabled by default.
Execution convention: scores are formed after month-end close data is available and are applied to the following close-to-close monthly return; the diagnostic does not model intraday tradability, taxes, market impact, or mandate constraints.
Portfolio diagnostic: monthly top-k equal-weight allocation from model scores, compared with equal-weight, SPY-only, 60/40 SPY/TLT, inverse-volatility, and deterministic momentum rules. The main portfolio table uses 5.0 basis points per unit of one-way turnover, with a separate transaction-cost sensitivity artifact for stress review.
Interpretation: educational workflow test; not investment advice; not a claim of market predictability or deployable trading performance.