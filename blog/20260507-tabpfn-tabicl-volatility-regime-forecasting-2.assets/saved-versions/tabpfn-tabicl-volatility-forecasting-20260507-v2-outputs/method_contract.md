# Volatility-Regime Workflow Contract

Target asset: SPY
Target: future 20-trading-day realized volatility high-regime indicator.
High-volatility threshold: 80% quantile estimated only through 2017-12-29.
Feature availability policy: exclude features with missing rate above 40% or fewer than 30 pre-calibration observations; retain missingness indicators for partially observed retained features; fit median imputers on pre-calibration data.
Splits: context through 2009-12-31; tuning through 2017-12-29; calibration through 2019-12-31; holdout from 2020-01-01.
Validation: rolling-origin chronological cross-validation with positive-count checks. The post-context tuning window starts after the TFM embedding context and now includes 2010-2017 to avoid selecting optional CPU benchmarks only on the quiet 2012-2017 slice.
Robustness diagnostics: target-threshold sensitivity over 0.7, 0.75, 0.8, 0.85, 0.9 and named holdout market phases covid_shock_2020, post_covid_reopen_2021, inflation_rate_shock_2022, quiet_2023, late_holdout_2024_2026.
Run mode: full research run with fast-mode reductions inactive.
Default model families: rule scores, a raw all-history XGBoost incumbent on the configured device, and direct TabPFN/TabICL classifiers. Optional CPU benchmarks: Logistic Regression and Random Forest are available but disabled by default.
Decision diagnostic: pre-specified exposure-reduction policies fitted on calibration-window score quantiles and evaluated only on holdout.
Interpretation: educational workflow test; not investment advice; not a claim of market predictability or deployable trading performance.