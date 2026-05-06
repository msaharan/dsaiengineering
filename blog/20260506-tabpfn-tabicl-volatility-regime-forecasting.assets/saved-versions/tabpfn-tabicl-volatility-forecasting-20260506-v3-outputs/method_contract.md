# Volatility-Regime Workflow Contract

Target asset: SPY
Target: future 20-trading-day realized volatility high-regime indicator.
High-volatility threshold: 80% quantile estimated only through 2017-12-29.
Feature availability policy: exclude features with missing rate above 40% or fewer than 30 pre-calibration observations; retain missingness indicators for partially observed retained features; fit median imputers on pre-calibration data.
Splits: context through 2011-12-30; tuning through 2017-12-29; calibration through 2019-12-31; holdout from 2020-01-01.
Validation: rolling-origin chronological cross-validation with positive-count checks.
Run mode: full research run with fast-mode reductions inactive.
Default model families: rule scores, XGBoost on the configured device, direct TabPFN/TabICL, and XGBoost plus TabPFN/TabICL embeddings. Optional CPU benchmarks: Logistic Regression and Random Forest are available but disabled by default.
Decision diagnostic: pre-specified exposure-reduction policies fitted on calibration-window score quantiles and evaluated only on holdout.
Interpretation: educational workflow test; not investment advice; not a claim of market predictability or deployable trading performance.