# Run Closeout

Artifact directory: tabpfn_tabicl_tactical_asset_allocation_20260512_v2_outputs
Run mode: FAST_MODE smoke test
Rows in model frame: 2,430
Months in model frame: 243
Asset count: 10
Feature count: 459
Feature set variant: identity_ablated
Execution return mode: next_open_to_next_open
FRED series included in features: 5
FRED series excluded for insufficient selection history or load failure: 2
Holdout rows: 750
Holdout months: 75
Holdout top-k label rate: 30.0000%
Successful model rows: 27
Model errors: 0
TabICL enabled: False (known unresolved by design when False)
Feature-variant sensitivity enabled: True
Selected model-family feature sensitivity enabled: True
Objective-specific retraining enabled: True
Multi-horizon audit months: [3, 6]
Benchmark asset: SPY

Important caveats: TabICL is disabled in this run and remains an unresolved empirical comparison by design; selected model-family feature sensitivity uses fixed GPU XGBoost and direct TabPFN by default and is saved separately from headline allocation outputs. Objective-specific retraining uses fixed GPU XGBoost, target-specific searched GPU XGBoost, and direct TabPFN by default with calibration-window, monthly rank, chronological CV, and horizon-aware score-to-return diagnostics. XGBoost searches use fold-level early stopping on chronological validation blocks to reduce wasted tree-building while preserving the same validation protocol and holdout boundary. These are focused diagnostics, not exhaustive hyperparameter-search reruns of every target, feature policy, and model family. Liquidity, tax, market-impact, ex-ante risk, constrained-allocation, benchmark-relative portfolio, and multi-month objective outputs are stress or interpretive proxies. Use the CSV/TXT/PNG artifacts for offline review because notebook editor output can be truncated.