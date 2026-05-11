# Run Closeout

Artifact directory: tabpfn_tabicl_tactical_asset_allocation_20260512_outputs
Run mode: full research run
Rows in model frame: 6,059
Months in model frame: 243
Asset count: 25
Feature count: 1,060
Feature set variant: identity_ablated
Execution return mode: next_open_to_next_open
FRED series included in features: 5
FRED series excluded for insufficient selection history or load failure: 2
Holdout rows: 1,875
Holdout months: 75
Holdout top-k label rate: 20.0000%
Successful model rows: 27
Model errors: 0
TabICL enabled: False (known unresolved by design when False)
Feature-variant sensitivity enabled: True
Selected model-family feature sensitivity enabled: True
Objective-specific retraining enabled: True
Multi-horizon audit months: [3, 6]
Benchmark asset: SPY

Important caveats: TabICL is disabled in this run and remains an unresolved empirical comparison by design; selected model-family feature sensitivity uses fixed GPU XGBoost and direct TabPFN by default, while objective-specific retraining uses fixed GPU XGBoost and direct TabPFN by default with calibration-window, monthly rank, fixed-model CV, and score-to-portfolio diagnostics. These are focused diagnostics, not exhaustive hyperparameter-search reruns of every target, feature policy, and model family. Liquidity, tax, market-impact, constrained-allocation, benchmark-relative portfolio, and multi-month objective portfolio outputs are stress or interpretive proxies. Use the CSV/TXT/PNG artifacts for offline review because notebook editor output can be truncated.