# Run Closeout

Artifact directory: tabpfn_tabicl_tactical_asset_allocation_20260511_outputs
Run mode: full research run
Rows in model frame: 6,059
Months in model frame: 243
Asset count: 25
Feature count: 956
Feature set variant: identity_ablated
Execution return mode: next_open_to_next_open
FRED series included in features: 5
FRED series excluded for insufficient selection history or load failure: 2
Holdout rows: 1,875
Holdout months: 75
Holdout top-k label rate: 20.0000%
Successful model rows: 22
Model errors: 0
TabICL enabled: False
Feature-variant sensitivity enabled: True
Multi-horizon audit months: [3, 6]
Benchmark asset: SPY

Important caveats: TabICL is disabled in this run; feature-variant sensitivity uses CPU Logistic Regression diagnostics; multi-horizon and benchmark-relative sections audit existing scores rather than retraining new objective-specific models; liquidity, tax, market-impact, and constrained-allocation outputs are stress proxies. Use the CSV/TXT/PNG artifacts for offline review because notebook editor output can be truncated.