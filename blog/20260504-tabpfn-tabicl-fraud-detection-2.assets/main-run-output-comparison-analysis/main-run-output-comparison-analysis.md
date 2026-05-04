# Main Run Output Comparison Analysis

Source notebooks:

- Main completed run: `tabpfn-tabicl-fraud-detection-20260504.ipynb`
- CPU-sec7 baseline: `tabpfn-tabicl-fraud-detection-20260504-with-sklearn-running-on-CPU-sec7-cell1.ipynb`

Extracted main-run tables, figures, streams, and comparison CSVs were saved in this folder. The CPU-sec7 CSVs and figures in `cpu-sec7-run-output-analysis` were used as the baseline.

## Run Integrity

- Main notebook execution errors: 0.
- CPU-sec7 notebook execution errors: 0.
- Main stderr outputs: 2. CPU-sec7 stderr outputs: 2.
- XGBoost mismatched-device warning present in main run: no.
- XGBoost mismatched-device warning present in CPU-sec7 run: yes.

The main run is structurally complete and the GPU/CuPy XGBoost patch appears to have removed the section-7 mismatched-device warning.

## Static Output Consistency

- `split_summary.csv`: matches.
- `cv_summary.csv`: semantically matches; exact text differs only because some displayed rates keep trailing zeros in the main run (`0.0020`, `0.0030`) while the CPU-sec7 CSV stores them as `0.002`, `0.003`.
- `duplicate_diagnostics.csv`: matches.
- `leakage_checks.csv`: matches.

The chronological split, CV fold topology, duplicate diagnostics, and leakage checks are consistent with the CPU-sec7 run. That means the data partitioning and leakage review did not drift when the notebook was rerun. The main environment table also has one expected extra row for `cupy`, which was introduced by the GPU XGBoost path.

## Main Metric Comparison

| Feature Set | Main Full AP | CPU-sec7 Full AP | AP Delta | Main Workflow Seconds | CPU-sec7 Workflow Seconds | Workflow Delta |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 0.7850 | 0.7898 | -0.0048 | 21.0 | 30.6 | -9.6 |
| Raw + TabPFN embeddings | 0.7811 | 0.7977 | -0.0166 | 428.3 | 486.0 | -57.7 |
| Raw + TabICL embeddings | 0.8128 | 0.8152 | -0.0024 | 236.6 | 379.9 | -143.3 |
| Raw + TabPFN + TabICL embeddings | 0.8029 | 0.8162 | -0.0133 | 640.4 | 837.1 | -196.8 |
| Raw all-history incumbent | 0.8097 | 0.7920 | +0.0177 | 160.4 | 270.9 | -110.5 |

The main run is directionally consistent with the CPU-sec7 run in the broad sense that TabICL embeddings remain useful and TabPFN alone is not attractive. However, the exact ranking changed:

- CPU-sec7 best full-holdout AP: `Raw + TabPFN + TabICL embeddings` at 0.8162.
- Main-run best full-holdout AP: `Raw + TabICL embeddings` at 0.8128.
- Main `Raw all-history incumbent` improved substantially versus CPU-sec7: 0.8097 vs 0.7920 AP.
- Main `Raw + TabPFN + TabICL embeddings` dropped versus CPU-sec7: 0.8029 vs 0.8162 AP.

This is the most important consistency finding: the story should not claim that combining TabPFN and TabICL is reliably best. In the fully rerun main notebook, TabICL alone is the better practical embedding enhancement.

## Operational Comparison

| Model Fragment | Target Recall | Main Alerts | CPU-sec7 Alerts | Main Precision | CPU-sec7 Precision |
|---|---:|---:|---:|---:|---:|
| Raw + TabICL embeddings | 0.8 | 91 | 91 | 0.6593 | 0.6593 |
| Raw + TabICL embeddings | 0.9 | 1087 | 903 | 0.0626 | 0.0753 |
| Raw + TabPFN + TabICL embeddings | 0.8 | 116 | 87 | 0.5172 | 0.6897 |
| Raw + TabPFN + TabICL embeddings | 0.9 | 2147 | 1079 | 0.0317 | 0.063 |
| Raw all-history incumbent | 0.8 | 94 | 95 | 0.6383 | 0.6316 |
| Raw all-history incumbent | 0.9 | 1064 | 2310 | 0.0639 | 0.0294 |
| Raw | 0.8 | 109 | 116 | 0.5505 | 0.5172 |
| Raw | 0.9 | 2284 | 2514 | 0.0298 | 0.027 |

The operational view reinforces the ranking change. On the main run, `Raw + TabICL embeddings` is the best 80%-recall operating-point choice. At 90% recall, the `Raw all-history incumbent` is narrowly better on alert count than `Raw + TabICL`, but both are much stronger than `Raw` and `Raw + both`. `Raw + TabPFN + TabICL embeddings` no longer gives the strongest operating-point result and remains much more expensive.

## Figure Comparison

Six main-run figures were extracted and compared with CPU-sec7 figures by filename and checksum. The checksums differ, as expected, because model scores and runtime values changed. The figure-level interpretation changed in the same way as the tables:

- Precision-recall curves should now be read as `Raw + TabICL` leading the full-holdout AP ranking, with `Raw all-history incumbent` also much stronger than before.
- Runtime-vs-AP plots should emphasize `Raw + TabICL` as the dominant practical point. `Raw + both` is slower and no longer has the best AP in the main run.
- Calibration plots remain secondary diagnostics. They do not provide enough evidence that calibration improves the workflow.

Visual inspection matches the numeric comparison. The full-holdout PR legend shows `Raw + TabICL` at AP 0.813, `Raw all-history incumbent` at AP 0.810, and `Raw + both` at AP 0.803. The runtime plot makes the tradeoff especially clear: `Raw + TabICL` is above `Raw + both` on AP while taking about one third of the workflow time. The calibration curves remain clustered near the origin, so they are still too compressed to carry the calibration argument alone.

## Timing Comparison

The GPU/CuPy path reduced XGBoost workflow time materially:

- `Raw`: 21.0s main vs 30.6s CPU-sec7 (-9.6s).
- `Raw + TabICL embeddings`: 236.6s main vs 379.9s CPU-sec7 (-143.3s).
- `Raw + TabPFN + TabICL embeddings`: 640.4s main vs 837.1s CPU-sec7 (-196.8s).
- `Raw all-history incumbent`: 160.4s main vs 270.9s CPU-sec7 (-110.5s).

Embedding extraction times changed only modestly; the main speedup is in section 7 XGBoost fit/predict time. This is consistent with the purpose of the GPU wrapper.

## Issues To Address

1. The main output is not numerically identical to CPU-sec7. That is acceptable after moving XGBoost onto GPU, but the notebook text should avoid overclaiming the CPU-sec7 ranking.
2. The most robust conclusion is now: TabICL embeddings can improve XGBoost, while TabPFN adds cost and does not reliably improve this fraud workflow.
3. The fair raw-vs-embedding tuning still has only one valid chronological fold, so the tuning robustness concern remains unresolved.
4. Displayed alert-budget output is still truncated; the notebook should save full result tables to CSV during execution.
5. Calibration still does not clearly improve probability quality and should be reported as a diagnostic, not as a guaranteed production improvement.

## Saved Artifacts

- Main extracted tables: `*.csv` files matching the notebook display outputs.
- Main extracted figures: `figure_*.png`.
- Table comparison: `table_consistency_summary.csv`.
- Semantic consistency summary: `semantic_consistency_summary.csv`.
- Figure comparison: `figure_consistency_summary.csv`.
- Runtime/AP deltas: `runtime_ap_delta_vs_cpu_sec7.csv`.
- Target-recall deltas: `target_recall_delta_vs_cpu_sec7.csv`.
- Run integrity comparison: `run_integrity_comparison.csv`.
