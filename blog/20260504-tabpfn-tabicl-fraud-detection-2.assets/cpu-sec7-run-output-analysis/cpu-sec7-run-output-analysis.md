# CPU Section 7 Run Output Analysis

Source notebook:
`tabpfn-tabicl-fraud-detection-20260504-with-sklearn-running-on-CPU-sec7-cell1.ipynb`

Extracted tables were saved as CSV files in this folder. The completed notebook had no execution errors, and all expected sections executed through the leakage checklist.
Embedded notebook figures were also extracted into this folder and reviewed directly.

## Run Integrity

- Environment: Python 3.12.12, pandas 2.3.3, cuDF 26.2.1, scikit-learn 1.6.1, XGBoost 3.2.0, torch 2.10.0+cu128, TabPFN 7.1.1, TabICL 2.1.1.
- GPU runtime was active with 2 CUDA devices.
- TabPFN and TabICL embedding extraction both succeeded.
- Feature bundle row counts were consistent after the earlier `X_classical_tune` row-concatenation fix:
  - Classical tune rows: 32,683 for every feature set.
  - Classical final-train rows: 170,884 for every feature set.
- Feature counts were internally consistent:
  - Raw: 29.
  - Raw + TabPFN: 221 = 29 + 192.
  - Raw + TabICL: 541 = 29 + 512.
  - Raw + both: 733 = 29 + 192 + 512.
- The section 7 run emitted the expected XGBoost device-mismatch warning because this artifact used the sklearn CPU path. The active notebook has since been patched with a CuPy-backed XGBoost wrapper.
- Six figures were extracted from the completed notebook:
  - `figure_precision_recall_sampled.png`
  - `figure_precision_recall_full.png`
  - `figure_runtime_vs_ap_sampled.png`
  - `figure_runtime_vs_ap_full.png`
  - `figure_calibration_sampled.png`
  - `figure_calibration_full.png`

## Main Result

On the full holdout, embedding-enhanced XGBoost improved Average Precision over raw XGBoost:

| Feature set | Calibration | Full-holdout AP | Workflow seconds |
|---|---:|---:|---:|
| Raw | none | 0.7898 | 30.6 |
| Raw + TabPFN | none | 0.7977 | 486.0 |
| Raw + TabICL | none | 0.8152 | 379.9 |
| Raw + TabPFN + TabICL | none | 0.8162 | 837.1 |
| Raw all-history incumbent | none | 0.7920 | 270.9 |

The result supports the notebook's core question: offline TFM embeddings can improve a production-style XGBoost fraud workflow. In this run, TabICL is the useful embedding source. TabPFN alone gives a small AP lift over raw, but its runtime cost is high. Combining TabPFN and TabICL gives only a marginal AP gain over TabICL alone.

## Operational View

The target-recall table is more informative than AP alone.

On the full holdout at 80% recall:

| Model | Alerts needed | Precision |
|---|---:|---:|
| Raw + TabPFN + TabICL | 87 | 0.6897 |
| Raw + TabICL | 91 | 0.6593 |
| Raw all-history incumbent | 95 | 0.6316 |
| Raw calibrated | 95 | 0.6316 |
| Raw | 116 | 0.5172 |

On the full holdout at 90% recall:

| Model | Alerts needed | Precision |
|---|---:|---:|
| Raw + TabICL | 903 | 0.0753 |
| Raw + TabPFN + TabICL | 1,079 | 0.0630 |
| Raw + TabPFN | 1,607 | 0.0423 |
| Raw all-history incumbent | 2,310 | 0.0294 |
| Raw | 2,514 | 0.0270 |

This changes the practical recommendation slightly: Raw + both wins AP and 80% recall alert count, but Raw + TabICL is better at the 90% recall operating point and has much lower workflow time.

## Figure Analysis

### Precision-recall curves

Figures:

- `figure_precision_recall_sampled.png`
- `figure_precision_recall_full.png`

The precision-recall figures are consistent with the tables:

- The curves are tightly clustered, which is expected because all successful models are XGBoost variants using the same raw feature base.
- On the full holdout, the legend AP values match the table ranking:
  - Raw + TabPFN + TabICL: AP 0.816.
  - Raw + TabICL: AP 0.815.
  - Raw calibrated: AP 0.799.
  - Raw: AP 0.790.
  - Raw all-history incumbent: AP 0.792.
- The visual separation is most visible in the high-recall drop-off region. The embedding-enhanced curves tend to stay above raw XGBoost around the operating range where recall is roughly 0.80-0.90.
- The sampled and full PR plots tell the same story, which is reassuring. The sampled holdout has higher base fraud rate and slightly higher AP values, but the ranking is broadly stable.

Issues:

- The curves overlap heavily, so the figure is not enough by itself to justify the result. The table and alert-budget metrics are necessary.
- The legend is large and covers part of the plot. It does not appear to hide the main high-recall drop-off, but it makes the figure look dense.
- Showing all top 10 variants is still visually busy. For publication, the plot would be clearer with 4-5 lines: Raw, Raw calibrated, Raw + TabICL, Raw + both, and Raw all-history incumbent.

Interpretation:

The PR figures support the claim that embeddings improve ranking, but they also show that the gain is incremental rather than transformational. This is the right industry-facing framing: TabICL embeddings can improve an already strong XGBoost workflow, not replace the need for careful classical modeling.

### Runtime versus Average Precision

Figures:

- `figure_runtime_vs_ap_sampled.png`
- `figure_runtime_vs_ap_full.png`

The runtime figures are consistent with the runtime table:

- Raw XGBoost is the fastest useful model, around 30-31 seconds one-path workflow time.
- Raw + TabICL gives most of the embedding benefit at much lower time than Raw + TabPFN + TabICL:
  - Full holdout Raw + TabICL: AP 0.815, one-path 379.9 seconds.
  - Full holdout Raw + both: AP 0.816, one-path 837.1 seconds.
- Raw + both is the best AP point, but it is not the best practical tradeoff.
- Raw + TabPFN alone is dominated in this run: slower than Raw + TabICL and lower AP than Raw + TabICL.
- Calibrated variants generally move downward in AP while adding time, which matches the table.

Issues:

- The x-axis is log-scaled, which is appropriate, but the chart needs a short caption explaining that `Workflow Seconds` includes shared embedding-preparation time repeated per one-path comparison.
- The legend is outside the plot and readable, but large. It is useful for analysis, less ideal for a polished blog figure.
- The figure could do more to highlight the Pareto frontier. The visual message is strongest if the notebook marks Raw, Raw + TabICL, and Raw + both as the relevant frontier points.

Interpretation:

The runtime figures strengthen the practical conclusion: TabICL is the best cost-benefit embedding source in this run. Raw + both has the best AP, but the extra TabPFN cost buys only about 0.001 AP over TabICL alone on the full holdout.

### Calibration curves

Figures:

- `figure_calibration_sampled.png`
- `figure_calibration_full.png`

The calibration figures are less useful than the PR and runtime figures:

- Only the top calibrated variants are shown, which is the right filtering choice.
- The full-holdout calibration points are tightly clustered around low predicted probabilities and low observed fraud rates.
- The sampled-holdout calibration plot shows the same low-probability clustering but with higher observed fraud rates, as expected from the fraud-enriched sampled holdout.
- The plots do not contradict the tables, but they also do not give strong evidence that calibration improved probability quality.

Issues:

- The calibration curves are visually under-informative. Most useful points sit near the origin, and the 0-0.15 zoom still leaves most of the plot empty.
- The plotted curves look like short line segments rather than full reliability curves. With rare positives and quantile binning, the diagnostic is unstable and hard to read.
- The legend dominates the top-left portion of the figure, although it does not hide much data.
- The table already showed that calibrated variants did not consistently improve log loss or AP. The calibration figures do not rescue that result.

Recommendation:

Keep calibration metrics in the table, but improve or de-emphasize the calibration plots. Better production-grade diagnostics would be:

- a compact table with Brier score, log loss, and expected calibration error;
- calibration evaluated only on the full holdout;
- a reliability table by score quantile with `rows`, `fraud_rows`, `mean_score`, and `observed_fraud_rate`;
- an "uncalibrated calibration-base" row so the calibration effect is separated from the training-window effect.

### Overall figure verdict

The figures are directionally consistent with the tables. They do not reveal a contradiction in the results. They do, however, sharpen the practical story:

1. PR curves: embeddings help, but the curves are close.
2. Runtime plot: TabICL is the best practical tradeoff; Raw + both is expensive for a tiny extra AP gain.
3. Calibration plots: calibration diagnostics need improvement before the notebook feels production-grade.

## Concerns

### 1. The fair raw-vs-embedding tuning was not actually multi-fold

`CLASSICAL_TUNING_CV_SPLITS` was 5, but the fair raw-vs-embedding CV summary contains only one valid fold:

- train rows: 24,185
- train fraud rows: 215
- validation rows: 4,249
- validation fraud rows: 9

The all-history incumbent has five valid chronological folds, but the primary raw-vs-embedding comparison does not. This weakens the robustness of the most important comparison. The likely cause is the sampled downstream training window plus sparse positives in later chronological slices.

Recommendation: tune the fair raw-vs-embedding XGBoost models on the full downstream base-training history (`train_period + validation_period`) instead of the sampled train window, while still keeping the embedding-context period excluded. This should produce more valid chronological folds and make the tuning protocol more industry-grade.

### 2. CV scores do not rank the feature sets the same way as the holdout

The fair CV AP scores were close and slightly favored raw features, but the full holdout favored TabICL and TabPFN+TabICL embeddings. With only one fair CV fold and very few validation fraud rows, the CV score should be treated as a hyperparameter-selection signal only, not evidence about feature-set quality.

Recommendation: explicitly state that the final holdout is the feature-set comparison, and strengthen the fair tuning folds as above.

### 3. Calibration did not improve probability quality in this run

Calibrated variants generally had lower AP, and log loss was not consistently better:

- Raw + TabICL full AP: 0.8152 uncalibrated vs 0.8042 calibrated.
- Raw + TabPFN + TabICL full AP: 0.8162 uncalibrated vs 0.8046 calibrated.
- Raw + TabICL full log loss: 0.0023 uncalibrated vs 0.0031 calibrated.

This is not necessarily a bug. The calibrated variant trains the base model without the calibration window, while the uncalibrated final model trains on train + validation + calibration. That means calibration comparisons also change the base-fit training set.

Recommendation: add an "uncalibrated calibration-base model" row trained on the same train + validation rows as the calibrated model. That would isolate the effect of calibration from the effect of withholding calibration labels from training.

### 4. The alert-budget output was truncated in the saved notebook

The displayed alert-budget table was truncated by pandas display settings, so only 11 rendered rows were recoverable from the completed notebook artifact. The target-recall table was fully recoverable.

Recommendation: have the notebook save key result tables to CSV directly:

- `fraud_summary.csv`
- `comparison_view.csv`
- `runtime_ap_summary.csv`
- `alert_summary.csv`
- `target_recall_summary.csv`
- `calibration_quality_view.csv`
- `leakage_checks.csv`

This is a production-grade notebook hygiene issue. Display output is not a reliable result artifact.

### 5. TFM context and training windows are intentionally enriched for fraud

The sampled TFM context has a 3.78% fraud rate and the sampled downstream train window has a 4.83% fraud rate, while the full holdout fraud rate is 0.13%. This is acceptable as a case-control training design, but it should be stated explicitly. Otherwise an industry reader may interpret it as an accidental base-rate distortion.

Recommendation: describe the sampled windows as fraud-enriched context/training windows and keep the validation, calibration, and full holdout as full-prevalence windows.

### 6. Duplicate-feature diagnostics need interpretation, not just reporting

The leakage checklist correctly reports:

- exact duplicate rows: 1,854
- exact duplicate groups crossing time windows: 0
- model-feature duplicate rows: 14,293
- model-feature duplicate groups crossing time windows: 2,178
- fraud rows inside cross-window model-feature duplicate groups: 0

This is a review finding, not a fatal leakage finding. The cross-window duplicates appear to be non-fraud rows, so they are less likely to inflate fraud-recall results, but they still deserve discussion.

## Practical Conclusion

The completed run is directionally sound and supports the notebook thesis. The strongest practical result is not "use all embeddings"; it is:

> TabICL embeddings improve XGBoost fraud ranking and alert efficiency, but the improvement must be weighed against offline feature-generation cost.

Before using the notebook as publishable benchmark evidence, I would prioritize:

1. Strengthen fair raw-vs-embedding tuning so it uses multiple chronological folds.
2. Save full result tables to CSV inside the notebook.
3. Add the uncalibrated calibration-base row to isolate calibration effects.
4. Keep the CuPy-backed XGBoost wrapper from the active notebook to remove the CPU prediction fallback.
