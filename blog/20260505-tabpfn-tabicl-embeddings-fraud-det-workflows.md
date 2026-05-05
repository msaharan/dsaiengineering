[Mohit Saharan](https://linkedin.com/in/msaharan), P17, 20260505

___

# Making the TabPFN and TabICL fraud-embedding notebook professional-ready

In [P16](./20260504-tabpfn-tabicl-embeddings-fraud-det-workflows.md), I tested a practical integration pattern for tabular foundation models:

1. keep XGBoost as the downstream fraud scorer;
2. use TabPFN and TabICL only as offline embedding generators;
3. append those embeddings to raw transaction features;
4. evaluate the workflow under chronological splitting, alert-count metrics, runtime, calibration diagnostics, and leakage checks.

The result was useful, but the output review in the assets folder showed that the notebook was not yet strong enough as a professional workflow artifact. The headline signal was that TabICL embeddings looked useful, the raw all-history XGBoost incumbent was very competitive, TabPFN added cost without clear benefit in that run, and the figures and calibration evidence needed cleanup.

Today's work turns that notebook from an exploratory experiment into a cleaner testbench notebook.

Notebook:

[tabpfn-tabicl-fraud-detection-20260505.ipynb](./20260505-tabpfn-tabicl-embeddings-fraud-det-workflows.assets/tabpfn-tabicl-fraud-detection-20260505.ipynb)

## What was implemented

The notebook is now self-standing. It no longer explains itself as "the next iteration" or as a change from yesterday's run. It opens with the professional question directly:

> Can offline row embeddings from TabPFN and TabICL improve a production-style XGBoost fraud-detection workflow?

The default model set is now compact and matches a more realistic adoption assumption: a team would usually choose one TFM embedding source to operate, monitor, and maintain, not concatenate multiple foundation-model embedding systems.

- raw XGBoost;
- raw all-history XGBoost incumbent;
- raw + TabPFN embeddings;
- raw + TabICL embeddings.

The combined Raw + TabPFN + TabICL feature set has been removed from the default workflow. Logistic Regression is still available as an optional CPU benchmark, but it is not a default publication row. This keeps the main workflow focused on the feature sets that matter most for the practical question.

The fair raw-vs-embedding tuning path now uses the full train-plus-validation history after the TFM context window by default. That directly addresses the biggest issue from the output review: the previous fair comparison collapsed to one valid chronological fold after fraud-count checks. The new default gives the tuner more chronological data while still excluding rows whose labels condition the TFM context.

The notebook now separates full audit output from publication output. Full CSVs are still saved for review, but the figures use a curated publication view instead of every successful row. This should make the notebook easier to present to data-science and ML-engineering readers.

The notebook writes a run record to:

```text
tabpfn_tabicl_fraud_detection_20260505_outputs/
```

Saved outputs include environment, data quality, split summary, tuning policy, feature bundles, CV folds, full model summary, publication summary, runtime/AP summary, alert budgets, target-recall tables, calibration quality, reliability bins, duplicate diagnostics, and leakage checks.

Calibration is now cleaner. For selected calibration candidates, the notebook evaluates:

- the normal final uncalibrated model;
- a matching uncalibrated calibration-base model;
- the sigmoid-calibrated model.

That means calibration can be judged against a matching base fit instead of being mixed with a different training-window contract.

## Figure cleanup

The precision-recall plots now use only the curated publication models instead of the top 10-12 variants by AP. That should make the overlapping curves easier to read.

The runtime-vs-AP plot now uses the curated publication models and highlights the Pareto frontier. This is a better visual for the real engineering question: which rows improve quality enough to justify their workflow time?

The calibration plot is now limited to the full holdout and selected calibration candidates. The notebook also saves reliability-bin tables, which are more useful than the calibration curve alone for this rare-event dataset.

## What remains

The notebook still needs a fresh Kaggle GPU execution. The source is clean and ready, but the improved tuning path and curated plots have not yet produced new completed-run artifacts.

After running it, the next checks are:

1. Did the fair raw-vs-embedding comparison produce more valid chronological folds?
2. Does either single-embedding workflow beat the fair raw baseline on full-holdout AP and alert efficiency?
3. Does the raw all-history incumbent remain competitive?
4. Which single embedding source, TabPFN or TabICL, gives the better quality/runtime tradeoff?
5. Do the calibration rows improve log loss, Brier score, or reliability-bin behavior relative to their matching calibration-base rows?
6. Are the publication figures clean enough to include directly in the post?

## Why this matters

A useful tabular foundation model example should not stop at "the model runs." It should show where the model fits into a workflow that a data team already understands.

For fraud detection, that means comparing against a strong XGBoost incumbent, preserving time order, reporting alert counts, measuring runtime, checking calibration, and being explicit about leakage limits.

That is the direction of this notebook. It is not trying to prove that tabular foundation models replace classical ML. It is testing whether TabPFN and TabICL embeddings can improve an existing classical workflow enough to justify the extra representation step.

That is a more useful standard for professional adoption.
