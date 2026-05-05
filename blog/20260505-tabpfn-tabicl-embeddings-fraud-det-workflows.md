[Mohit Saharan](https://linkedin.com/in/msaharan), P17, 20260505

___

# Hardening the TabPFN and TabICL fraud-embedding workflow

This post continues from [P16](./20260504-tabpfn-tabicl-embeddings-fraud-det-workflows.md), where I tested a practical fraud-detection workflow: keep XGBoost as the downstream fraud scorer, use TabPFN and TabICL only as offline embedding generators, and ask whether the resulting representation features improve the classical workflow.

The useful result from that run was cautious rather than dramatic. TabICL embeddings improved the full-holdout Average Precision over the fair raw XGBoost baseline, but the raw all-history XGBoost incumbent was also very strong. TabPFN embeddings alone did not help in that completed run, and combining TabPFN with TabICL was slower without improving the practical tradeoff.

That is exactly the kind of result I want these notebooks to surface. The goal is not to make a leaderboard claim. The goal is to build a reusable workflow that can answer an industrial question:

> If a team already has a strong tabular ML workflow, where can a tabular foundation model actually help, and what is the engineering cost?

Today I am not adding another model just to expand the table. I am improving the notebook so the next completed run is a stronger piece of evidence.

You can find the May 5 notebook here:

[tabpfn-tabicl-fraud-detection-20260505.ipynb](./20260505-tabpfn-tabicl-embeddings-fraud-det-workflows.assets/tabpfn-tabicl-fraud-detection-20260505.ipynb)

## Why the previous run needed another iteration

The May 4 output review showed five important issues.

First, the fair raw-vs-embedding tuning path was weaker than intended. The notebook requested five chronological tuning folds, but after enforcing minimum fraud counts, the fair comparison had only one valid fold. That means the final holdout was still useful, but the tuning protocol was not as robust as it should be.

Second, the exact ranking was not stable enough to overclaim. In one completed run, Raw + TabPFN + TabICL looked best by a tiny margin. In the rerun used for the post, Raw + TabICL was the better practical point. The stable conclusion was not "use all embeddings." It was: TabICL embeddings are worth testing; TabPFN added cost in this workflow and did not reliably improve the result.

Third, calibration was not isolated cleanly. The calibrated rows were trained on a smaller base-fit window because the calibration window had to be held out. The uncalibrated final rows could train on more pre-holdout labels. That mixes two effects: calibration and training-data size.

Fourth, some notebook displays were too fragile as evidence. Pandas display truncation meant important operating-point tables were not fully recoverable from the rendered notebook output. A serious workflow should save the tables directly.

Fifth, the public dataset limits leakage review. The notebook can check target exclusion, chronological order, time-feature policy, and duplicate rows. It cannot check customer-level leakage, merchant-level leakage, feature-generation timing, or label delay because those fields are not exposed.

## What changed in the May 5 notebook

The notebook now makes four practical changes.

### 1. The fair tuning path uses full pre-holdout history

The May 4 notebook tuned the fair raw-vs-embedding comparison on:

- the fraud-enriched sampled downstream training window;
- the full validation window.

That was fast, but it produced too few valid chronological folds under severe class imbalance.

The May 5 default changes the fair tuning matrix to:

- full train period;
- full validation period;
- still excluding the earlier TFM embedding-context period.

The configuration flag is:

```python
USE_FULL_PREHOLDOUT_TUNING_FOR_FAIR_COMPARISON = True
```

This is slower, but it is a better default for a benchmark-style run. A faster exploratory run can still set it to `False`.

The raw all-history incumbent remains separate. It can use the embedding-context period as ordinary raw-feature history, because it does not use that period to condition a TFM representation model.

### 2. The notebook writes a real run record

The May 5 notebook writes key outputs into:

```text
tabpfn_tabicl_fraud_detection_20260505_outputs/
```

The saved tables include environment, splits, tuning policy, feature bundles, CV folds, model summary, runtime/AP, alert budgets, target recall, calibration quality, reliability bins, duplicate diagnostics, and leakage checks.

This matters because the notebook display should not be the only evidence. The CSV artifacts make the run easier to review, compare, and reuse.

### 3. Calibration now has a matching uncalibrated base row

The notebook now adds `Calibration Base` rows.

These rows are uncalibrated models trained on the same train-plus-validation base window that the sigmoid-calibrated rows use before calibration. That makes the calibration comparison cleaner:

- ordinary `none` rows answer the final pre-holdout ranking question;
- `none_calibration_base` rows show the uncalibrated base model before calibration;
- `sigmoid` rows show what the calibration step changed.

This should make it easier to tell whether calibration helped probability quality or merely changed the training-window contract.

### 4. Calibration diagnostics now include reliability bins

The notebook still reports Brier score and log loss, but it now also computes a quantile-bin expected calibration error and saves reliability-bin tables.

For rare-event fraud data, a calibration curve can be visually compressed near the origin. The reliability table is often more useful because it shows, bin by bin:

- how many rows are in the score bin;
- how many fraud cases are present;
- the mean predicted score;
- the observed fraud rate;
- the absolute calibration error.

That is closer to the diagnostic a production team would want before treating a fraud score as a probability.

## How I will read the next completed run

After rerunning the May 5 notebook, I will focus on these questions.

1. Did the fair raw-vs-embedding comparison get more valid chronological folds?
2. Does TabICL still improve full-holdout ranking after stronger tuning?
3. Is the raw all-history incumbent still competitive?
4. At 80% and 90% recall, which model needs the fewest alerts?
5. Does calibration improve log loss, Brier score, or reliability-bin behavior relative to the matching calibration-base row?
6. Does TabPFN justify its added feature-generation cost in this workflow?

The most important comparison is still not just AP. For fraud detection, I care about ranking, review capacity, runtime, calibration quality, and leakage review together.

## Why this direction matters

Most tabular foundation model examples are still too small to answer practical workflow questions. They show that a model can run, but not whether it fits into an existing fraud, credit-risk, churn, pricing, or transaction-monitoring stack.

This notebook is moving in the other direction. It keeps a familiar production-style scorer, adds TabPFN and TabICL embeddings as optional representation features, and records the caveats around time splits, sampling, calibration, runtime, and leakage.

That is the type of testbench I want to build: not a toy demo, and not only a benchmark score, but a reusable workflow that helps model developers and data teams understand where tabular foundation models are useful.

## Current status

The May 5 notebook is now a clean source notebook ready for rerun. I cleared the copied May 4 outputs after changing the protocol, so the next post update should be based on a fresh completed execution rather than stale outputs.

The next step is to run it on Kaggle with GPU enabled, extract the saved artifacts, and compare the new run against the May 4 baseline.
