[Mohit Saharan](https://linkedin.com/in/msaharan), P17, 20260505

___

# TabPFN and TabICL embeddings for fraud-detection workflows

This post continues my series on tabular foundation models. I started with the vocabulary of tabular foundation models in [P3](https://www.linkedin.com/posts/msaharan_20260415-tabular-foundation-models-1pdf-activity-7450221503234621441-QYwS?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), posterior predictive distributions in [P4](https://www.linkedin.com/posts/msaharan_20260416-understanding-tfms-ppdpdf-activity-7450580114225938432-9UYN?utm_source=share&utm_medium=web), architecture in [P5](https://www.linkedin.com/posts/msaharan_20260417-understanding-tfm-architecture-tabpfnpdf-activity-7450946343922999318-6Lw_?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), and pre-training in [P6](https://www.linkedin.com/posts/msaharan_20260420-understanding-tfms-pretraining-synthetic-datapdf-activity-7452030755720888320-INN6?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g). I then moved through the TabPFN repository in [P7](https://www.linkedin.com/posts/msaharan_20260421-understanding-tfm-tabpfn-repopdf-activity-7452397229723623425-DVO3?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), TabPFN examples and embeddings in [P8](https://www.linkedin.com/posts/msaharan_20260422-understanding-tfms-tabpfn-handson-demopdf-activity-7452807834171387904-s5Ah?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), [P9](https://www.linkedin.com/posts/msaharan_20260423-understanding-tfm-trying-tabpfn-clientpdf-activity-7453126821384073216-2bqA?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), and [P10](https://www.linkedin.com/posts/msaharan_tabpfn-tabularfoundationmodels-machinelearning-activity-7453455329779941376-ymp3?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), and later compared TabPFN, TabICL, and supervised ML models in [P14](https://open.substack.com/pub/dsaiengineering/p/p14-tabular-foundation-models-comparing?utm_campaign=post-expanded-share&utm_medium=web), used TabPFN and TabICL directly for fraud detection in [P15](https://open.substack.com/pub/dsaiengineering/p/p15-tabpfn-and-tabicl-for-fraud-detection?r=535odk&utm_campaign=post-expanded-share&utm_medium=web), and used TabPFN and TabICL embeddings in an XGBoost fraud workflow in [P16](./20260504-tabpfn-tabicl-embeddings-fraud-det-workflows.md).

This post builds on that last idea. The question is not whether TabPFN or TabICL should replace XGBoost as the fraud model. The question is more practical:

> Can offline row embeddings from TabPFN or TabICL improve a production-style XGBoost fraud-detection workflow enough to justify the extra representation step?

I am still learning these models by building examples, so I treat this as a workflow demonstration rather than a benchmark claim. The goal is to make the integration pattern, the benefits, and the caveats visible. I hope that is useful both for people building tabular foundation models and for data and AI practitioners who want to understand how these models might fit into workflows they already know.

The notebook is here:

[tabpfn-tabicl-fraud-detection-20260505.ipynb](./20260505-tabpfn-tabicl-embeddings-fraud-det-workflows.assets/tabpfn-tabicl-fraud-detection-20260505.ipynb)

The results below use the latest notebook in the assets directory. I treat that notebook as the publishable version for this post.

## Conceptual background

### What stays the same as supervised ML

For a supervised binary classification problem, we have rows

$$
(x_i, y_i), \quad i = 1, \ldots, n
$$

where $x_i \in \mathbb{R}^{d}$ is the feature vector for row $i$, $d$ is the number of raw input features, and $y_i \in \{0, 1\}$ is the label. In this dataset, $y_i = 1$ means fraud and $y_i = 0$ means not fraud.

A supervised model learns task-specific parameters from the available labelled data. I write that as

$$
\hat{p}_i = h_\theta(x_i)
$$

where $\hat{p}_i$ is the model score or estimated fraud probability, $h_\theta$ is the fitted supervised model, and $\theta$ represents parameters learned for this task.

For XGBoost, a simplified view is that the model learns an additive ensemble of trees:

$$
s_i = \sum_{m=1}^{M} f_m(x_i)
$$

where $M$ is the number of trees, $f_m$ is tree $m$, and $s_i$ is the raw score for row $i$. The raw score can be mapped to a probability-like value by

$$
\hat{p}_i = \sigma(s_i) = \frac{1}{1 + \exp(-s_i)}
$$

where $\sigma(\cdot)$ is the sigmoid function.

That familiar supervised ML structure stays intact in this notebook. XGBoost is still trained on labelled data, selected with chronological validation folds, evaluated on future holdout rows, and judged with fraud-specific metrics.

### What changes when TabPFN or TabICL produces embeddings

The new part is the representation step before XGBoost.

TabPFN and TabICL are used as tabular foundation models. In this notebook, they are not used as the final fraud scorers. Instead, each model sees an earlier labelled context:

$$
C = \{(x_j, y_j)\}_{j=1}^{m}
$$

where $C$ is the representation context, $j$ indexes context rows, and $m$ is the number of context rows.

After conditioning on that context, the tabular foundation model maps a later row $x_i$ into an embedding:

$$
z_i = g_\phi(x_i; C)
$$

where $z_i \in \mathbb{R}^{q}$ is the embedding vector, $q$ is the embedding dimension, $g_\phi$ is the embedding function induced by the pretrained model and the context, and $\phi$ represents pretrained model parameters.

The downstream XGBoost model then receives an augmented feature vector:

$$
\tilde{x}_i = [x_i, z_i]
$$

where $[x_i, z_i]$ means concatenation of the raw features and the embedding features. XGBoost is then trained as

$$
\hat{p}_i = h_\theta(\tilde{x}_i)
$$

This is the main conceptual distinction:

- Supervised ML capability: XGBoost learns a task-specific scoring function from the fraud dataset.
- Tabular foundation model capability: TabPFN or TabICL contributes a learned representation of each row after seeing an earlier labelled task context.
- Workflow question: Does the representation $z_i$ add enough useful information to the raw features $x_i$ to justify the cost and complexity of generating it?

For a practitioner familiar with supervised ML but new to foundation models, the simplest way to read the notebook is this: the production-style scorer remains familiar, but the feature set is augmented by a context-conditioned representation generated by a pretrained tabular model.

### Why the time split matters

The public credit-card fraud dataset has a `Time` column. The notebook sorts rows by `Time` and uses chronological windows:

- Earliest 20% of transactions: representation context for TabPFN and TabICL.
- Next 40%: downstream XGBoost training period.
- Next 10%: validation period for model selection.
- Next 10%: calibration period for post-hoc probability calibration.
- Final 20%: full holdout.

This split matters because TabPFN and TabICL use labelled context rows. If a row's own label were used to generate that row's embedding, the evaluation would be contaminated. The notebook avoids that by giving TabPFN and TabICL only earlier context rows when creating embeddings for later training, validation, calibration, and holdout rows.

This is still a simplified public-data workflow. A production fraud project would also need customer, account, card, merchant, device, and label-availability fields. This dataset does not expose those fields, so some production leakage checks remain unresolved.

### Why Average Precision is not enough

The dataset has 284,807 transactions and 492 fraud cases, so the fraud rate is about 0.1727%. Accuracy is not a useful headline metric because a model that predicts "not fraud" for every transaction would be more than 99% accurate while catching no fraud.

The notebook therefore focuses on ranking and alert-queue metrics.

Precision is

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Recall is

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

where $TP$ means true positives, $FP$ means false positives, and $FN$ means false negatives.

Average Precision summarizes the precision-recall curve:

$$
AP = \sum_{k=1}^{K} (R_k - R_{k-1})P_k
$$

where $K$ is the number of threshold steps, $R_k$ is recall at step $k$, and $P_k$ is precision at step $k$.

Average Precision is useful, but fraud teams also work with review capacity. That leads to operational questions:

- If the team reviews the top 100 alerts, how many frauds are found?
- If the team reviews the top 1% of transactions, what recall is reached?
- How many alerts are needed to reach 80% recall?
- How many alerts are needed to reach 90% recall?

That is why the notebook reports both AP and alert-count metrics.

### Calibration is a separate question

A fraud score can be useful for ranking without being a well-calibrated probability. Calibration asks whether predicted probabilities match observed frequencies. If a model assigns a group of transactions a predicted fraud probability near 1%, then approximately 1% of those transactions should be fraud if the probabilities are calibrated.

The notebook reports the Brier score:

$$
\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(\hat{p}_i - y_i)^2
$$

It also reports log loss:

$$
\text{LogLoss} =
-\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)
\right]
$$

Here, $N$ is the number of evaluated rows. Lower is better for both Brier score and log loss.

I treat calibration as diagnostic in this post. The full holdout has only 75 fraud cases, which is enough to inspect behavior but not enough to claim production-grade probability calibration.

## Hands-on demo

### Experimental design

The notebook compares four publication rows:

- Raw XGBoost: XGBoost trained on the 29 raw model features.
- Raw all-history XGBoost: an incumbent-style XGBoost row that uses all pre-holdout raw historical data.
- Raw + TabPFN embeddings: raw features plus 192 TabPFN embedding features, giving 221 total features.
- Raw + TabICL embeddings: raw features plus 512 TabICL embedding features, giving 541 total features.

The combined `Raw + TabPFN + TabICL` feature set is intentionally not part of the publication comparison. A team may test TabPFN embeddings or TabICL embeddings, but operating both embedding systems together is a different adoption choice with additional cost and monitoring burden.

The fair raw-vs-embedding tuning path uses 142,403 downstream tuning rows after excluding 56,961 representation-context rows. Those tuning rows contain 227 fraud cases. The raw all-history incumbent uses 199,364 pre-holdout rows and 384 fraud cases because it does not need to exclude the representation-context rows.

The fair raw-vs-embedding comparison has five chronological validation folds. The validation fraud counts across those folds are 25, 19, 78, 8, and 20. This is not a large number of positive cases, but it is much better than judging the workflow from a single valid fold.

### Full-holdout ranking result

The full holdout is the deployment-facing view because it keeps the final-window fraud base rate. The main ranking result is:

- Raw all-history XGBoost: AP 0.8097, workflow time 173.3 seconds, Top 0.5% Recall 0.8533, Top 1% Recall 0.8533, Brier 0.000424, ECE 10 0.000104.
- Raw XGBoost: AP 0.8034, workflow time 147.0 seconds, Top 0.5% Recall 0.8400, Top 1% Recall 0.8667, Brier 0.000403, ECE 10 0.000231.
- Raw + TabICL: AP 0.7970, workflow time 1147.8 seconds, Top 0.5% Recall 0.8267, Top 1% Recall 0.9067, Brier 0.000451, ECE 10 0.000485.
- Raw + TabPFN: AP 0.7909, workflow time 840.2 seconds, Top 0.5% Recall 0.8000, Top 1% Recall 0.8400, Brier 0.000381, ECE 10 0.000184.

The raw all-history XGBoost row has the best point AP. Raw XGBoost is close and much faster. Neither single-source embedding workflow improves point AP over the raw XGBoost rows in this run.

That is an important outcome. The value of the notebook is not that the embedding rows automatically win. The value is that the workflow gives a practical way to test whether a foundation-model representation helps under a realistic baseline.

![Full-holdout precision-recall curves](./20260505-tabpfn-tabicl-embeddings-fraud-det-workflows.assets/publication_precision_recall_curves_full.png)

The precision-recall curves are close. The dashed line is the fraud base rate. When curves are this close, I would not rely on the figure alone. The operating-point view is more useful for interpreting the practical difference.

### Alert-budget interpretation

At 80% recall on the full holdout:

- Raw all-history XGBoost needs 94 alerts to find 60 frauds, with precision 0.6383.
- Raw XGBoost needs 116 alerts to find 60 frauds, with precision 0.5172.
- Raw + TabICL needs 127 alerts to find 60 frauds, with precision 0.4724.
- Raw + TabPFN needs 199 alerts to find 60 frauds, with precision 0.3015.

At 90% recall on the full holdout:

- Raw + TabICL needs 490 alerts to find 68 frauds, with precision 0.1388.
- Raw all-history XGBoost needs 1,064 alerts to find 68 frauds, with precision 0.0639.
- Raw XGBoost needs 1,177 alerts to find 68 frauds, with precision 0.0578.
- Raw + TabPFN needs 1,598 alerts to find 68 frauds, with precision 0.0426.

This is the main operational nuance. AP favors the raw all-history incumbent, but the 90% recall operating point favors TabICL. If a fraud team is targeting very high recall, this is the part of the result I would investigate further.

For fixed alert budgets on the full holdout:

- At 100 alerts, Raw all-history XGBoost finds 60 frauds and reaches 80.0% recall.
- At 500 alerts, Raw + TabICL finds 68 frauds and reaches 90.7% recall.
- At 1000 alerts, Raw + TabICL still finds 68 frauds and remains at 90.7% recall.
- At the top 0.5% of transactions, Raw all-history XGBoost finds 64 frauds and reaches 85.3% recall.
- At the top 1% of transactions, Raw + TabICL finds 68 frauds and reaches 90.7% recall.

This is why I do not want to summarize the notebook with one metric. AP, fixed alert budgets, and target-recall alert counts each answer a different question.

### Uncertainty around the result

The notebook uses bootstrap resampling on the full holdout to estimate uncertainty. This is useful because the full holdout has only 75 fraud cases.

For full-holdout AP:

- Raw XGBoost has point AP 0.8034, bootstrap median 0.8032, and 95% interval from 0.6972 to 0.8709.
- Raw all-history XGBoost has point AP 0.8097, bootstrap median 0.8140, and 95% interval from 0.7189 to 0.8947.
- Raw + TabPFN has point AP 0.7909, bootstrap median 0.7905, and 95% interval from 0.7061 to 0.8687.
- Raw + TabICL has point AP 0.7970, bootstrap median 0.7973, and 95% interval from 0.7063 to 0.8723.

These intervals overlap heavily. My reading is that raw all-history XGBoost has the best point estimate, but the AP differences should not be overstated.

For alerts needed at 90% recall:

- Raw XGBoost has point estimate 1,177, bootstrap median 1,230, and 95% interval from 212 to 9,822 alerts.
- Raw all-history XGBoost has point estimate 1,064, bootstrap median 1,046, and 95% interval from 129 to 7,311 alerts.
- Raw + TabPFN has point estimate 1,598, bootstrap median 1,598, and 95% interval from 451 to 7,972 alerts.
- Raw + TabICL has point estimate 490, bootstrap median 495, and 95% interval from 254 to 3,449 alerts.

The TabICL point estimate remains interesting, but the intervals are wide. This should be read as a workflow signal, not as a production decision.

### Runtime and memory

The runtime plot shows the engineering tradeoff:

![Full-holdout runtime versus Average Precision](./20260505-tabpfn-tabicl-embeddings-fraud-det-workflows.assets/publication_runtime_vs_average_precision_full.png)

The workflow times are:

- Raw XGBoost: 147.0 seconds.
- Raw all-history XGBoost: 173.3 seconds.
- Raw + TabPFN: 840.2 seconds, including 371.4 seconds of shared TabPFN embedding preparation.
- Raw + TabICL: 1147.8 seconds, including 120.5 seconds of shared TabICL embedding preparation.

One subtle point is that TabICL embedding preparation is faster than TabPFN embedding preparation in this run, but the full Raw + TabICL workflow is slower. The likely reason is the downstream XGBoost search over a wider feature matrix: TabICL contributes 512 embedding columns, while TabPFN contributes 192.

The embedding matrix sizes also matter:

- TabPFN final training embeddings: 170,884 rows, 192 columns, about 125.2 MB.
- TabPFN full-holdout embeddings: 56,962 rows, 192 columns, about 41.7 MB.
- TabICL final training embeddings: 170,884 rows, 512 columns, about 333.8 MB.
- TabICL full-holdout embeddings: 56,962 rows, 512 columns, about 111.3 MB.

Both embedding paths fit on the Kaggle two-T4 GPU runtime used for the notebook. TabPFN used both CUDA devices and reached about 936.8 MB maximum allocated memory per device during extraction. TabICL used device 0 more heavily, reaching about 3657.3 MB maximum allocated memory and 4548.0 MB reserved memory after extraction.

For a practitioner, the lesson is that representation quality should be judged together with representation cost. The question is not only "does the embedding help?" It is also "does the embedding help enough to justify extraction, storage, downstream training time, and monitoring?"

### Calibration diagnostics

Calibration remains diagnostic rather than a clear improvement.

On the full holdout:

- Raw all-history uncalibrated: AP 0.8097, Brier 0.000424, log loss 0.002824, ECE 10 0.000104.
- Raw calibration base: AP 0.8047, Brier 0.000396, log loss 0.002835, ECE 10 0.000209.
- Raw calibrated: AP 0.8047, Brier 0.000422, log loss 0.003287, ECE 10 0.000356.
- Raw + TabPFN uncalibrated: AP 0.7909, Brier 0.000381, log loss 0.002500, ECE 10 0.000184.
- Raw + TabPFN calibration base: AP 0.7918, Brier 0.000407, log loss 0.002669, ECE 10 0.000163.
- Raw + TabPFN calibrated: AP 0.7918, Brier 0.000441, log loss 0.003349, ECE 10 0.000528.
- Raw + TabICL uncalibrated: AP 0.7970, Brier 0.000451, log loss 0.002583, ECE 10 0.000485.
- Raw + TabICL calibration base: AP 0.7978, Brier 0.000445, log loss 0.002719, ECE 10 0.000359.
- Raw + TabICL calibrated: AP 0.7978, Brier 0.000430, log loss 0.003091, ECE 10 0.000298.

Sigmoid calibration leaves AP unchanged for the calibration-base rows because it is a monotonic score transformation. It worsens log loss for all selected rows. It improves TabICL Brier and ECE relative to the TabICL calibration base, but not enough to make calibration a clear overall improvement. For TabPFN, sigmoid calibration worsens Brier, log loss, and ECE relative to its calibration-base row.

The calibration plot is useful as a visual diagnostic, but the reliability-bin CSVs are more important for interpretation because most meaningful probabilities are near zero.

![Full-holdout calibration curves](./20260505-tabpfn-tabicl-embeddings-fraud-det-workflows.assets/publication_calibration_curves_full.png)

### Leakage and public-data limits

The notebook includes a leakage and reuse checklist. The public dataset allows some checks but not all of the checks a production fraud project would need.

Checks that pass or are explicit design choices:

- `Class` is excluded from the model features.
- `Time` is preserved for chronological splitting.
- `Time` is not used as a model feature in the default run.
- The chronological split order passes, although equal boundary timestamps can occur.

Checks that remain unresolved because of public-data limitations:

- Future aggregate feature lineage is unknown because `V1` to `V28` are anonymized PCA-style features.
- Entity-level leakage is unknown because there is no customer, card, account, merchant, or device ID.
- Label delay is unknown because the public dataset does not expose when a fraud label became available.

Duplicate diagnostics:

- Exact duplicate rows including time and target: 1,854.
- Exact duplicate groups crossing windows: 0.
- Model-feature duplicate rows: 14,293.
- Model-feature duplicate groups crossing windows: 2,178.
- Fraud rows inside cross-window model-feature duplicate groups: 0.

I read this as a review finding, not a fatal leakage failure. In a company dataset, I would want entity IDs, feature creation timestamps, and label availability timestamps before treating the leakage review as complete.

## Known shortcomings

This is one public dataset, so the result should not be generalized to all fraud datasets, transaction workflows, or tabular foundation models.

The dataset is anonymized. The public features are useful for a controlled workflow demonstration, but they limit business interpretation, entity-level validation, delayed-label analysis, and feature-lineage review.

The full holdout has only 75 fraud cases. The bootstrap intervals help, but they also show why point estimates should be interpreted cautiously.

The TabICL embedding path uses model internals rather than a stable public embedding method comparable to TabPFN's `get_embeddings`. That does not make the experiment invalid, but it means the code path should be version-pinned and reviewed.

I have not yet added interpretability methods such as SHAP, missing-data stress tests, categorical stress tests, drift-by-period analysis, or group-aware splitting. Those are important next steps for a broader testbench.

## Summary and Conclusion

This notebook tests a practical integration pattern:

1. Keep XGBoost as the downstream fraud scorer.
2. Use TabPFN or TabICL as an offline row-embedding generator.
3. Append those embeddings to raw transaction features.
4. Evaluate the result with chronological splits, AP, alert counts, runtime, memory, calibration diagnostics, and leakage checks.

The result is not a simple "TFM embeddings win" result. Raw all-history XGBoost has the best full-holdout point AP, and raw XGBoost is close while being much faster. Single-source TabPFN and TabICL embeddings do not beat the raw baselines by AP in this run. The bootstrap intervals overlap heavily, so I would not overstate the AP ranking.

The useful nuance is the high-recall operating point. At 90% recall, Raw + TabICL needs 490 alerts to recover 68 of 75 fraud cases, compared with 1,064 alerts for raw all-history XGBoost and 1,177 alerts for raw XGBoost. That makes TabICL worth investigating for high-recall review-queue settings, even though it is not the best AP/runtime row overall.

My cautious interpretation is that tabular foundation model embeddings should be evaluated as workflow components, not only as standalone model scores. The important question is not only whether an embedding row wins AP. It is whether the embedding improves a business-relevant operating point enough to justify the added representation path.

For practitioners, the reusable lesson is the evaluation design. A TFM embedding experiment should be compared against a serious classical baseline, with time-aware splitting, alert-budget metrics, calibration diagnostics, runtime accounting, memory accounting, and leakage checks.

For labs and researchers, I hope this kind of notebook is useful as a field-facing test. It does not replace formal benchmarks, but it can show how model capabilities appear when inserted into workflows that data teams already understand.

## Outlook

The immediate notebook cleanup is to make output artifacts impossible to confuse across future executions. After that, I want to extend the workflow in directions that matter for real data science teams:

- interpretability;
- missing-data behavior;
- categorical features;
- time-derived feature policy;
- drift by time period;
- group-aware splitting when entity IDs are available;
- additional datasets beyond this public credit-card fraud dataset.

My current goal is not to prove that one model family is always better. It is to build reusable examples that make benefits, costs, and caveats visible enough for both model builders and practitioners to reason about them.
