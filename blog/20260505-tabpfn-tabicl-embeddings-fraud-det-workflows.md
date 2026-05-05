[Mohit Saharan](https://linkedin.com/in/msaharan), P17, 20260505

___

# TabPFN and TabICL embeddings for fraud-detection workflows

This post continues my series on tabular foundation models. So far, I have covered the basic vocabulary of tabular foundation models in [P3](https://www.linkedin.com/posts/msaharan_20260415-tabular-foundation-models-1pdf-activity-7450221503234621441-QYwS?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), the posterior predictive distribution in [P4](https://www.linkedin.com/posts/msaharan_20260416-understanding-tfms-ppdpdf-activity-7450580114225938432-9UYN?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), the architecture in [P5](https://www.linkedin.com/posts/msaharan_20260417-understanding-tfm-architecture-tabpfnpdf-activity-7450946343922999318-6Lw_?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), pre-training in [P6](https://www.linkedin.com/posts/msaharan_20260420-understanding-tfms-pretraining-synthetic-datapdf-activity-7452030755720888320-INN6?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), the TabPFN repository in [P7](https://www.linkedin.com/posts/msaharan_20260421-understanding-tfm-tabpfn-repopdf-activity-7452397229723623425-DVO3?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), the hands-on demo's classification and regression examples in [P8](https://www.linkedin.com/posts/msaharan_20260422-understanding-tfms-tabpfn-handson-demopdf-activity-7452807834171387904-s5Ah?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), TabPFN Client in [P9](https://www.linkedin.com/posts/msaharan_20260423-understanding-tfm-trying-tabpfn-clientpdf-activity-7453126821384073216-2bqA?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), TabPFN embeddings in [P10](https://www.linkedin.com/posts/msaharan_tabpfn-tabularfoundationmodels-machinelearning-activity-7453455329779941376-ymp3?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g), TabPFN's predictive behavior in [P11](https://open.substack.com/pub/dsaiengineering/p/p11-understanding-tabular-foundation?utm_campaign=post-expanded-share&utm_medium=web), time series forecasting with TabPFN in [P12](https://open.substack.com/pub/dsaiengineering/p/p12-understanding-tabular-foundation?utm_campaign=post-expanded-share&utm_medium=web), using TabPFN for causal inference in [P13](https://open.substack.com/pub/dsaiengineering/p/p13-understanding-tabular-foundation?utm_campaign=post-expanded-share&utm_medium=web), comparing TabPFN, TabICL, and supervised ML models in [P14](https://open.substack.com/pub/dsaiengineering/p/p14-tabular-foundation-models-comparing?utm_campaign=post-expanded-share&utm_medium=web), using TabPFN and TabICL directly for fraud detection in [P15](https://open.substack.com/pub/dsaiengineering/p/p15-tabpfn-and-tabicl-for-fraud-detection?r=535odk&utm_campaign=post-expanded-share&utm_medium=web), and using TabPFN and TabICL embeddings to enhance an existing XGBoost fraud-detection workflow in [P16](./20260504-tabpfn-tabicl-embeddings-fraud-det-workflows.md).

P16 was an important step for me because it moved the discussion from "can TabPFN or TabICL score this dataset directly?" to "can these models improve a workflow that a data team may already recognize?" The notebook used TabPFN and TabICL as offline embedding generators, appended those embeddings to raw transaction features, and then trained XGBoost as the downstream fraud scorer. That setup is closer to how a company might pilot a tabular foundation model without replacing its whole production model stack.

After reviewing the P16 run and the first May 5 run, I found that the notebook needed to become more disciplined. The earlier version included too many model variations, the figures were too crowded, the fair raw-vs-embedding tuning path needed more chronological data, and the combined `Raw + TabPFN + TabICL` feature set was not aligned with the adoption question I wanted to ask. A team may reasonably test TabPFN embeddings or TabICL embeddings, but operating both representation systems at once is a different and more expensive proposal.

Today's notebook therefore turns the idea into a cleaner workflow testbench. It asks:

> Can offline row embeddings from TabPFN or TabICL improve a production-style XGBoost fraud-detection workflow enough to justify the extra representation step?

That question is deliberately modest. I am still learning these models by building examples, so I do not treat one public-dataset run as a benchmark claim about TabPFN, TabICL, XGBoost, or fraud modeling in general. My goal is to demonstrate a reusable pattern, show the results honestly, and highlight the practical nuances that matter when researchers, labs, and data or AI practitioners think about tabular foundation models in real workflows.

Notebook:

[tabpfn-tabicl-fraud-detection-20260505.ipynb](./20260505-tabpfn-tabicl-embeddings-fraud-det-workflows.assets/tabpfn-tabicl-fraud-detection-20260505.ipynb)

The current notebook source is marked `P17, v3`. The completed numerical results discussed in this post come from the v2 run saved in the assets directory. The v3 source keeps the same core workflow, adds stronger reproducibility and uncertainty artifacts, and is ready for a fresh Kaggle GPU execution.

## Conceptual background

### The supervised ML baseline

In an ordinary supervised tabular workflow, we have rows

$$
(x_i, y_i), \quad i = 1, \ldots, n
$$

where $x_i$ is the feature vector for row $i$ and $y_i$ is the target. In this fraud example, $y_i \in \{0, 1\}$, where 1 means fraud and 0 means not fraud.

A supervised model such as Logistic Regression or XGBoost learns task-specific parameters from the training data. We can write the prediction function as

$$
\hat{p}_i = h_\theta(x_i)
$$

where $\hat{p}_i$ is the model's fraud score or estimated fraud probability, $h_\theta$ is the fitted model, and $\theta$ represents parameters learned from this dataset.

For XGBoost, the model is an additive ensemble of decision trees. A simplified binary-classification view is

$$
s_i = \sum_{m=1}^{M} f_m(x_i)
$$

where each $f_m$ is a tree and $s_i$ is the raw score. The score can then be mapped to a probability-like value through a sigmoid function:

$$
\hat{p}_i = \sigma(s_i) = \frac{1}{1 + \exp(-s_i)}
$$

The important point is that XGBoost learns from the current task data. It does not arrive with a broad learned prior over many tabular tasks in the same sense as a tabular foundation model. Its strength is different: it is fast, familiar, strong on many tabular datasets, easy to tune, and widely used in production-style systems.

That is why the notebook keeps XGBoost as the downstream fraud scorer. If tabular foundation models are useful in this workflow, they need to improve a strong existing supervised ML setup, not only beat a weak baseline.

### What changes when we add tabular foundation model embeddings

TabPFN and TabICL enter this notebook in a different role. They are not used as final fraud scorers. They are used to generate row embeddings.

An embedding is a numerical vector that represents a row in a learned feature space. In this notebook, the embedding is generated after the tabular foundation model has seen an earlier labelled context:

$$
C = \{(x_j, y_j)\}_{j=1}^{m}
$$

Here, $C$ is the representation context. It contains earlier transactions and labels. The tabular foundation model then maps a later row $x_i$ into a representation:

$$
z_i = g_\phi(x_i; C)
$$

where $g_\phi$ is the embedding function induced by the pretrained model and its context, $\phi$ represents pretrained model parameters, and $z_i$ is the row embedding.

The downstream supervised model then receives an augmented feature vector:

$$
\tilde{x}_i = [x_i, z_i]
$$

and XGBoost is trained as

$$
\hat{p}_i = h_\theta(\tilde{x}_i)
$$

This is the key similarity and difference from supervised ML:

- The downstream fraud scorer is still supervised ML. It still learns from labelled training rows, is tuned on validation rows, and is evaluated on future holdout rows.
- The new capability comes from the tabular foundation model representation step. TabPFN or TabICL can produce additional row-level features after conditioning on a task context, without training a new XGBoost-like model from scratch inside the TFM itself.

This is why I think embeddings are a useful integration pattern to test. They let a team ask whether a foundation model can improve the representation of the data while keeping the final scorer, monitoring path, and operating metrics closer to an existing classical ML workflow.

### Why the context window matters

A tabular foundation model uses labelled context rows as part of the task description. That creates an immediate evaluation concern: a row's own label should not be used to create features for that same row.

The notebook therefore sorts the fraud dataset by `Time` and uses five chronological windows:

- earliest 20% of transactions: TabPFN/TabICL representation context;
- next 40%: downstream XGBoost training period;
- next 10%: validation period for model selection;
- next 10%: calibration period for post-hoc probability calibration;
- final 20%: full holdout.

TabPFN and TabICL see only the earlier representation-context rows when generating embeddings for the later training, validation, calibration, and holdout rows. XGBoost then sees raw features or raw plus embedding features. The full holdout remains a future window relative to both the representation context and the downstream training process.

This split is still a simplified public-data workflow. A real production fraud project would also need customer, card, account, merchant, device, and label-availability information. This dataset does not expose those fields. Still, the chronological design is more relevant than a random split for a fraud review workflow.

### Why Average Precision is not enough

Fraud detection is a rare-event ranking problem. The public credit-card fraud dataset has 284,807 transactions and 492 fraud cases. The fraud rate is about 0.1727%.

Accuracy is not a useful headline metric here. A model that predicts "not fraud" for every row would be more than 99% accurate and still catch no fraud.

The notebook therefore focuses on precision, recall, Average Precision, and alert counts.

Precision is

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Recall is

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

where $TP$ means true positives, $FP$ means false positives, and $FN$ means false negatives.

Average Precision summarizes the precision-recall curve. One useful way to write it is

$$
AP = \sum_{k=1}^{K} (R_k - R_{k-1})P_k
$$

where $P_k$ and $R_k$ are precision and recall values along the ranked score curve.

Average Precision is useful, but a fraud team often works with review capacity. That means we also need questions such as:

- If we review the top 100 alerts, how many frauds do we catch?
- If we review the top 1% of transactions, what recall do we get?
- How many alerts are needed to reach 80% recall?
- How many alerts are needed to reach 90% recall?

This is where a model can have slightly lower AP but still be operationally interesting. If it reaches a high recall target with fewer alerts, it may help a review team even if its average curve summary is not the best.

### Calibration is a separate question

A fraud score can be useful for ranking without being a well-calibrated probability. Calibration asks whether predicted probabilities match observed frequencies. If a model assigns a group of transactions a fraud probability near 1%, then a calibrated model should see about 1% fraud in that group.

The notebook reports Brier score,

$$
\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(\hat{p}_i - y_i)^2
$$

and log loss,

$$
\text{LogLoss} =
-\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)
\right]
$$

Lower is better for both. The notebook also reports reliability-bin summaries and expected calibration error. I treat calibration as diagnostic in this post because the public holdout contains only 75 fraud cases. That is enough to inspect behavior, but not enough to claim production probability calibration.

## Hands-on demo

### Notebook setup and run state

The notebook is designed to run on Kaggle with GPU enabled. It installs `cudf-cu12`, uses the cuDF pandas accelerator, generates TabPFN and TabICL embeddings on CUDA devices, and uses a GPU XGBoost path for the downstream scorer.

The v2 completed run used:

| Component | Version |
|---|---:|
| Python | 3.12.12 |
| CUDA devices | 2 |
| pandas | 2.3.3 |
| cuDF | 26.2.1 |
| CuPy | 14.0.1 |
| NumPy | 2.0.2 |
| scikit-learn | 1.6.1 |
| XGBoost | 3.2.0 |
| torch | 2.10.0+cu128 |
| TabPFN | 7.1.1 |
| TabICL | 2.1.1 |

The v2 notebook ran without notebook execution errors. The current v3 source adds more provenance, memory, and uncertainty outputs, but those v3 outputs still need a fresh completed GPU run.

### Split and tuning policy

The split and tuning policy are central to the result. The fair raw-vs-embedding comparison excludes the earliest representation-context rows from downstream tuning, because those labels were used to condition TabPFN and TabICL embeddings.

In the v2 run:

| Comparison | Tuning rows | Fraud rows | Context rows excluded | Sampled train used |
|---|---:|---:|---:|---:|
| fair raw-vs-embedding | 142,403 | 227 | 56,961 | false |
| raw all-history incumbent | 199,364 | 384 | 0 | false |

The fair raw-vs-embedding path had five valid chronological folds. This matters because the earlier workflow had too little valid chronological tuning evidence after fraud-count checks. The v2/v3 design gives the XGBoost tuner more chronological history while still respecting the TFM context boundary.

### Feature bundles

The v2 run evaluated the intended feature bundles:

| Feature set | Embedding source | Feature count | Shared feature prep seconds |
|---|---|---:|---:|
| Raw | none | 29 | 0.0 |
| Raw + TabPFN embeddings | TabPFN | 221 | 373.2 |
| Raw + TabICL embeddings | TabICL | 541 | 120.3 |

There is no combined `Raw + TabPFN + TabICL` row in the v2 publication workflow. I removed it because the main question is one-embedding-source adoption. This makes the result less dramatic than the initial completed run, where the combined row had the best AP, but I think it makes the comparison more realistic.

The two embedding paths are also technically different. TabPFN exposes a public `get_embeddings` method. TabICL does not expose the same sklearn-level public embedding method, so the notebook extracts representations through the fitted TabICL wrapper internals and records the version/checkpoint metadata. That is useful for experimentation, but it should be treated as a version-sensitive path.

### Full-holdout ranking result

The full holdout is the deployment-facing view because it keeps the original final-window fraud base rate. The result is conservative:

| Model | AP | Workflow seconds | Top 0.5% recall | Top 1% recall | Brier | ECE 10 |
|---|---:|---:|---:|---:|---:|---:|
| Raw all-history XGBoost | 0.8097 | 166.5 | 0.8533 | 0.8533 | 0.000424 | 0.000104 |
| Raw XGBoost | 0.8034 | 139.2 | 0.8400 | 0.8667 | 0.000403 | 0.000231 |
| Raw + TabICL | 0.7970 | 1134.1 | 0.8267 | 0.9067 | 0.000451 | 0.000485 |
| Raw + TabPFN | 0.7909 | 825.3 | 0.8000 | 0.8400 | 0.000381 | 0.000184 |

The raw all-history XGBoost incumbent has the best full-holdout Average Precision. The ordinary raw XGBoost row is close and fastest. Neither single-source embedding workflow improves AP over the raw XGBoost baselines in this run.

This is an important outcome. It means that the notebook is not a marketing-style example where the foundation-model feature always wins. It is a workflow test. In this public fraud dataset, the classical incumbent is strong.

The precision-recall figure shows the same situation visually.

![Full-holdout precision-recall curves](./20260505-tabpfn-tabicl-embeddings-fraud-det-workflows.assets/publication_precision_recall_curves_full.png)

The curves are close. The dashed horizontal line is the fraud base rate, which is near zero. When the curves are this close, I do not think the figure alone should carry the interpretation. The operating-point tables are more useful for understanding where a model may matter in practice.

### Alert-budget interpretation

At 80% recall, the raw all-history incumbent is best:

| Model | Alerts needed | Frauds found | Precision |
|---|---:|---:|---:|
| Raw all-history XGBoost | 94 | 60 | 0.6383 |
| Raw XGBoost | 116 | 60 | 0.5172 |
| Raw + TabICL | 127 | 60 | 0.4724 |
| Raw + TabPFN | 199 | 60 | 0.3015 |

At 90% recall, the result changes:

| Model | Alerts needed | Frauds found | Precision |
|---|---:|---:|---:|
| Raw + TabICL | 490 | 68 | 0.1388 |
| Raw all-history XGBoost | 1,064 | 68 | 0.0639 |
| Raw XGBoost | 1,177 | 68 | 0.0578 |
| Raw + TabPFN | 1,598 | 68 | 0.0426 |

This is the main operational signal from the v2 run. TabICL does not win by full-holdout AP, but it needs far fewer alerts to recover 68 of the 75 fraud cases in the full holdout. For a review team targeting very high recall, that is worth further investigation.

This also shows why an evaluation should not stop at one metric. AP is a useful ranking summary. Alert counts translate the score into a workflow. Both are needed.

### Runtime and workflow cost

The runtime plot makes the tradeoff visible:

![Full-holdout runtime versus Average Precision](./20260505-tabpfn-tabicl-embeddings-fraud-det-workflows.assets/publication_runtime_vs_average_precision_full.png)

The raw and raw all-history rows are much faster. The embedding rows are slower because they add an offline representation step and because the downstream XGBoost search runs on wider matrices.

The timing table is:

| Feature set | Embedding prep seconds | Total workflow seconds |
|---|---:|---:|
| Raw | 0.0 | 139.2 |
| Raw all-history incumbent | 0.0 | 166.5 |
| Raw + TabPFN | 373.2 | 825.3 |
| Raw + TabICL | 120.3 | 1134.1 |

One nuance is easy to miss: TabICL embedding preparation was faster than TabPFN embedding preparation in this run, but the total TabICL workflow was slower. The reason is that the TabICL feature matrix was wider, and XGBoost tuning on that matrix took longer.

This is why I prefer the phrase workflow seconds rather than model seconds. In a production-style test, the representation is only useful if its quality or operating-point gain justifies the whole path: extraction, storage, downstream fitting, prediction, calibration, and monitoring.

### Calibration diagnostics

Calibration is not a clear win in the v2 run.

Full-holdout calibration rows:

| Row | AP | Brier | Log loss | ECE 10 |
|---|---:|---:|---:|---:|
| Raw all-history none | 0.8097 | 0.000424 | 0.002824 | 0.000104 |
| Raw base | 0.8047 | 0.000396 | 0.002835 | 0.000209 |
| Raw calibrated | 0.8047 | 0.000422 | 0.003287 | 0.000356 |
| Raw + TabPFN none | 0.7909 | 0.000381 | 0.002500 | 0.000184 |
| Raw + TabPFN base | 0.7892 | 0.000408 | 0.002685 | 0.000161 |
| Raw + TabPFN calibrated | 0.7892 | 0.000435 | 0.003340 | 0.000501 |
| Raw + TabICL none | 0.7970 | 0.000451 | 0.002583 | 0.000485 |
| Raw + TabICL base | 0.7978 | 0.000445 | 0.002719 | 0.000359 |
| Raw + TabICL calibrated | 0.7978 | 0.000430 | 0.003091 | 0.000298 |

Sigmoid calibration leaves AP unchanged for each calibration-base row, as expected for a monotonic score transform. It worsens log loss for the selected rows. It improves TabICL Brier/ECE relative to the TabICL calibration base, but not enough to make calibration an obvious overall improvement.

The calibration plot is included below, but this is one place where the table is more useful than the figure. Most meaningful probabilities are near zero because fraud is rare.

![Full-holdout calibration curves](./20260505-tabpfn-tabicl-embeddings-fraud-det-workflows.assets/publication_calibration_curves_full.png)

The v3 notebook tightens this plot and keeps reliability-bin CSVs as the primary calibration artifact.

### Leakage and reuse checks

The notebook checks the public dataset as far as the available fields allow:

| Check | Status | Interpretation |
|---|---|---|
| Target excluded from features | pass | `Class` is not used as a model feature |
| Time preserved for temporal split | pass | `Time` is available for chronological splitting |
| Time used as model feature | not used | default excludes `Time` |
| Chronological split order | pass | equal boundary timestamps can occur |
| Duplicate rows and duplicate model-feature rows | review | requires interpretation |
| Future aggregate features | unknown | feature lineage hidden |
| Entity-level leakage | unknown | no customer/card/account ID |
| Label delay | unknown | label timing hidden |

The duplicate diagnostics are:

| Metric | Value |
|---|---:|
| exact duplicate rows including time and target | 1,854 |
| exact duplicate groups crossing windows | 0 |
| model-feature duplicate rows | 14,293 |
| model-feature duplicate groups crossing windows | 2,178 |
| fraud rows inside cross-window model-feature duplicate groups | 0 |

I read this as a review finding, not as a fatal leakage failure. The public dataset does not expose enough raw identifiers to know whether repeated-looking rows represent real recurring payments, retries, preprocessing artifacts, or duplicates. In a company dataset, this section would need to be much stronger.

## Known shortcomings

There are several limitations that should be kept in view.

First, this is one public dataset. The result should not be generalized to all fraud datasets, all transaction workflows, or all tabular foundation models.

Second, the public credit-card fraud dataset is anonymized. The `V1` to `V28` columns are PCA-style features. There are no customer, card, account, merchant, device, chargeback timing, or feature-lineage fields. That limits leakage analysis, entity-level validation, drift analysis, and business interpretation.

Third, the full holdout has only 75 fraud cases. That makes AP and alert-count differences sensitive to a small number of ranked examples. The v3 notebook adds bootstrap uncertainty for AP and target-recall alert counts so the next completed run can make this uncertainty visible.

Fourth, TabICL embedding extraction uses model internals rather than a stable public embedding API equivalent to TabPFN's `get_embeddings`. That does not make the experiment invalid, but it means the code path should be version-pinned and reviewed.

Fifth, the current post discusses completed v2 outputs while the notebook source is now v3. The v3 notebook adds memory, provenance, publication operating-point tables, and bootstrap uncertainty, but it still needs a fresh Kaggle GPU execution before those new artifacts can be interpreted.

Sixth, I have not yet added interpretability methods such as SHAP, missing-data stress tests, categorical stress tests, or drift-by-period analysis. Those are important for a broader testbench, but I kept this notebook focused on the first embedding workflow question.

## Summary and Conclusion

This notebook tests a practical integration pattern for tabular foundation models:

1. keep XGBoost as the downstream fraud scorer;
2. use TabPFN or TabICL as an offline row-embedding generator;
3. append those embeddings to raw transaction features;
4. evaluate the result with chronological splits, AP, alert counts, runtime, calibration diagnostics, and leakage checks.

The completed v2 result is not a simple "TFM embeddings win" result. The raw all-history XGBoost incumbent has the best full-holdout Average Precision, and raw XGBoost is close while being much faster. Single-source TabPFN and TabICL embeddings do not beat the raw baselines by AP in this run.

The useful nuance is the high-recall operating point. At 90% recall, Raw + TabICL needed 490 alerts to recover 68 of 75 fraud cases, compared with 1,064 alerts for raw all-history XGBoost and 1,177 alerts for raw XGBoost. That makes TabICL worth investigating for high-recall review-queue settings, even though it is not the best AP/runtime row overall.

My interpretation is cautious. This result suggests that tabular foundation model embeddings should be evaluated as workflow components, not only as standalone model scores. The right question is not only "did the embedding row win AP?" It is also "did the embedding row improve the operating point enough to justify the added representation path?"

For practitioners, the reusable lesson is the evaluation design. A TFM embedding experiment should be compared against a serious classical baseline, with time-aware splitting, alert-budget metrics, calibration diagnostics, runtime accounting, memory accounting, and leakage checks.

For labs and researchers, I hope this kind of notebook is useful as a field-facing test. It does not replace formal benchmarks, but it can show how model capabilities appear when inserted into workflows that data teams already understand.

## Outlook

The next step is to run the v3 notebook on Kaggle and review the new artifacts:

- `publication_alert_summary.csv`;
- `publication_target_recall_summary.csv`;
- `publication_bootstrap_uncertainty_full.csv`;
- `source_provenance_summary.csv`;
- `embedding_matrix_summary.csv`;
- `cuda_memory_summary.csv`.

The main questions for the rerun are:

1. Does raw all-history XGBoost still lead by AP?
2. Does TabICL still reduce the alert count at 90% recall?
3. Are the AP differences small relative to bootstrap uncertainty?
4. How much CUDA memory do the TabPFN and TabICL embedding paths use?
5. Are the cleaned publication figures ready to include directly?

After that, I want to extend the workflow in directions that matter for real data science teams: interpretability, missing-data behavior, categorical features, time-derived feature policy, drift, group-aware splits, and additional datasets. My current goal is not to prove that one model family is always better. It is to build reusable examples that make the benefits, costs, and caveats visible enough for both model builders and practitioners to reason about them.
