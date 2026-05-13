# Machine Learning Engineering

This page covers the production and MLOps layer. It explains what would be
needed to deploy, monitor, and maintain models after the notebook stage.

The current DSAIEngineering posts are mostly pre-production. They demonstrate
important preparation steps such as validation discipline, artifact saving,
runtime awareness, leakage checks, and calibration analysis. They do not yet
deploy live services, scheduled production pipelines, model registries, or
monitoring systems.

The central ML engineering idea is:

> A model is production software only when it can be reliably trained, shipped,
> observed, updated, and retired.

## Boundary of This Page

This page does not cover:

- software engineering basics such as functions, paths, and dependencies;
- data cleaning and feature construction;
- ML model theory;
- business positioning.

It covers the system that would surround a trained model in production.

## ML System Lifecycle

A production ML system has a lifecycle:

$$
\text{data}
\rightarrow
\text{training}
\rightarrow
\text{validation}
\rightarrow
\text{registration}
\rightarrow
\text{deployment}
\rightarrow
\text{monitoring}
\rightarrow
\text{maintenance}.
$$

The posts currently cover the early part of this lifecycle:

- data preparation;
- target and feature contracts;
- train, calibration, and holdout separation;
- model comparison;
- calibration analysis;
- artifact creation;
- runtime constraints;
- limitations.

The later parts remain future work:

- model registry;
- deployment;
- automated monitoring;
- scheduled retraining;
- rollback;
- service-level objectives;
- production governance.

## Production Readiness

A model is not production-ready merely because it has a good holdout metric.

Production readiness asks:

1. Can the model be trained repeatably?
2. Can the exact model artifact be saved and loaded?
3. Can features be produced at inference time?
4. Can predictions be served within latency and cost constraints?
5. Can failures be detected?
6. Can model behavior be monitored after deployment?
7. Can a bad model be rolled back?
8. Can the system be audited?

The posts are building the evidence needed before answering yes to these
questions.

## Model Artifact

A model artifact is the saved object used for future prediction.

An artifact may include:

- model weights or fitted parameters;
- preprocessing objects;
- calibration objects;
- feature column order;
- class labels;
- package versions;
- training data period;
- metric summary;
- creation timestamp.

The artifact contract is:

\[
\hat{y}
=
g_{\text{artifact}}(x),
\]

where \(g_{\text{artifact}}\) includes all transformations required to turn
input features into a prediction.

If preprocessing and calibration are not saved with the model, then the
artifact is incomplete.

## Training Pipeline

A production training pipeline automates the steps needed to produce a model
candidate.

At minimum, it should:

- load approved data;
- validate schema;
- construct features consistently;
- split data according to policy;
- train candidate models;
- evaluate them;
- save artifacts;
- write metadata.

The current notebooks perform many of these steps manually or semi-manually.
The future engineering step is to turn repeated notebook logic into a scheduled
or callable pipeline.

## Validation Gate

Before deployment, a candidate model should pass a validation gate.

A validation gate can check:

- performance metrics;
- calibration metrics;
- runtime;
- memory use;
- feature availability;
- leakage checks;
- comparison against the current production model;
- stability across relevant slices.

Mathematically, deployment can be treated as a decision:

$$
\text{deploy}
=
\mathbf{1}\{
Q(\text{candidate}) \geq Q(\text{current}) + \delta
\},
$$

where \(Q\) is a quality score and \(\delta\) is the minimum improvement needed
to justify change.

The posts do not yet implement deployment gates, but they compute the kinds of
metrics that such gates would use.

## Model Registry

A model registry stores model versions and metadata.

A registry entry should answer:

- What model was trained?
- On which data period?
- With which feature set?
- With which package versions?
- Which metrics were achieved?
- Which artifact file is deployable?
- Who approved it?
- Is it active, archived, or rejected?

The current work saves artifacts and reports metrics, but it does not yet use a
formal registry.

## Batch Deployment

Batch deployment means predictions are generated on a schedule.

Examples:

- score all transactions from the previous hour;
- score the next-month asset universe;
- compute daily volatility-regime probabilities;
- refresh a risk dashboard each morning.

The production contract is:

$$
\text{input table at time } t
\rightarrow
\text{prediction table for time } t.
$$

Batch systems must handle:

- scheduled runs;
- missing input files;
- late-arriving data;
- partial failures;
- idempotent reruns;
- output versioning.

The finance workflows in the posts are closest to future batch systems, but
they are still notebook experiments.

## Online Deployment

Online deployment means predictions are returned in response to requests.

The production contract is:

$$
\text{request}
\rightarrow
\text{features}
\rightarrow
\text{model}
\rightarrow
\text{response}.
$$

Online systems must handle:

- latency;
- concurrency;
- input validation;
- service errors;
- fallback behavior;
- logging;
- security;
- model version routing.

The current posts do not implement online serving. Fraud detection could
eventually require online or near-real-time scoring, but the public notebooks
only demonstrate offline evaluation.

## Feature Serving

Production inference requires the same feature definitions used during
training. This is a systems problem, not only a data science problem.

The serving-time feature contract is:

$$
X_{\text{serve}}
\sim
\text{same schema and semantics as }
X_{\text{train}}.
$$

The system must guarantee:

- column names match;
- column order is correct where required;
- data types are valid;
- missingness policy is applied;
- values are available at inference time;
- transformations match training.

The posts discuss feature availability and leakage at the notebook level. A
production system would enforce these rules automatically.

## Prediction Logging

A production model should log its predictions.

Useful logs include:

- request or row identifier;
- timestamp;
- model version;
- feature version;
- raw score;
- calibrated probability;
- decision threshold or selected action;
- error codes;
- latency.

Without prediction logs, monitoring and debugging are weak. If performance
degrades later, the team needs to know what the model saw and what it predicted.

## Monitoring

Monitoring checks whether the system is still behaving acceptably after
deployment.

Production monitoring has several layers:

### Data Monitoring

Data monitoring checks whether inputs still look valid.

Examples:

- schema changes;
- missing columns;
- unusual missingness;
- out-of-range values;
- distribution shift.

Population Stability Index is one possible drift measure:

$$
\operatorname{PSI}
=
\sum_b
(p_b-q_b)
\log
\frac{p_b}{q_b},
$$

where \(p_b\) and \(q_b\) are bin proportions in a reference period and a
current period.

### Prediction Monitoring

Prediction monitoring checks the model output distribution.

Useful questions:

- Are scores suddenly concentrated near zero?
- Did the alert rate change?
- Did the top-k set become unstable?
- Did calibrated probabilities drift?
- Did a model stop producing valid values?

### Performance Monitoring

Performance monitoring requires labels. In many real systems, labels arrive
later than predictions.

When labels arrive, the system can recompute:

- ranking metrics;
- classification metrics;
- calibration metrics;
- regression errors;
- decision outcomes.

The label delay is part of the production design.

### System Monitoring

System monitoring checks software behavior:

- latency;
- throughput;
- memory use;
- error rates;
- failed jobs;
- missing artifacts;
- API failures.

This is where ML engineering overlaps with ordinary software operations.

## Maintenance

Models need maintenance because data, behavior, and business requirements
change.

Maintenance includes:

- retraining;
- recalibration;
- threshold updates;
- feature updates;
- dependency updates;
- model retirement;
- documentation updates.

One useful maintenance rule is:

$$
\text{retrain}
=
\mathbf{1}\{
\text{drift is high}
\lor
\text{performance is low}
\lor
\text{scheduled refresh is due}
\}.
$$

The posts do not yet automate retraining. They build the validation and
comparison habits that would support it.

## Rollback

Rollback means returning to a previous model or fallback behavior when a new
model fails.

A rollback-ready system needs:

- versioned artifacts;
- versioned feature definitions;
- stored previous model metadata;
- ability to route traffic back;
- monitoring that detects failure quickly.

No production model should be deployed without a credible rollback path.

## Governance

Governance is the process for approving, documenting, and auditing model use.

For the workflows in the posts, governance would ask:

- What is the intended use?
- What data was used?
- What limitations are known?
- What failure modes are expected?
- What metrics justify deployment?
- Who reviews changes?
- How are incidents handled?

The posts already practice one part of governance: stating limitations. A full
production system would need a more formal process.

## Production Gaps in the Current Work

The current work has not yet implemented:

- reusable training packages;
- automated tests around pipelines;
- model registry entries;
- scheduled training jobs;
- batch scoring jobs;
- online model serving;
- feature serving infrastructure;
- production prediction logging;
- monitoring dashboards;
- automated retraining;
- rollback procedures.

These are not minor details. They are the difference between an analytical
workflow and an operating ML system.

## Current Depth

The ML engineering demonstrated so far is strongest in pre-production
discipline:

- separating train, calibration, and holdout roles;
- saving artifacts;
- comparing baselines;
- measuring runtime;
- documenting leakage risks;
- explaining public-data limitations;
- connecting scores to possible decisions.

The ML engineering still to be built is production MLOps:

- deployable model services;
- batch prediction systems;
- model registries;
- monitoring;
- maintenance;
- rollback;
- governance.

This page should become deeper as future posts move from notebooks toward
production systems.
