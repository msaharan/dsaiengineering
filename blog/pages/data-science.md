# Data Science

This page covers the data science layer demonstrated in the posts: turning raw
or public data into an analyzable table, defining rows and targets, cleaning and
processing features, preventing leakage, and preparing evaluation datasets.

It deliberately avoids three neighboring topics:

- software engineering mechanics belong to Coding Fundamentals;
- model theory and learning algorithms belong to Machine Learning Theory;
- deployment, monitoring, and maintenance belong to Machine Learning
  Engineering.

The central data science idea is:

> A model can only answer the question encoded in the data table.

If the row definition, target, feature policy, or split design is wrong, a
model can produce an impressive metric while answering the wrong question.

## Data Science Object

Most posts work with a supervised tabular dataset:

$$
\mathcal{D}
=
\{(x_i,y_i)\}_{i=1}^{n}.
$$

Here:

- \(i\) indexes the unit of observation;
- \(x_i\) is the feature vector available for that unit;
- \(y_i\) is the label or outcome assigned to that unit.

The data science task is to define what \(i\), \(x_i\), and \(y_i\) mean before
any model is fitted.

## Unit of Observation

The unit of observation is the meaning of one row.

The posts use several row definitions:

- in early tabular demos, one row is one generic supervised example;
- in fraud detection, one row is one transaction;
- in volatility-regime forecasting, one row is one market signal date;
- in tactical allocation, one row is one asset-month pair \((j,t)\);
- in time-series forecasting, one row can be one item-time pair after
  reshaping.

The same DataFrame shape can represent different scientific questions. A
10,000-row fraud table and a 10,000-row asset-month table are not equivalent,
even if both are rectangular.

## Data Sources

The posts use public and reproducible data sources, including:

- credit-card fraud data;
- yfinance market data;
- Cboe VIX history;
- FRED macro and rates data;
- public ETF universes.

For each source, the data science questions are:

1. What population does this data represent?
2. What time period is covered?
3. Which columns are observed?
4. Which columns are missing?
5. Are the values revised after the fact?
6. Is the data point-in-time?
7. What real-world entities are not visible?

Public data makes notebooks reproducible, but it also imposes limitations. For
example, a public fraud dataset may not include merchant IDs, customer histories,
delayed labels, or raw feature lineage. Public market data may not be a full
institutional point-in-time data stack.

## Data Dictionary

A data dictionary explains what each column means.

At minimum, a useful dictionary separates:

- identifiers;
- timestamps;
- raw observed variables;
- engineered features;
- target columns;
- split labels;
- prediction columns;
- diagnostic columns.

This distinction prevents accidental modeling mistakes. An identifier may be
useful for joining tables but inappropriate as a model feature. A target column
may be useful for evaluation but forbidden as an input.

## Table Shape

The basic feature matrix is:

$$
X =
\begin{bmatrix}
x_1^\top \\
x_2^\top \\
\vdots \\
x_n^\top
\end{bmatrix}
\in \mathbb{R}^{n \times p}.
$$

The target vector is:

$$
y =
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix}.
$$

Data science work decides which rows enter \(X\), which columns enter \(p\), and
how \(y\) is constructed.

## Target Construction

The target defines the question.

### Fraud Target

In fraud detection, the target is binary:

$$
y_i =
\begin{cases}
1, & \text{transaction } i \text{ is fraud} \\
0, & \text{otherwise}.
\end{cases}
$$

This creates a rare-event problem because:

$$
\pi = P(Y=1)
$$

is small. The target therefore implies that ranking and precision-focused
evaluation will matter.

### Volatility-Regime Target

In volatility-regime forecasting, the target is constructed from future returns.
For a signal date \(t\) and horizon \(h\), future realized volatility can be
written as:

$$
RV_{t,h}
=
\sqrt{252}
\operatorname{std}(r_{t+1}, r_{t+2}, \ldots, r_{t+h}).
$$

A binary high-volatility label is:

$$
y_t =
\mathbf{1}\{RV_{t,h} \geq c\},
$$

where \(c\) is a threshold such as a quantile computed from the appropriate
training period.

The important data science rule is that \(RV_{t,h}\) uses future data, so it can
only be a label, not a feature available at time \(t\).

### Tactical Allocation Target

In tactical allocation, the target can be cross-sectional top-k membership. For
asset \(j\) at month \(t\), let:

$$
R_{j,t+1}
$$

be next-month return. If \(\mathcal{U}_t\) is the investable universe at time
\(t\), the top-k label is:

$$
y_{j,t}
=
\mathbf{1}
\{
j \in \operatorname{TopK}_{a \in \mathcal{U}_t} R_{a,t+1}
\}.
$$

The base rate is approximately:

$$
\frac{K}{|\mathcal{U}_t|}.
$$

This is a data design choice, not a model architecture choice.

## Feature Policy

A feature policy states which columns are allowed as inputs.

For a time-indexed task, a valid feature at time \(t\) must be observable at or
before \(t\):

$$
x_t \in \mathcal{I}_t,
$$

where \(\mathcal{I}_t\) is the information set available at signal time.

Examples of feature families used in the posts include:

- lagged market returns;
- rolling volatility;
- trend and momentum variables;
- VIX and VIX-derived features;
- macro and rates features;
- asset-level return history;
- missingness indicators.

The feature policy should also state which fields are excluded:

- target columns;
- future returns;
- future volatility;
- post-outcome diagnostics;
- split labels;
- prediction columns from the same evaluation stage.

## Missing Data

Missing values are data, but they need an explicit policy.

Let \(x_{ij}\) be feature \(j\) for row \(i\). A missingness indicator is:

$$
m_{ij}
=
\mathbf{1}\{x_{ij} \text{ is missing}\}.
$$

The posts use missingness-aware design because public finance features often
start on different dates. Dropping every row with any missing value can destroy
the usable history. Imputing without indicators can hide important availability
patterns.

A practical data science policy is:

1. identify missingness;
2. decide whether missingness is expected or suspicious;
3. impute where appropriate;
4. add missingness indicators when the missingness pattern may be informative;
5. record which features are affected.

## Time Alignment

Time alignment is one of the most important data science responsibilities.

For a signal date \(t\):

- features must be available by \(t\);
- labels may use outcomes after \(t\);
- evaluation must not let future rows influence past decisions.

A clean time-indexed dataset separates:

$$
\text{signal time} \neq \text{outcome time}.
$$

For example, if a monthly allocation decision is made at the end of month \(t\),
then next-month return \(R_{t+1}\) can define the target but cannot be used as a
feature.

## Data Splits

A split is a data design decision.

For time-ordered work, the posts use chronological separation:

$$
t_{\text{train}} < t_{\text{validation}} < t_{\text{calibration}} <
t_{\text{holdout}}.
$$

The exact names vary by notebook, but the principle is stable:

- earlier data supports fitting or context construction;
- intermediate data supports model selection or calibration;
- final data supports honest evaluation.

Random splitting can be inappropriate when rows are ordered through time or
when future outcomes are used to construct labels.

## Leakage

Leakage occurs when the dataset gives the model information that would not be
available at prediction time.

Common leakage risks in the posts include:

- using future returns as features;
- computing thresholds from the full dataset instead of the training period;
- letting holdout rows influence calibration;
- reusing labels incorrectly in embedding workflows;
- allowing overlapping time windows to create overly optimistic uncertainty;
- treating public revised data as if it were point-in-time.

The data science standard is not to claim that leakage is impossible. The
standard is to state what leakage checks were performed and which risks remain.

## Data Cleaning

Data cleaning means making the table internally coherent before analysis.

Typical cleaning checks include:

- parse dates correctly;
- sort by timestamp and entity;
- remove impossible rows;
- verify duplicate keys;
- standardize column names;
- convert numeric columns to numeric types;
- handle missing values intentionally;
- inspect start and end dates;
- check class balance or target prevalence.

Cleaning is not cosmetic. A wrong date type or duplicate key can invalidate a
time split.

## Feature Engineering

Feature engineering turns raw observations into useful variables.

In the posts, feature engineering includes:

- lags;
- rolling summaries;
- volatility estimates;
- momentum and trend variables;
- cross-sectional asset features;
- macro and rate joins;
- missingness flags.

For a rolling feature with window \(w\):

$$
z_t
=
g(x_{t-w+1}, \ldots, x_t),
$$

where \(g\) might be a mean, standard deviation, drawdown, or cumulative return.

The right edge of the window matters. A feature for time \(t\) should not use
\(x_{t+1}\).

## Exploratory Data Analysis

Exploratory data analysis asks what the table looks like before modeling.

Useful checks include:

- number of rows and columns;
- date coverage;
- target prevalence;
- class imbalance;
- feature missingness;
- distribution ranges;
- outliers;
- duplicated entities or timestamps;
- correlations or obvious proxies;
- train versus holdout distribution changes.

EDA is not separate from rigor. It is how many invalid assumptions are found.

## Evaluation Dataset Construction

Evaluation begins before metrics are computed. The evaluation dataset must
represent the claim.

For fraud:

- the holdout should preserve full prevalence;
- metrics should reflect alert ranking and probability quality;
- public-data limits such as missing entity history should be stated.

For volatility:

- the holdout should be later in time;
- high-volatility thresholds should be defined without holdout leakage;
- overlapping labels should be acknowledged.

For tactical allocation:

- each month should have a clear investable universe;
- top-k membership should be computed within month;
- portfolio diagnostics should be separated from model fitting.

## Artifacts as Data Products

The posts save or discuss outputs such as:

- processed tables;
- prediction tables;
- metric summaries;
- split summaries;
- calibration outputs;
- portfolio diagnostic tables.

These artifacts are data products. They allow later inspection without rerunning
the full workflow.

An artifact should make clear:

- what rows it contains;
- which model or method produced it;
- which split it belongs to;
- which columns are predictions, labels, or diagnostics;
- when it was generated.

## Current Depth

The data science work demonstrated so far is strongest in:

- defining tabular prediction tasks;
- constructing time-aware targets;
- managing public-data limitations;
- building feature policies;
- handling missingness;
- preserving chronological splits;
- documenting leakage risks;
- turning model outputs into evaluation datasets.

The work is not yet a full data platform. It does not include production data
contracts, feature stores, streaming ingestion, institutional point-in-time
finance data, or automated data-quality services. Those belong to future ML
engineering and data engineering work.
