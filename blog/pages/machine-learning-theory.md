# Machine Learning Theory

This page covers the machine learning concepts used in the posts. It does not
cover data cleaning, feature processing, software engineering, deployment, or
business evaluation. Those topics belong to the Data Science, Coding
Fundamentals, Machine Learning Engineering, and Business Case pages.

The central theory question is:

> Given examples, a target, a model class, and a loss, what function are we
> trying to learn?

## Supervised Learning

In supervised learning, we observe labelled examples:

$$
\mathcal{D}
=
\{(x_i,y_i)\}_{i=1}^{n}.
$$

The goal is to learn a function:

$$
f: \mathcal{X} \rightarrow \mathcal{Y}
$$

that predicts \(y\) from \(x\).

For classical supervised ML, fitting can often be written as empirical risk
minimization:

$$
\hat{f}
=
\arg\min_{f \in \mathcal{F}}
\frac{1}{n}
\sum_{i=1}^{n}
\ell(y_i,f(x_i))
+
\lambda\Omega(f).
$$

Here:

- \(\mathcal{F}\) is the model class;
- \(\ell\) is the loss function;
- \(\Omega(f)\) is a regularization penalty;
- \(\lambda\) controls the strength of regularization.

Different algorithms differ in \(\mathcal{F}\), \(\ell\), optimization, and
regularization.

## Classification

In binary classification:

$$
y_i \in \{0,1\}.
$$

A probabilistic classifier estimates:

$$
\eta(x)
=
P(Y=1 \mid X=x).
$$

A hard classifier can be obtained by thresholding:

$$
\hat{y}
=
\mathbf{1}\{\eta(x) \geq \tau\}.
$$

The model theory is about estimating \(\eta(x)\). The choice of \(\tau\) is a
decision-layer question and belongs outside pure model theory.

## Regression

In regression, the target is numerical:

$$
y_i \in \mathbb{R}.
$$

A squared-error regression model estimates the conditional mean:

$$
f^*(x)
=
\mathbb{E}[Y \mid X=x].
$$

This follows because the function minimizing expected squared error is:

$$
f^*
=
\arg\min_f
\mathbb{E}\left[(Y-f(X))^2\right]
=
\mathbb{E}[Y \mid X].
$$

An absolute-error regression model targets the conditional median:

$$
f^*(x)
\in
\operatorname{median}(Y \mid X=x).
$$

The loss function therefore changes what the model is theoretically trying to
estimate.

## Ranking

Some prediction problems are not mainly about assigning every row a correct
class. They are about ordering examples.

If a model outputs scores \(s_i\), ranking uses the order:

$$
s_{(1)} \geq s_{(2)} \geq \cdots \geq s_{(n)}.
$$

For rare-event detection and top-k allocation, the ranking can matter more than
the default classification threshold. In theory terms, this means that the
quality of the score ordering is distinct from the quality of calibrated
probabilities.

## Loss Functions

A loss function measures prediction error during fitting or evaluation.

Common losses used or discussed in the posts include:

### Logistic Loss

For binary classification with predicted probability \(\hat{p}_i\):

$$
\ell_i
=
-
\left[
y_i\log(\hat{p}_i)
+
(1-y_i)\log(1-\hat{p}_i)
\right].
$$

Logistic loss rewards assigning high probability to the observed class and
penalizes confident wrong predictions heavily.

### Squared Error

For regression:

$$
\ell_i
=
(y_i-\hat{y}_i)^2.
$$

Squared error penalizes large errors more than small errors because the error is
squared.

### Absolute Error

For regression:

$$
\ell_i
=
|y_i-\hat{y}_i|.
$$

Absolute error is less sensitive to extreme errors than squared error.

### Pinball Loss

For quantile level \(q\), pinball loss is:

$$
\ell_q(y,\hat{q})
=
\max(q(y-\hat{q}), (q-1)(y-\hat{q})).
$$

This loss appears when a model predicts conditional quantiles rather than only a
mean.

## Bias, Variance, and Regularization

Regularization restricts a model so that it does not simply memorize the
training data.

The broad tradeoff is:

\[
\text{generalization error}
\approx
\text{bias}^2
+
\text{variance}
+
\text{irreducible noise}.
\]

High-bias models may be too simple. High-variance models may be too sensitive
to the training sample. Regularization tries to improve out-of-sample
performance by controlling complexity.

Examples:

- Logistic Regression can use coefficient penalties.
- Random Forest controls depth, number of trees, and split behavior.
- XGBoost controls tree depth, learning rate, subsampling, and penalties.
- Tabular foundation models control behavior through pretraining and context
  limits rather than ordinary task-specific refitting.

## Logistic Regression

Logistic Regression models the log odds as a linear function:

$$
\log
\frac{P(Y=1 \mid X=x)}{P(Y=0 \mid X=x)}
=
\beta_0 + \beta^\top x.
$$

Equivalently:

$$
P(Y=1 \mid X=x)
=
\sigma(\beta_0+\beta^\top x),
$$

where:

$$
\sigma(z)
=
\frac{1}{1+e^{-z}}.
$$

Logistic Regression is theoretically useful because it is:

- interpretable through coefficients and log odds;
- a strong baseline for linearly separable signal;
- a calibration reference;
- a simple way to test whether nonlinear models are necessary.

It is not merely a toy model.

## Random Forest

A Random Forest is an ensemble of decision trees.

If each tree is \(T_b(x)\), the regression prediction is:

$$
\hat{f}(x)
=
\frac{1}{B}
\sum_{b=1}^{B}
T_b(x).
$$

For classification, the forest can average class probabilities or votes.

Random Forests reduce variance by combining many decorrelated trees. The
decorrelation comes from:

- bootstrap samples of rows;
- random subsets of features at splits.

The theoretical intuition is that a single tree can be unstable, but an average
of many partially independent trees is more stable.

## Gradient Boosting and XGBoost

Gradient boosting builds an additive model:

$$
F_M(x)
=
\sum_{m=1}^{M}
\nu f_m(x),
$$

where each \(f_m\) is a weak learner and \(\nu\) is the learning rate.

At each step, boosting adds a learner that improves the current model according
to the loss gradient.

XGBoost is a regularized gradient boosting implementation for tree ensembles.
Its practical strength comes from:

- nonlinear interactions;
- additive correction of previous errors;
- regularization;
- shrinkage through learning rate;
- row and column subsampling;
- efficient implementation.

In the posts, XGBoost is important as a strong classical incumbent for tabular
tasks.

## Calibration

A probabilistic classifier is calibrated if predicted probabilities match
observed frequencies.

For binary classification, ideal calibration means:

$$
P(Y=1 \mid \hat{p}(X)=p)
=
p.
$$

Calibration is a model-property question. If a model outputs scores that rank
well but are not calibrated, the scores may still be useful for ordering but
not directly interpretable as probabilities.

Common calibration methods include:

- Platt scaling;
- isotonic regression;
- temperature scaling in broader ML contexts.

The posts use calibration because probability quality matters in fraud and
volatility-style workflows.

## Tabular Foundation Models

A tabular foundation model is pretrained across many synthetic or real tabular
tasks and then applied to a new tabular task.

The key difference from ordinary task-specific fitting is:

- classical ML learns task-specific parameters from the current dataset;
- a tabular foundation model uses pretrained parameters and conditions on the
  current task context.

For a PFN-style model:

$$
\hat{p}(y_* \mid x_*, \mathcal{D}_{\text{context}})
=
f_{\theta}(x_*, \mathcal{D}_{\text{context}}),
$$

where \(\theta\) is pretrained and the labelled context rows describe the
current task.

## Posterior Predictive View

The posterior predictive distribution is:

$$
p(y_* \mid x_*, \mathcal{D})
=
\int
p(y_* \mid x_*, \theta)
p(\theta \mid \mathcal{D})
d\theta.
$$

PFN-style models are motivated by approximating posterior predictive inference
over tasks. They do not literally refit a Bayesian model for each user task in
the ordinary way. Instead, pretraining teaches the model to map task context to
predictive behavior.

This view explains why labelled context rows matter so much.

## TabPFN

TabPFN is a prominent tabular foundation model. The posts study it through:

- conceptual introduction;
- repository reading;
- hands-on classification and regression examples;
- client API experiments;
- embedding extraction;
- fraud, volatility, and allocation workflows.

The theory idea is not that TabPFN is magic. The idea is that it changes the
learning setup. The fitted task information is supplied in context, while the
general inference behavior is learned during pretraining.

## TabICL

TabICL is another in-context learning approach for tabular data.

The posts treat TabICL as commercially important because open-source and local
execution paths may matter for future applied work. The current limitations
observed in the posts are practical, but the theory concept is the same broad
one:

$$
\text{prediction} =
\text{pretrained in-context model}
(\text{context rows}, \text{query row}).
$$

The theory question is how much task structure can be inferred from context and
how that compares with fitting a task-specific model.

## Embeddings and Representations

An embedding is a learned representation:

$$
z_i = \phi_{\theta}(x_i).
$$

The posts explore whether TFM-derived embeddings can improve downstream models.
The theoretical hope is that \(\phi_{\theta}\) captures useful structure that
raw features do not expose directly.

A downstream model may use:

$$
\tilde{x}_i =
[x_i, z_i].
$$

The representation is useful only if the added signal outweighs the cost of
larger feature dimension, runtime, and possible instability.

## Time Series as Supervised Learning

Some time-series forecasting problems can be reframed as supervised learning by
constructing examples from history:

$$
x_t =
(y_{t-L}, \ldots, y_{t-1}),
\qquad
y_t = y_t.
$$

For multiple series, the example may include item identifiers, calendar
features, or covariates. The theory point is that a temporal problem can become
a supervised learning problem once inputs and targets are defined.

The data science details of alignment and leakage belong to the Data Science
page.

## Causal ML and CATE

The posts introduce causal inference through conditional average treatment
effects.

For treatment \(T \in \{0,1\}\), potential outcomes are \(Y(1)\) and \(Y(0)\).
The CATE is:

$$
\tau(x)
=
\mathbb{E}[Y(1)-Y(0) \mid X=x].
$$

Machine learning can help estimate nuisance functions such as:

$$
\mu_t(x) = \mathbb{E}[Y \mid X=x, T=t],
$$

or propensity scores:

$$
e(x) = P(T=1 \mid X=x).
$$

The theory warning is essential:

> Flexible prediction does not create causal identification.

TabPFN or any other ML model can be a base learner, but causal claims still
depend on assumptions such as consistency, overlap, and unconfoundedness.

## Model Interpretation

Model interpretation asks how a model uses inputs to produce predictions.

The posts use interpretation in a practical sense:

- Logistic Regression coefficients explain linear log-odds effects.
- Tree models expose feature importance and nonlinear interactions.
- Embedding workflows test whether learned representations add signal.
- Calibration curves show whether probabilities are meaningful.

Interpretation is model-specific. A coefficient, a split count, a SHAP value,
and an embedding dimension are not the same kind of explanation.

## Current Depth

The ML theory covered so far is strongest in:

- supervised learning;
- classification and regression;
- ranking;
- losses and calibration;
- Logistic Regression;
- Random Forest;
- XGBoost;
- tabular foundation models;
- posterior predictive intuition;
- embeddings;
- time series reframed as supervised learning;
- CATE as causal ML.

Topics should be added to this page only when future posts actually use them.
