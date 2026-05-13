# Probability and Statistics

This page collects the probability and statistics concepts that have appeared in the DSAIEngineering posts so far. The goal is not to create a general statistics textbook. The goal is to make the statistical tools used in the posts explicit enough that a reader can return here when reading the notebooks.

The central principle is:

> A model metric is not a fact about the world. It is an estimate computed from data under assumptions.

That principle appears in confidence intervals, chi-square goodness-of-fit, posterior predictive distributions, calibration, bootstrap uncertainty, and finance-specific block resampling.

## Random variables and notation

A random variable is a quantity whose value is uncertain before it is observed. In supervised learning, the feature vector and target are often written as random variables:

$$
X \in \mathcal{X},
\quad
Y \in \mathcal{Y}.
$$

For a concrete row \(i\), the observed feature vector and target are:

$$
x_i,
\quad
y_i.
$$

This distinction matters. A model is trained or contextualized on observed values, but its purpose is to make statements about future or unobserved values.

In the posts, the main target types are:

- binary fraud labels;
- high-volatility regime labels;
- top-k allocation labels;
- regression targets in the TabPFN demo;
- future time-series values in the TabPFN-TS example;
- treatment effects in the CATE example.

## Conditional probability

Most model scores can be read as conditional statements. In binary classification, a probabilistic model tries to estimate:

$$
\mathbb{P}(Y=1 \mid X=x).
$$

In a context-conditioned TFM workflow, the conditioning includes the labelled context:

$$
\mathbb{P}(Y_\ast=1 \mid X_\ast=x_\ast, X_{\text{context}}, y_{\text{context}}).
$$

In words: given the query row and the labelled context rows, what is the probability that the unknown label is positive?

Conditional probability is also the language of calibration. If a model gives a score near 0.30 to many rows, calibration asks whether about 30% of those rows are actually positive.

## Classification error and confidence intervals

P1 introduced the idea that model performance should be reported with uncertainty. For a binary or multiclass classifier, accuracy is:

$$
\text{accuracy} =
\frac{\text{number of correct predictions}}{\text{number of predictions}}.
$$

Classification error is:

$$
\text{error} =
\frac{\text{number of incorrect predictions}}{\text{number of predictions}}.
$$

If a model has error rate \(\hat{e}\) on \(n\) independent examples, a simple Wald-style confidence interval is:

$$
\hat{e}
\pm
z_{\alpha/2}
\sqrt{
\frac{\hat{e}(1-\hat{e})}{n}
}.
$$

For a 95% interval, \(z_{\alpha/2}\approx 1.96\). This is simple and useful for intuition, but it depends on assumptions: the evaluation examples should be approximately independent, and the sample should be large enough that the normal approximation is not too poor.

The later posts move beyond this simple interval because fraud and finance workflows often violate independence assumptions.

## Cross-validation uncertainty

In P8, repeated cross-validation was used to report mean metric values and uncertainty. If a metric is computed across \(m\) folds or repeated folds:

$$
s_1, s_2, \ldots, s_m,
$$

the mean score is:

$$
\bar{s} = \frac{1}{m}\sum_{j=1}^{m}s_j.
$$

A simple standard error is:

$$
\text{SE}(\bar{s}) =
\frac{\hat{\sigma}_s}{\sqrt{m}},
$$

where \(\hat{\sigma}_s\) is the sample standard deviation of the fold scores. A Student-\(t\) interval is:

$$
\bar{s}
\pm
t_{0.975,m-1}
\frac{\hat{\sigma}_s}{\sqrt{m}}.
$$

This is useful for the early static demos. It is less appropriate when fold scores are strongly dependent or when chronological validation is required.

## Chi-square distribution

P2 introduced the chi-square distribution. If:

$$
Z_1,\ldots,Z_\nu \sim \mathcal{N}(0,1)
$$

are independent standard normal variables, then:

$$
\chi^2 = \sum_{i=1}^{\nu}Z_i^2
$$

has a chi-square distribution with \(\nu\) degrees of freedom.

Its probability density is:

$$
f(\chi^2;\nu)
=
\frac{(\chi^2)^{\nu/2-1}e^{-\chi^2/2}}
{2^{\nu/2}\Gamma(\nu/2)}.
$$

The expected value is:

$$
\mathbb{E}[\chi^2] = \nu,
$$

and the variance is:

$$
\operatorname{Var}(\chi^2) = 2\nu.
$$

This distribution becomes useful in goodness-of-fit tests when residuals are independent, Gaussian, and scaled by known standard deviations.

## Goodness of fit

Suppose measurements are:

$$
y_i \pm \sigma_i,
\quad i=1,\ldots,N,
$$

and a model predicts:

$$
\hat{y}_i = f(x_i;\hat{\theta}).
$$

The residual is:

$$
r_i = y_i - \hat{y}_i.
$$

The chi-square statistic is:

$$
\chi^2 =
\sum_{i=1}^{N}
\frac{(y_i - f(x_i;\hat{\theta}))^2}{\sigma_i^2}.
$$

If the model is appropriate, errors are independent and Gaussian, and the \(\sigma_i\) values are correct, then \(\chi^2\) can be compared to a chi-square distribution with:

$$
\nu = N - m
$$

degrees of freedom, where \(m\) is the number of fitted parameters.

The reduced chi-square is:

$$
\chi^2_\nu = \frac{\chi^2}{\nu}.
$$

As a rough diagnostic:

- \(\chi^2_\nu \approx 1\): residuals are broadly consistent with the assumed uncertainties.
- \(\chi^2_\nu \gg 1\): the model may fit poorly, uncertainties may be underestimated, or systematics may be missing.
- \(\chi^2_\nu \ll 1\): uncertainties may be overestimated, residuals may be correlated, or the model may be too flexible.

The important lesson for ML is not that every model should use chi-square. The lesson is that metrics are meaningful only under assumptions.

## Likelihood and negative log likelihood

If measurement errors are independent Gaussian variables:

$$
y_i \sim \mathcal{N}(f(x_i;\theta), \sigma_i^2),
$$

then the likelihood is:

$$
L(\theta)
=
\prod_{i=1}^{N}
\frac{1}{\sqrt{2\pi\sigma_i^2}}
\exp
\left[
-
\frac{(y_i-f(x_i;\theta))^2}{2\sigma_i^2}
\right].
$$

Maximizing this likelihood is equivalent to minimizing negative log likelihood. Up to constants, this becomes the chi-square objective:

$$
-2\log L(\theta)
=
\sum_{i=1}^{N}
\frac{(y_i-f(x_i;\theta))^2}{\sigma_i^2}
+ \text{constant}.
$$

Negative log likelihood also appears in the TFM pretraining discussion. Tabular foundation models are pretrained to predict held-out target cells from synthetic supervised tasks. The loss is a negative log likelihood over test target cells, averaged over many sampled tasks.

## Posterior predictive distribution

The posterior predictive distribution is the most important probability object for the TabPFN and TabICL posts.

For a new row \(x_{\text{new}}\), labelled context \(X_{\text{train}}, y_{\text{train}}\), and unknown target \(y\), the object is:

$$
p(y \mid x_{\text{new}}, X_{\text{train}}, y_{\text{train}}).
$$

This is the distribution of the unknown target after conditioning on the current task context.

If \(\phi\) denotes a latent supervised learning task, then:

$$
p(y \mid x_{\text{new}}, X_{\text{train}}, y_{\text{train}})
=
\int
p(y \mid x_{\text{new}}, \phi)
p(\phi \mid X_{\text{train}}, y_{\text{train}})
d\phi.
$$

Using Bayes' rule:

$$
p(\phi \mid X_{\text{train}}, y_{\text{train}})
\propto
p(X_{\text{train}}, y_{\text{train}} \mid \phi)p(\phi).
$$

Substituting this into the predictive distribution gives the intuition:

$$
p(y \mid x_{\text{new}}, X_{\text{train}}, y_{\text{train}})
\propto
\int
p(y \mid x_{\text{new}}, \phi)
p(X_{\text{train}}, y_{\text{train}} \mid \phi)
p(\phi)
d\phi.
$$

In words, the model averages predictions over possible latent tasks, weighted by how plausible those tasks are under the prior and the observed context data.

TabPFN and TabICL do not explicitly compute this integral at inference time. They are pretrained to approximate this kind of inference in one forward-pass style workflow.

## Priors over tasks

In ordinary Bayesian modeling, a prior expresses assumptions before seeing data. In PFN-style tabular foundation models, the prior is a procedure for generating supervised learning tasks during pretraining.

The posts discuss synthetic task generation through structural causal models. A sampled task can include:

- number of rows;
- number of features;
- feature-target relationships;
- noise levels;
- class structure;
- hidden variables;
- train/test split inside the synthetic table.

This is important because the prior determines what kinds of tabular problems the model learns to solve. The prior is not a background detail. It is one of the main sources of model behavior.

## Precision, recall, and rare-event ranking

Fraud detection and volatility scoring made precision and recall central.

For binary classification:

$$
\text{Precision} =
\frac{TP}{TP+FP},
$$

$$
\text{Recall} =
\frac{TP}{TP+FN}.
$$

Precision asks: among rows flagged as positive, what fraction are truly positive?

Recall asks: among all positive rows, what fraction did we catch?

In rare-event settings, accuracy is often misleading. If fraud appears in only about 0.17% of transactions, a model predicting "not fraud" for every transaction can have very high accuracy and still be operationally useless.

That is why the fraud notebooks focus on rankings and alert queues.

## Average Precision

Average Precision summarizes the precision-recall curve. One way to write it is:

$$
AP =
\sum_{k=1}^{K}
(R_k - R_{k-1})P_k,
$$

where \(P_k\) and \(R_k\) are precision and recall at threshold step \(k\).

AP is useful when the positive class is rare or when the decision process is ranked. In the posts, AP is used for:

- fraud review queues;
- high-volatility risk queues;
- tactical allocation top-k membership.

AP should be read against the base rate. In tactical allocation, the base rate is structurally determined by \(K/N\). For example:

$$
\bar{y}_t = \frac{3}{9}=0.333
$$

in the 9-ETF top-3 setup, while:

$$
\bar{y}_t = \frac{5}{25}=0.200
$$

in the 25-ETF top-5 setup.

Therefore, AP values across different target definitions are not directly comparable unless the base rate is considered.

## ROC AUC

ROC AUC measures pairwise ranking quality. It can be interpreted as the probability that a randomly chosen positive row receives a higher score than a randomly chosen negative row:

$$
\operatorname{AUC}
=
\mathbb{P}(s^+ > s^-).
$$

ROC AUC is useful, but it can look good even when the top of the alert queue is not operationally useful. That is why the posts usually treat AP and operating-point metrics as more relevant for rare-event and ranked decision workflows.

## Brier score

Brier score measures squared error of probability predictions:

$$
\text{Brier}
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{p}_i - y_i)^2.
$$

Here, \(\hat{p}_i\) is the predicted probability and \(y_i \in \{0,1\}\). Lower is better.

Brier score mixes probability quality and class prevalence. It is useful, but it should be read with calibration plots and ranking metrics.

## Log loss

Log loss is:

$$
\text{LogLoss}
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log(\hat{p}_i)
+
(1-y_i)\log(1-\hat{p}_i)
\right].
$$

It penalizes confident wrong predictions strongly. Lower is better.

In the posts, log loss is used as a probability-quality diagnostic, not as the only model-selection criterion.

## Expected calibration error

Expected calibration error, or ECE, bins predictions by score and compares predicted probabilities with observed frequencies.

If bin \(b\) contains \(n_b\) rows, average predicted probability \(\operatorname{conf}(b)\), and observed positive rate \(\operatorname{acc}(b)\), then:

$$
ECE
=
\sum_{b=1}^{B}
\frac{n_b}{n}
\left|
\operatorname{acc}(b) - \operatorname{conf}(b)
\right|.
$$

ECE is useful because a model can rank well while producing poorly calibrated probabilities.

In the posts, calibration was especially difficult to visualize in fraud because the event rate was very low. That is why reliability tables and scalar diagnostics matter alongside plots.

## Quantiles and prediction intervals

For regression, the predictive distribution can be summarized by quantiles. If:

$$
F_x(y) = \mathbb{P}(Y \leq y \mid X=x,D),
$$

then the \(\alpha\)-quantile is:

$$
Q_\alpha(x)
=
\inf \{y : F_x(y) \geq \alpha\}.
$$

An 80% central prediction interval is:

$$
[Q_{0.1}(x), Q_{0.9}(x)].
$$

Coverage asks whether the interval contains the realized target at the advertised frequency. On held-out data:

$$
\text{coverage}_{80}
=
\frac{1}{m}
\sum_{j=1}^{m}
\mathbf{1}
\{
y_j \in [Q_{0.1}(x_j), Q_{0.9}(x_j)]
\}.
$$

If the interval is well calibrated, this should be near 0.8, allowing for sampling uncertainty.

## Pinball loss

The time-series and quantile discussion introduced pinball loss as a quantile-regression concept. For quantile level \(\alpha\), true value \(y\), and quantile prediction \(q\):

$$
L_\alpha(y,q)
=
(\alpha - \mathbf{1}\{y < q\})(y-q).
$$

This loss penalizes under-prediction and over-prediction asymmetrically, which is necessary for estimating quantiles instead of only means.

## Bootstrap uncertainty

Bootstrap uncertainty estimates variability by resampling from the observed evaluation data. If the evaluation set is:

$$
\{(x_i,y_i)\}_{i=1}^{n},
$$

ordinary bootstrap samples \(n\) rows with replacement and recomputes the metric. Repeating this many times gives an empirical distribution of the metric.

This is useful, but ordinary row bootstrap assumes rows are exchangeable enough for the resampling scheme to make sense.

## Block bootstrap

Finance workflows often violate row-level independence:

- 20-day future volatility labels overlap across adjacent dates.
- Market regimes persist through time.
- Asset-month rows in the same month share the same market environment.
- Top-k labels are defined within a monthly cross-section.

For these cases, the posts use block bootstrap. Instead of resampling individual rows, resample blocks such as calendar months:

$$
\mathcal{B}_1,\mathcal{B}_2,\ldots,\mathcal{B}_M.
$$

A bootstrap sample is formed by sampling blocks with replacement, then recomputing the metric on the rows inside those selected blocks.

This preserves more of the dependence structure than row-level bootstrap.

## Drift and distribution shift

The finance posts use population stability index, or PSI, as a drift diagnostic. PSI compares the distribution of a feature in a reference window and a later window.

If \(p_b\) is the reference fraction in bin \(b\), and \(q_b\) is the holdout fraction in the same bin, then:

$$
PSI
=
\sum_b
(q_b - p_b)
\log\left(\frac{q_b}{p_b}\right).
$$

Large PSI values indicate distribution shift. In the volatility posts, rate, yield-curve, stock-bond correlation, and volatility features drifted substantially between pre-holdout and 2020-forward holdout periods.

The lesson is that model evaluation should be read together with drift. A model can pass holdout evaluation under one regime and still require monitoring when regimes change.

## Statistical humility

The recurring statistical lesson is humility about claims.

If bootstrap intervals overlap heavily, the right conclusion is not "model A definitively wins." It is more accurate to say that model A has the best point estimate in this run, while the uncertainty limits the strength of the claim.

If a calibration curve is compressed by rare-event prevalence, the right response is not to over-read the figure. Use reliability tables and scalar diagnostics.

If a portfolio diagnostic has wide confidence intervals, it should not be turned into a trading claim.

The current standard is:

> Report point estimates, uncertainty, assumptions, and limitations together.
