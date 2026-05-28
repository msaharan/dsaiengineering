[Mohit Saharan](https://linkedin.com/in/msaharan), P31, 20260528
___
# Understanding Tabular Foundation models: the architecture of TabICLv2 - 6

Subtitle: Quantile predictions for regression
___
The previous post covered many-class classification, where TabICLv2 handles large label spaces through mixed-radix and hierarchical structure. This post covers quantile predictions for regression, the regression strategy TabICLv2 uses to model predictive uncertainty without discretizing the target into classification bins.

As a reminder, the architecture of TabICLv2 is illustrated in the following figure. Here, given an input \(X\in\mathbb{R}^{n\times m}\), repeated feature grouping encodes columns into multigroups via circular shifts to break feature symmetries, and target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each feature through a set transformer, \(\text{TF}_\text{row}\) aggregates features into row representations \(h\), and  \(\text{TF}_\text{icl}\) performs in-context learning to predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmax) is applied in part of  \(\text{TF}_\text{col}\), where inducing points aggregate input information, and in \(\text{TF}_\text{icl}\) to mitigate attention fading and improve long-context generalization. 

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-6.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Quantile predictions for regression

Tabular foundation models adopt different strategies for regression. TabPFNv2 and TabPFN-2.5 model the full predictive distribution by discretizing the target space into bins and applying cross-entropy loss. TabICLv2 trains separate models for classification and regression.

For regression, TabICLv2 predicts quantiles. Let \(Y\) be a real-valued target random variable. For a probability level \(\alpha\in(0,1)\), an \(\alpha\)-quantile is any value \(q_\alpha\) such that
$$
P(Y\leq q_\alpha)\geq \alpha
\quad\text{and}\quad
P(Y\geq q_\alpha)\geq 1-\alpha,
$$
where \(P(\cdot)\) denotes probability. If the distribution is continuous and strictly increasing at \(q_\alpha\), this reduces to
$$
F_Y(q_\alpha)=\alpha,
$$
where \(F_Y\) is the cumulative distribution function (CDF). For example, \(q_{0.5}\) is the median, \(q_{0.9}\) is the 90th percentile, and \(q_{0.1}\) is the 10th percentile.

In supervised regression, the target distribution depends on the input \(x\). The conditional \(\alpha\)-quantile is
$$
q_\alpha(x)=Q_x(\alpha),
\qquad
Q_x(\alpha)=\inf\{q\in\mathbb{R}:F_{Y\mid X=x}(q)\geq\alpha\}.
$$
Here \(F_{Y\mid X=x}\) is the conditional CDF of \(Y\) given \(X=x\), and \(Q_x\) is its generalized inverse. When the conditional CDF is continuous and strictly increasing, this is the usual inverse \(F^{-1}_{Y\mid X=x}(\alpha)\).

TabICLv2 predicts 999 such quantiles at probability levels
$$
\mathcal{A}=\{0.001,0.002,\ldots,0.999\}.
$$
This gives a dense approximation to \(Q_x(\alpha)\), so the model learns more than a single point estimate. It learns many points of the conditional predictive distribution \(Y\mid X=x\).

Each quantile is trained with pinball loss, also called quantile loss. If the model predicts \(\hat{q}_\alpha(x)\) and the observed target is \(y\), define the residual
$$
u=y-\hat{q}_\alpha(x).
$$
The pinball loss is
$$
\rho_\alpha(u)
=
\begin{cases}
\alpha u, & u\geq 0,\\
(\alpha-1)u, & u<0.
\end{cases}
$$
Equivalently,
$$
\rho_\alpha(y-\hat{q})
=
(\alpha-\mathbf{1}\{y<\hat{q}\})(y-\hat{q}).
$$
Here \(\mathbf{1}\{y<\hat{q}\}\) is an indicator that equals \(1\) when \(y<\hat{q}\) and \(0\) otherwise. The loss is shaped like a tilted absolute-value function. Underprediction means \(y>\hat{q}\), so \(u>0\), and the penalty slope with respect to the residual \(u\) is \(\alpha\). Overprediction means \(y<\hat{q}\), so \(u<0\), and the penalty slope magnitude is \(1-\alpha\).

This asymmetry is what makes the loss target a specific quantile. For \(\alpha=0.5\),
$$
\rho_{0.5}(u)=0.5|u|,
$$
so minimizing it recovers the median. For \(\alpha=0.9\), overprediction is penalized with slope \(0.1\), while underprediction is penalized with slope \(0.9\). The model therefore learns to place \(\hat{q}_{0.9}\) high enough that about 90% of outcomes fall below it.

The quantile property can be shown directly. For a fixed input \(x\), suppress \(x\) in the notation and consider choosing a scalar \(q\) to minimize the expected pinball risk
$$
R_\alpha(q)=\mathbb{E}[\rho_\alpha(Y-q)].
$$
Assuming for exposition that \(Y\) has a continuous distribution, so \(P(Y=q)=0\), the derivative is
$$
\frac{dR_\alpha(q)}{dq}
=
P(Y<q)-\alpha.
$$
Setting this to zero gives
$$
P(Y<q)=\alpha,
$$
which is precisely the \(\alpha\)-quantile condition for a continuous distribution. More generally, when the distribution has atoms or flat regions, the minimizers are values satisfying
$$
P(Y<q)\leq \alpha \leq P(Y\leq q).
$$
This is the standard quantile interval condition.

TabICLv2 sums this loss over all predicted quantile levels:
$$
\mathcal{L}(x,y)
=
\sum_{\alpha\in\mathcal{A}}
\rho_\alpha\left(y-\hat{q}_\alpha(x)\right),
\qquad
\mathcal{A}=\{0.001,0.002,\ldots,0.999\}.
$$

These quantiles should be monotone in \(\alpha\). If \(\alpha_1<\alpha_2\), then
$$
\alpha_1<\alpha_2
\quad\Rightarrow\quad
Q_x(\alpha_1)\leq Q_x(\alpha_2).
$$
Neural networks do not automatically guarantee this ordering when each quantile is predicted as a separate output dimension, so predicted quantiles can cross. For probabilistic predictions, TabICLv2 constructs a full distribution from the quantiles by enforcing monotonicity via sorting by default, or isotonic regression (Barlow & Brunk, 1972; Busing, 2022). It then extrapolates beyond the smallest and largest predicted probability levels with parametric exponential tails and derives closed-form PDF, CDF, and moments.

Prediction intervals are a direct use of quantiles. For a chosen error rate \(\gamma\in(0,1)\), a central \((1-\gamma)\) interval is
$$
\left[\hat{q}_{\gamma/2}(x),\ \hat{q}_{1-\gamma/2}(x)\right].
$$
For example, a 90% interval uses \(\gamma=0.1\):
$$
\left[\hat{q}_{0.05}(x),\ \hat{q}_{0.95}(x)\right].
$$
If the predicted quantiles are calibrated, such intervals should contain the true target approximately 90% of the time over repeated samples from the same data-generating process.

For point estimation, TabICLv2 takes the average of the predicted quantiles. This can be interpreted through the identity
$$
\mathbb{E}[Y\mid X=x]=\int_0^1 Q_x(\alpha)\,d\alpha,
$$
when the conditional expectation exists. With a dense, evenly spaced grid of quantiles, the integral can be approximated by an average:
$$
\hat{\mu}(x)
\approx
\frac{1}{|\mathcal{A}|}\sum_{\alpha\in\mathcal{A}}\hat{q}_\alpha(x).
$$
Here \(\hat{\mu}(x)\) is the point prediction and \(|\mathcal{A}|=999\) is the number of predicted quantile levels.
This explains why averaging many predicted quantiles can serve as a fast point estimate while preserving the richer distributional information needed for intervals, CDFs, PDFs, and moments.

## Summary

For regression, TabICLv2 predicts a dense grid of conditional quantiles rather than a single scalar or a discretized target distribution. These quantiles are trained with pinball loss, support point prediction through averaging, and support probabilistic prediction through a reconstructed monotone predictive distribution. This post completes the miniseries on the architecture of TabICLv2. 
