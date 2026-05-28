# Understanding Tabular Foundation models: the architecture of TabICLv2-1

Source: TabICLv2 paper. https://arxiv.org/pdf/2602.11139.

This post starts a six-part miniseries on the architecture of TabICLv2. The goal of the series is to cover the architecture one subsection at a time, so each post can focus on the details needed to understand that component without making a single article too long. In this first post, we cover repeated feature grouping, the mechanism TabICLv2 uses to give each feature contextual views of other features while preserving feature-level resolution.

## Illustration and summary

The architecture of TabICLv2 is illustrated in the following figure. Here, given an input \(X\in\mathbb{R}^{n\times m}\), repeated feature grouping encodes columns into multigroups via circular shifts to break feature symmetries, and target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each feature through a set transformer, \(\text{TF}_\text{row}\) aggregates features into row representations \(h\), and  \(\text{TF}_\text{icl}\) performs in-context learning tomorrow predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmaxx), is applied in part of  \(\text{TF}_\text{col}\) where inducing points aggregate input information and  \(\text{TF}_\text{icl}\) to mitigate attention fading and improve long-context generalization. 

The following subsections elaborate on the summary.

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-1.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Repeated feature grouping

TabICL embeds each feature independently, which can lead to representation collapse when features share similar distributions. TabPFNv2 and TabPFN-2.5 mitigate this collapse by grouping multiple columns into single tokens, which also reduces the number of effective features to improve efficiency, but this reduction may lose fine-grained feature information. TabICLv2 proposes repeated feature grouping, which places each feature into multiple groups via circular shifts while preserving the number of effective features. Specifically, for a table with \(m\) columns, it creates \(m\) groups where the \(j\)-th group contains columns at positions \((j, j+1, j+3)\,\text{mod}\,m \). Each group is encoded by a shared linear \(\text{Lin}: \mathbb{R}^3\rightarrow\mathbb{R}^d\):
$$
E_1[i, j] = \text{Lin}(x_{i,j}, x_{i, (j+1) \,\text{mod}\,m},x_{i, (j+3)\,\text{mod}\,m}).
$$
The shift pattern \(0, 1, 3\) ensures that for \(\geq7\) columns, no pair of columns appears together in more than one group. 

## Summary

Repeated feature grouping addresses a core weakness of independently embedding tabular features: columns with similar value distributions can become hard to distinguish. TabICLv2 groups each feature with shifted companion features, giving the model multiple contextual views while keeping the number of effective feature positions unchanged.

#  Appendix

## Representation collapse

Let a tabular dataset be represented by random variables
$$
(X_1,\ldots,X_m,Y),
$$
where \(X_j\) is the \(j\)-th feature and \(Y\) is the target. Two features can have similar marginal distributions,
$$
P_{X_a}\approx P_{X_b},
$$
while having different predictive roles:
$$
P(Y\mid X_a)\neq P(Y\mid X_b),
$$
or more generally,
$$
P(Y\mid X_a,X_{-a})\neq P(Y\mid X_b,X_{-b}).
$$
This is common in tabular data. For example, `days_since_signup` and `days_since_last_purchase` may both be positive, right-skewed variables, but their relationship to churn can be very different.

Suppose a model embeds each feature independently using a shared encoder
$$
\phi:\mathbb{R}^n\rightarrow\mathbb{R}^d,
$$
where the column vector \(x_{\cdot j}=(x_{1j},\ldots,x_{nj})\) is mapped to a feature representation
$$
e_j=\phi(x_{\cdot j}).
$$
If \(\phi\) receives only the values of one feature at a time, and if two columns have similar empirical distributions, then it may produce similar embeddings:
$$
\|e_a-e_b\|_2 \approx 0
\quad \text{or} \quad
\cos(e_a,e_b)\approx 1.
$$
Representation collapse refers to this failure mode: distinct features become nearly indistinguishable in representation space even though their semantics or target relationships differ.

The issue is not that similar distributions are inherently bad. The issue is that marginal distribution is insufficient to identify a feature's role. A feature is characterized not only by \(P_{X_j}\), but also by its relationships to other features and to the target. A more complete statistical object is the joint behavior
$$
P(X_j,X_{-j},Y),
$$
or at least target-relevant summaries such as \(P(Y\mid X_j)\). Independent feature embedding can underuse this context.

There is also a symmetry perspective. If two columns \(a\) and \(b\) are processed by the same function \(\phi\) and have similar value distributions, the model has little information with which to break the symmetry
$$
x_{\cdot a}\leftrightarrow x_{\cdot b}.
$$
Downstream attention layers then receive nearly interchangeable tokens. Once this happens early, later layers may need to recover feature identity from weak signals, which is difficult.

Repeated feature grouping reduces this risk by embedding a feature together with other features. In TabICLv2, a group around feature \(j\) has the form
$$
g_j(i)=\left(x_{i,j},x_{i,(j+1)\bmod m},x_{i,(j+3)\bmod m}\right),
$$
and the first representation is
$$
E_1[i,j]=\text{Lin}(g_j(i)).
$$
So feature \(j\) is no longer represented only through its own marginal values. It is represented through local multifeature contexts. If two features have similar \(P_{X_j}\) but different relationships with neighboring or shifted companion features, their grouped representations can separate:
$$
P_{g_a}\not\approx P_{g_b}
\quad \Rightarrow \quad
E_1[\cdot,a]\not\approx E_1[\cdot,b].
$$

TabICLv2's shift pattern gives each feature multiple contextual views while preserving \(m\) effective feature positions. This is different from simply merging many columns into fewer tokens. The objective is to break harmful feature symmetries without discarding fine-grained feature-level information.
