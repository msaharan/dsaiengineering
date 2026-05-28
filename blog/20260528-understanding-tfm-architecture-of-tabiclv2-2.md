# Understanding Tabular Foundation models: the architecture of TabICLv2-2

Source: TabICLv2 paper. https://arxiv.org/pdf/2602.11139.

In the previous post, we covered repeated feature grouping, which gives each feature contextual views of other columns to reduce representation collapse. In this post, we cover target-aware embedding, the step where TabICLv2 injects observed targets into the feature representations of training rows.

## Illustration and summary

The architecture of TabICLv2 is illustrated in the following figure. Here, given an input \(X\in\mathbb{R}^{n\times m}\), repeated feature grouping encodes columns into multigroups via circular shifts to break feature symmetries, and target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each feature through a set transformer, \(\text{TF}_\text{row}\) aggregates features into row representations \(h\), and  \(\text{TF}_\text{icl}\) performs in-context learning tomorrow predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmaxx), is applied in part of  \(\text{TF}_\text{col}\) where inducing points aggregate input information and  \(\text{TF}_\text{icl}\) to mitigate attention fading and improve long-context generalization. 

The following subsections elaborate on the summary.

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-2.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Target-aware embedding

After repeated feature grouping produces input data representations $E_1\in\mathbb{R}^{n\times m \times d}$, target embeddings are added to each training token:
$$
E_2[i,j] = E_1[i,j] + \text{Embed}_\text{TAE}(y_i), \quad i \in \mathcal{D}_\text{train},
$$
where \(\text{Embed}_\text{TAE}\) is a linear layer for regression or a learnable lookup table for classification. Unlike TabPFNv2 appending the target as an additional column, TabICLv2 directly adds target embeddings to all features. This also helps mitigate representation collapse because even when two features share similar distributions, their association with target values often differ across samples.

## Summary

Target-aware embedding turns feature-only training-row representations into feature-target representations. By adding the target embedding to every feature token in a labeled row, TabICLv2 exposes outcome information early, before column-wise and row-wise processing, without increasing the number of feature tokens.

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

### \(E_2\)

After repeated feature grouping, TabICLv2 has a tensor
$$
E_1\in\mathbb{R}^{n\times m\times d},
$$
where \(E_1[i,j]\in\mathbb{R}^d\) is the embedding for row \(i\) and grouped feature position \(j\). The target-aware tensor \(E_2\) has the same shape:
$$
E_2\in\mathbb{R}^{n\times m\times d}.
$$
It is obtained by adding a target embedding to the feature-group embeddings of training rows.

Let
$$
\mathcal{I}_\text{train}\subseteq \{1,\ldots,n\}
$$
be the set of rows whose targets are observed, and define a row mask
$$
M_i=
\begin{cases}
1, & i\in \mathcal{I}_\text{train},\\
0, & i\notin \mathcal{I}_\text{train}.
\end{cases}
$$
Let
$$
e_y(y_i)=\text{Embed}_\text{TAE}(y_i)\in\mathbb{R}^d
$$
be the target embedding. A compact way to write the target-aware representation is
$$
E_2[i,j]=E_1[i,j]+M_i e_y(y_i),
\qquad i=1,\ldots,n,\quad j=1,\ldots,m.
$$
Equivalently, for training rows,
$$
E_2[i,j]=E_1[i,j]+e_y(y_i),
\qquad i\in\mathcal{I}_\text{train},
$$
and for test rows,
$$
E_2[i,j]=E_1[i,j],
\qquad i\notin\mathcal{I}_\text{train},
$$
unless an implementation uses a special unknown-target embedding.

The addition is well-defined because both terms live in \(\mathbb{R}^d\). The target embedding is broadcast over feature groups:
$$
E_2[i,1]-E_1[i,1]
=
E_2[i,2]-E_1[i,2]
=\cdots=
E_2[i,m]-E_1[i,m]
=e_y(y_i)
$$
for every training row \(i\). Thus the row receives one shared label-derived offset, while each feature group keeps its own content through \(E_1[i,j]\).

For classification with \(K\) classes, the target embedding can be written as a lookup table
$$
W_\text{cls}\in\mathbb{R}^{K\times d},
\qquad
e_y(y_i)=W_\text{cls}[y_i].
$$
For regression, a simple linear embedding has the form
$$
e_y(y_i)=a y_i+b,
\qquad a,b\in\mathbb{R}^d,
$$
or, equivalently, a learned affine map from the scalar target into the \(d\)-dimensional feature-token space.

This construction differs from appending the target as another column. Appending would change the token count from \(m\) to \(m+1\). Target-aware addition keeps the feature-token count fixed:
$$
\text{shape}(E_2)=\text{shape}(E_1)=n\times m\times d.
$$
The label information is therefore available at every grouped feature token before the column-wise and row-wise transformer stages, without introducing an extra target column token.

Statistically, \(E_2\) changes the representation of a training example from a feature-only encoding to a feature-target encoding:
$$
E_1[i,\cdot]\approx \phi(x_i),
\qquad
E_2[i,\cdot]\approx \psi(x_i,y_i).
$$
This helps the model learn how feature patterns co-vary with observed outcomes during in-context learning. It also helps with representation collapse: if two rows or feature groups look similar in \(E_1\) but have different targets, then their \(E_2\) representations differ by the target embedding term.

The masking condition is essential. For test rows, \(y_i\) is the quantity to be predicted, so adding \(e_y(y_i)\) would leak the answer. TabICLv2 uses target-aware representations only where labels are observed, then predicts unknown test targets in the later ICL stage from the labeled context.
