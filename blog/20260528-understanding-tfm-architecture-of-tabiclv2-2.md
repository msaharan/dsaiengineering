[Mohit Saharan](https://linkedin.com/in/msaharan), P27, 20260528
___
# Understanding Tabular Foundation models: the architecture of TabICLv2 - 2

Subtitle: Target-aware embedding
___
The previous post covered repeated feature grouping, which gives each feature contextual views of other columns to reduce representation collapse. This post covers target-aware embedding, the step where TabICLv2 injects observed targets into the feature representations of training rows.

As a reminder, the architecture of TabICLv2 is illustrated in the following figure. Here, given an input \(X\in\mathbb{R}^{n\times m}\), repeated feature grouping encodes columns into multigroups via circular shifts to break feature symmetries, and target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each feature through a set transformer, \(\text{TF}_\text{row}\) aggregates features into row representations \(h\), and  \(\text{TF}_\text{icl}\) performs in-context learning to predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmax) is applied in part of  \(\text{TF}_\text{col}\), where inducing points aggregate input information, and in \(\text{TF}_\text{icl}\) to mitigate attention fading and improve long-context generalization. 

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-2.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Target-aware embedding

The previous post showed that repeated feature grouping produces a tensor
$$
E_1\in\mathbb{R}^{n\times m\times d},
$$
where \(n\) is the number of rows, \(m\) is the number of grouped feature positions, \(d\) is the embedding dimension, and \(E_1[i,j]\in\mathbb{R}^d\) is the embedding for row \(i\) and grouped feature position \(j\). At this stage, the representation is still feature-only: it encodes the input table, but it has not yet injected the observed targets of the training rows.

Target-aware embedding changes this. It converts \(E_1\) into a target-aware tensor
$$
E_2\in\mathbb{R}^{n\times m\times d}
$$
by adding an embedding of the observed target \(y_i\) to the feature-group embeddings of training rows.

Let
$$
\mathcal{I}_\text{train}\subseteq \{1,\ldots,n\}
$$
be the set of rows whose targets are observed. For \(i\in\mathcal{I}_\text{train}\), \(y_i\) denotes the observed target for row \(i\). Define a row-level target vector
$$
u_i=
\begin{cases}
\text{Embed}_\text{TAE}(y_i), & i\in \mathcal{I}_\text{train},\\
0, & i\notin \mathcal{I}_\text{train},
\end{cases}
$$
where \(0\in\mathbb{R}^d\) is the zero vector. The target-aware representation can then be written compactly without referring to unknown test targets:
$$
E_2[i,j]=E_1[i,j]+u_i,
\qquad i=1,\ldots,n,\quad j=1,\ldots,m.
$$
Equivalently, for training rows,
$$
E_2[i,j]=E_1[i,j]+\text{Embed}_\text{TAE}(y_i),
\qquad i\in\mathcal{I}_\text{train},
$$
and for test rows,
$$
E_2[i,j]=E_1[i,j],
\qquad i\notin\mathcal{I}_\text{train},
$$
so the true target is never inserted for rows whose labels are unknown.

The addition is well-defined because both \(E_1[i,j]\) and \(\text{Embed}_\text{TAE}(y_i)\) live in \(\mathbb{R}^d\). The same target vector is broadcast across all \(m\) grouped feature positions in a training row:
$$
E_2[i,1]-E_1[i,1]
=
E_2[i,2]-E_1[i,2]
=\cdots=
E_2[i,m]-E_1[i,m]
=\text{Embed}_\text{TAE}(y_i).
$$
Thus the row receives one shared label-derived offset, while each feature group keeps its own content through \(E_1[i,j]\).

For classification with \(K\) classes, where \(y_i\in\{0,\ldots,K-1\}\), \(\text{Embed}_\text{TAE}\) can be written as a learnable lookup table
$$
W_\text{cls}\in\mathbb{R}^{K\times d},
\qquad
\text{Embed}_\text{TAE}(y_i)=W_\text{cls}[y_i].
$$
For regression, a simple target embedding is a learned affine map from the scalar target to the \(d\)-dimensional token space:
$$
\text{Embed}_\text{TAE}(y_i)=a y_i+b,
\qquad a,b\in\mathbb{R}^d.
$$
In both cases, the target is converted into the same representation space as the feature tokens so the two can be added.

This design differs from appending the target as another column. Appending would change the number of tokens from \(m\) to \(m+1\). Target-aware addition keeps the shape fixed:
$$
\text{shape}(E_2)=\text{shape}(E_1)=n\times m\times d.
$$
The label information is therefore available at every grouped feature token before the column-wise and row-wise transformer stages, without introducing an extra target column token.

This also connects back to representation collapse, but from a different angle than repeated feature grouping. Let a tabular dataset be represented by random variables
$$
(X_1,\ldots,X_m,Y).
$$
Here \(X_j\) is the \(j\)-th feature and \(Y\) is the target. Two features can have similar marginal distributions,
$$
P_{X_a}\approx P_{X_b},
$$
where \(P_{X_j}\) denotes the marginal distribution of \(X_j\), while having different relationships to the target:
$$
P(Y\mid X_a)\neq P(Y\mid X_b).
$$
Repeated feature grouping helps by adding feature context. Target-aware embedding adds target context. Even when two feature groups look similar in \(E_1\), their training-row representations can differ in \(E_2\) because their associated outcomes differ:
$$
E_2[i,j]-E_1[i,j]=\text{Embed}_\text{TAE}(y_i),
\qquad i\in\mathcal{I}_\text{train}.
$$
Statistically, for training rows, the representation changes from a feature-only encoding to a feature-target encoding:
$$
E_1[i,\cdot]\approx \phi(x_i),
\qquad
E_2[i,\cdot]\approx \psi(x_i,y_i).
$$
Here \(x_i\) denotes the feature vector for row \(i\), while \(\phi\) and \(\psi\) are informal names for feature-only and feature-target representation functions. This helps the downstream transformers learn how feature patterns co-vary with observed outcomes during in-context learning.

The masking condition is essential. For test rows, \(y_i\) is exactly what the model must predict, so adding \(\text{Embed}_\text{TAE}(y_i)\) would leak the answer. TabICLv2 injects target information only where labels are known, then uses those labeled training representations as context for predicting unknown test targets later in the ICL stage.

## Summary

Target-aware embedding turns feature-only training-row representations into feature-target representations. By adding the target embedding to every feature token in a labeled row, TabICLv2 exposes outcome information early, before column-wise and row-wise processing, without increasing the number of feature tokens. The next post covers the compression-then-ICL pipeline, which turns target-aware feature tokens into row representations and then performs in-context learning over those rows.
