[Mohit Saharan](https://linkedin.com/in/msaharan), P27, 20260528
___
# Understanding Tabular Foundation Models: the architecture of TabICLv2 - 2

Subtitle: Target-aware embedding
___
The previous post covered repeated feature grouping, which gives each feature contextual views of other columns to reduce representation collapse. This post covers target-aware embedding, the step where TabICLv2 injects observed training targets into the feature-token representations.

As a reminder, the architecture of TabICLv2 is illustrated in the following figure. Given an input table \(X\in\mathbb{R}^{n\times m}\), with \(n\) rows and \(m\) original features, repeated feature grouping encodes columns into multiple feature groups via circular shifts. Target-aware embedding then adds observed target information to the grouped feature tokens for training rows. After that, \(\text{TF}_\text{col}\) embeds each grouped feature position through a set transformer, \(\text{TF}_\text{row}\) aggregates feature tokens into row representations \(h\), and \(\text{TF}_\text{icl}\) performs in-context learning to predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmax) is applied in the part of \(\text{TF}_\text{col}\) where inducing points aggregate input information, and in \(\text{TF}_\text{icl}\), to mitigate attention fading and improve long-context generalization.

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-2.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Target-aware embedding

The previous post showed that repeated feature grouping produces a tensor
$$
E_1\in\mathbb{R}^{n\times m\times d},
$$
where \(d\) is the token embedding dimension. After repeated feature grouping, \(m\) also denotes the number of grouped feature positions; in TabICLv2's default grouping pattern this equals the number of original features. The entry \(E_1[i,j]\in\mathbb{R}^d\) is the token for row \(i\in\{1,\ldots,n\}\) and grouped feature position \(j\in\{1,\ldots,m\}\). At this stage, the representation is still feature-only: it encodes the input table, but not the observed targets of the training rows.

Target-aware embedding changes this. It converts \(E_1\) into a target-aware tensor
$$
E_2\in\mathbb{R}^{n\times m\times d}
$$
by adding an embedding of the observed target \(y_i\) to each grouped feature token in training row \(i\).

Let
$$
\mathcal{I}_\text{train}\subseteq \{1,\ldots,n\}
$$
be the set of rows whose targets are observed. For \(i\in\mathcal{I}_\text{train}\), \(y_i\) denotes the observed target for row \(i\). Define a row-level target vector
$$
u_i=
\begin{cases}
\text{Embed}_\text{TAE}(y_i), & i\in \mathcal{I}_\text{train},\\
\mathbf{0}_d, & i\notin \mathcal{I}_\text{train},
\end{cases}
$$
where \(\text{Embed}_\text{TAE}\) is the target-aware embedding map and \(\mathbf{0}_d\in\mathbb{R}^d\) is the zero vector. The target-aware representation is
$$
E_2[i,j]=E_1[i,j]+u_i,
\qquad i=1,\ldots,n,\quad j=1,\ldots,m.
$$
Equivalently, for a training row,
$$
E_2[i,j]=E_1[i,j]+\text{Embed}_\text{TAE}(y_i),
\qquad i\in\mathcal{I}_\text{train},
$$
while for a test row,
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

For classification with \(K\leq 10\) classes, where \(y_i\in\{0,\ldots,K-1\}\), \(\text{Embed}_\text{TAE}\) can be written as a learnable lookup table
$$
W_\text{cls}\in\mathbb{R}^{10\times d},
\qquad
\text{Embed}_\text{TAE}(y_i)=W_\text{cls}[y_i].
$$
Here \(W_\text{cls}\) stores one \(d\)-dimensional vector for each label supported by the pretrained label encoder. The active task may use only the first \(K\) labels. For more than 10 classes, the paper uses mixed-radix ensembling before the later hierarchical classification stage rather than a single unrestricted lookup table.

For regression, where \(y_i\in\mathbb{R}\), the target embedding is a learned linear layer, which can be written as an affine map from the scalar target to the \(d\)-dimensional token space:
$$
\text{Embed}_\text{TAE}(y_i)=a y_i+b,
\qquad a,b\in\mathbb{R}^d,
$$
where \(a\) and \(b\) are learned vectors. In both cases, the target is converted into the same representation space as the feature tokens so the two can be added.

This design differs from appending the target as another column. Appending would change the number of tokens from \(m\) to \(m+1\). Target-aware addition keeps the shape fixed:
$$
\text{shape}(E_2)=\text{shape}(E_1)=n\times m\times d.
$$
The label information is therefore available at every grouped feature token before the column-wise and row-wise transformer stages, without introducing an extra target column token.

This connects back to representation collapse, but from a different angle than repeated feature grouping. Let a tabular task be represented by random variables
$$
(X_1,\ldots,X_m,Y).
$$
Here \(X_j\) is the \(j\)-th feature and \(Y\) is the target. Two features, say \(X_a\) and \(X_b\), can have similar marginal distributions,
$$
P_{X_a}\approx P_{X_b},
$$
where \(P_{X_j}\) denotes the marginal distribution of feature \(X_j\), while having different relationships to the target:
$$
P(Y\mid X_a=x)\neq P(Y\mid X_b=x)
$$
for suitable values of \(x\) where the two conditionals can be compared. Repeated feature grouping helps by adding feature context: a feature is no longer encoded entirely in isolation. Target-aware embedding adds supervised context: during column-wise processing, feature tokens from training rows carry both feature information and the observed outcome for that row.

The important nuance is that target-aware embedding does not distinguish feature positions within the same row by itself. The same vector \(\text{Embed}_\text{TAE}(y_i)\) is added to every grouped feature token in row \(i\). Its usefulness comes from how the transformer sees many labeled examples. For training rows \(i\) and \(r\) with different targets,
$$
E_2[i,j]-E_1[i,j]=\text{Embed}_\text{TAE}(y_i),
\qquad
E_2[r,j]-E_1[r,j]=\text{Embed}_\text{TAE}(y_r).
$$
Thus the column-wise transformer receives examples of the form "this feature value occurred in a row with this target." Across many rows, that makes feature-target association available earlier than it would be in a purely feature-only embedding.

Informally, for a training row with feature vector \(x_i=(x_{i1},\ldots,x_{im})\), the representation changes from a feature-only encoding to a feature-target encoding:
$$
E_1[i,\cdot]\approx \phi(x_i),
\qquad
E_2[i,\cdot]\approx \psi(x_i,y_i).
$$
Here \(E_1[i,\cdot]\) and \(E_2[i,\cdot]\) denote all grouped feature tokens for row \(i\), while \(\phi\) and \(\psi\) are informal names for feature-only and feature-target representation functions. This helps the downstream transformers learn how feature patterns co-vary with observed outcomes during in-context learning.

The masking condition is essential. For test rows, \(y_i\) is exactly what the model must predict, so adding \(\text{Embed}_\text{TAE}(y_i)\) would leak the answer. TabICLv2 injects target information only where labels are known. This early injection is separate from the later dataset-wise ICL stage: the point here is that labels have already shaped the row representations before that final in-context prediction stage.

## Summary

Target-aware embedding turns feature-only training-row representations into feature-target representations. By adding the target embedding to every feature token in a labeled row, TabICLv2 exposes outcome information early, before column-wise and row-wise processing, without increasing the number of feature tokens. The next post covers the compression-then-ICL pipeline, which turns target-aware feature tokens into row representations and then performs in-context learning over those rows.
