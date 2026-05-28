# Understanding Tabular Foundation models: the architecture of TabICLv2-3

Source: TabICLv2 paper. https://arxiv.org/pdf/2602.11139.

In the previous post, we covered target-aware embedding, where labels are injected into the feature tokens of training rows. In this post, we cover the compression-then-ICL pipeline, which turns target-aware feature tokens into row representations and then performs in-context learning over those rows.

## Illustration and summary

The architecture of TabICLv2 is illustrated in the following figure. Here, given an input \(X\in\mathbb{R}^{n\times m}\), repeated feature grouping encodes columns into multigroups via circular shifts to break feature symmetries, and target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each feature through a set transformer, \(\text{TF}_\text{row}\) aggregates features into row representations \(h\), and  \(\text{TF}_\text{icl}\) performs in-context learning tomorrow predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmaxx), is applied in part of  \(\text{TF}_\text{col}\) where inducing points aggregate input information and  \(\text{TF}_\text{icl}\) to mitigate attention fading and improve long-context generalization. 

The following subsections elaborate on the summary.

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-3.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Compression then ICL

TabICLv2 processes \(E_2\) in three stages:

1. column-wise embedding applies a set transformer \(\text{TF}_\text{col}\) (Lee et al., 2019) to each column; 
2. row-wise interaction uses a transformer \(\text{TF}_\text{col}\) with \([\text{CLS}]\) tokens to collapse feature embeddings per row into a single vector;
3. dataset-wise ICL combines row embeddings with target embeddings and ues a transformer \(\text{TF}_\text{icl}\) where test samples attend to training samples for prediction.

## Summary

Compression then ICL separates feature processing from dataset-level prediction. TabICLv2 first embeds columns, then compresses each row into a fixed-dimensional representation, and finally lets test rows attend to labeled training rows through the ICL transformer.

#  Appendix

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
