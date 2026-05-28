[Mohit Saharan](https://linkedin.com/in/msaharan), P28, 20260528
___
# Understanding Tabular Foundation Models: the architecture of TabICLv2 - 3
Subtitle: Compression then ICL
___
The previous post covered target-aware embedding, where labels are injected into the feature tokens of training rows. This post covers the compression-then-ICL pipeline: first TabICLv2 compresses row-feature tokens into row representations, then it performs in-context learning over those rows.

As a reminder, the architecture of TabICLv2 is illustrated in the following figure. Given an input table \(X\in\mathbb{R}^{n\times m}\), where \(n\) is the number of rows and \(m\) is the number of columns, repeated feature grouping encodes columns into grouped feature positions using circular shifts to reduce feature-order symmetries. Target-aware embedding injects observed target information for training rows. Then \(\text{TF}_\text{col}\) embeds each grouped feature position across rows, \(\text{TF}_\text{row}\) aggregates grouped feature embeddings into row representations \(h_i\), and \(\text{TF}_\text{icl}\) performs in-context learning to predict test targets \(\hat{y}_i\). QASSMax (query-aware scalable softmax) is used inside parts of \(\text{TF}_\text{col}\) and \(\text{TF}_\text{icl}\) to reduce attention fading when the context contains many rows.

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-3.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Compression then ICL

Before compression begins, TabICLv2 has already constructed a target-aware token tensor
$$
E_2\in\mathbb{R}^{n\times m\times d}.
$$
Here \(d\) is the token embedding dimension. Each token \(E_2[i,j]\in\mathbb{R}^d\) represents row \(i\) and grouped feature position \(j\). After repeated feature grouping, \(m\) denotes the number of grouped feature positions being processed by the transformer stack.

This tensor comes from the feature-only tensor
$$
E_1\in\mathbb{R}^{n\times m\times d}
$$
produced by repeated feature grouping. Let \(\mathcal{I}_\text{train}\subseteq\{1,\ldots,n\}\) be the set of training row indices, and let \(\mathcal{I}_\text{test}\) be the complementary set of test row indices whose targets must be predicted. For \(i\in\mathcal{I}_\text{train}\), \(y_i\) is the observed target. Define the row-level target token
$$
u_i =
\begin{cases}
\text{Embed}_\text{TAE}(y_i), & i\in \mathcal{I}_\text{train},\\
0, & i\in \mathcal{I}_\text{test},
\end{cases}
$$
where \(\text{Embed}_\text{TAE}\) maps an observed target into \(\mathbb{R}^d\), and \(0\in\mathbb{R}^d\) is the zero vector. Then each grouped feature token is target-aware:
$$
E_2[i,j]=E_1[i,j]+u_i.
$$
For a labeled row, the same target vector is broadcast across all grouped feature positions:
$$
E_2[i,1]-E_1[i,1]
=
E_2[i,2]-E_1[i,2]
=\cdots=
E_2[i,m]-E_1[i,m]
=\text{Embed}_\text{TAE}(y_i).
$$
So \(E_2\) has the same shape as \(E_1\), but training rows now carry outcome information inside every feature token. This is the first place where targets enter the architecture; a second target embedding is added later at the row-token level before dataset-wise ICL.

The compression-then-ICL pipeline explains what happens next. TabICLv2 must convert the \(n\times m\) grid of row-feature tokens into row-level representations that can be used for prediction. It does this in three stages:

1. column-wise embedding applies a Set Transformer \(\text{TF}_\text{col}\) to each grouped feature position;
2. row-wise interaction uses \(\text{TF}_\text{row}\) with learned \([\text{CLS}]\) tokens to compress grouped feature embeddings within each row;
3. dataset-wise ICL uses \(\text{TF}_\text{icl}\) so test row representations can attend to labeled training row representations.

The first stage, column-wise embedding, processes each grouped feature position across rows. Conceptually, for each grouped feature index \(j\), the model sees the sequence
$$
(E_2[1,j],E_2[2,j],\ldots,E_2[n,j]).
$$
\(\text{TF}_\text{col}\) lets the model compare how the same grouped feature behaves across rows. In the implementation, rows are updated as queries while labeled training rows provide the context keys and values. This keeps the column-wise contextualization anchored to observed examples.

Let \(\tilde{E}\in\mathbb{R}^{n\times m\times d}\) denote the output of the column-wise stage. The second stage, row-wise interaction, aggregates feature information within each row. For a fixed row \(i\), the model has grouped feature embeddings
$$
(\tilde{E}[i,1],\tilde{E}[i,2],\ldots,\tilde{E}[i,m]),
$$
and \(\text{TF}_\text{row}\) processes them together with learned \([\text{CLS}]\) tokens. The outputs at the \([\text{CLS}]\) positions are used as the row summary. If those outputs are concatenated or otherwise merged into one vector, the result is a row representation
$$
h_i\in\mathbb{R}^{d_\text{row}},
$$
where \(d_\text{row}\) is the row-representation dimension. This is the main compression step: the model moves from \(n\times m\) feature tokens to \(n\) row representations.

The third stage is dataset-wise in-context learning. The row representations
$$
h_1,\ldots,h_n
$$
become the row tokens over which \(\text{TF}_\text{icl}\) operates. For training rows, the model adds another target embedding to the row representation; for test rows, no true target is supplied. Define
$$
z_i =
\begin{cases}
\displaystyle h_i+\text{Embed}_\text{ICL}(y_i), & i\in\mathcal{I}_\text{train},\\
h_i, & i\in\mathcal{I}_\text{test},
\end{cases}
$$
where \(z_i\in\mathbb{R}^{d_\text{row}}\) is the row token passed to \(\text{TF}_\text{icl}\), and \(\text{Embed}_\text{ICL}\) maps an observed target into \(\mathbb{R}^{d_\text{row}}\). The ICL transformer lets test rows attend to labeled training rows and outputs \(\hat{y}_i\), the predicted target for test row \(i\).

This staged design is important because it separates two kinds of structure. Column-wise and row-wise processing learn feature and row representations; dataset-wise ICL performs the final train-test interaction. The model avoids doing expensive full cell-level attention throughout the entire pipeline while still giving the prediction stage row-level context enriched by feature and target information.

## Summary

Compression then ICL separates feature processing from dataset-level prediction. TabICLv2 first contextualizes grouped feature positions across rows, then compresses each row into a fixed-dimensional representation, and finally lets test rows attend to labeled training rows through the ICL transformer. The next post covers query-aware scalable softmax, the attention-scaling mechanism TabICLv2 uses to preserve selective attention as the number of context samples grows.
