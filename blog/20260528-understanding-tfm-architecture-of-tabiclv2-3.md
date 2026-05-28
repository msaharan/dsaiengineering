[Mohit Saharan](https://linkedin.com/in/msaharan), P28, 20260528
___
# Understanding Tabular Foundation models: the architecture of TabICLv2 - 3
Subtitle: Compression then ICL
___
The previous post covered target-aware embedding, where labels are injected into the feature tokens of training rows. This post covers the compression-then-ICL pipeline, which turns target-aware feature tokens into row representations and then performs in-context learning over those rows.

As a reminder, the architecture of TabICLv2 is illustrated in the following figure. Here, given an input \(X\in\mathbb{R}^{n\times m}\), repeated feature grouping encodes columns into multigroups via circular shifts to break feature symmetries, and target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each feature through a set transformer, \(\text{TF}_\text{row}\) aggregates features into row representations \(h\), and  \(\text{TF}_\text{icl}\) performs in-context learning to predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmax) is applied in part of  \(\text{TF}_\text{col}\), where inducing points aggregate input information, and in \(\text{TF}_\text{icl}\) to mitigate attention fading and improve long-context generalization. 

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-3.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Compression then ICL

Before compression begins, TabICLv2 has already constructed a target-aware tensor
$$
E_2\in\mathbb{R}^{n\times m\times d}.
$$
Here \(n\) is the number of rows, \(m\) is the number of grouped feature positions, and \(d\) is the embedding dimension. Each token \(E_2[i,j]\in\mathbb{R}^d\) represents row \(i\) and grouped feature position \(j\).

This tensor comes from the feature-only representation
$$
E_1\in\mathbb{R}^{n\times m\times d}
$$
produced by repeated feature grouping. Let \(\mathcal{I}_\text{train}\subseteq\{1,\ldots,n\}\) be the set of rows whose targets are observed. For \(i\in\mathcal{I}_\text{train}\), \(y_i\) denotes the observed target for row \(i\). For these labeled rows, TabICLv2 adds a target embedding to every feature-group token:
$$
E_2[i,j]=E_1[i,j]+\text{Embed}_\text{TAE}(y_i),
\qquad i\in\mathcal{I}_\text{train}.
$$
\(\text{Embed}_\text{TAE}\) is the target-aware embedding function, mapping the observed target into \(\mathbb{R}^d\) so it can be added to the feature token.

For rows whose labels are unknown, the true target cannot be injected:
$$
E_2[i,j]=E_1[i,j],
\qquad i\notin\mathcal{I}_\text{train},
$$
so test targets are not leaked into the representation.

A compact way to write both cases is to define a row-level target vector
$$
u_i=
\begin{cases}
\text{Embed}_\text{TAE}(y_i), & i\in \mathcal{I}_\text{train},\\
0, & i\notin \mathcal{I}_\text{train},
\end{cases}
$$
where \(0\in\mathbb{R}^d\) is the zero vector. Then
$$
E_2[i,j]=E_1[i,j]+u_i.
$$
The same target vector is broadcast across all feature groups in a labeled row:
$$
E_2[i,1]-E_1[i,1]
=
E_2[i,2]-E_1[i,2]
=\cdots=
E_2[i,m]-E_1[i,m]
=\text{Embed}_\text{TAE}(y_i).
$$
So \(E_2\) has the same shape as \(E_1\), but training rows now carry outcome information inside every feature token.

The compression-then-ICL pipeline explains what happens next. TabICLv2 must convert a grid of row-feature tokens into row-level representations that can be used for in-context prediction. It does this in three stages:

1. column-wise embedding applies a set transformer \(\text{TF}_\text{col}\) (Lee et al., 2019) to each grouped feature position; 
2. row-wise interaction uses a transformer \(\text{TF}_\text{row}\) with \([\text{CLS}]\) tokens to collapse feature embeddings per row into a single vector;
3. dataset-wise ICL uses a transformer \(\text{TF}_\text{icl}\) where test samples attend to labeled training samples for prediction.

The first stage, column-wise embedding, processes each feature position across rows. Conceptually, for each grouped feature index \(j\), the model sees the sequence
$$
(E_2[1,j],E_2[2,j],\ldots,E_2[n,j]).
$$
\(\text{TF}_\text{col}\) lets the model compare how the same grouped feature behaves across rows. This is where per-feature information is contextualized across the dataset.

The second stage, row-wise interaction, aggregates feature information within each row. For a fixed row \(i\), the model has feature-group embeddings
$$
(\tilde{E}[i,1],\tilde{E}[i,2],\ldots,\tilde{E}[i,m]),
$$
where \(\tilde{E}\) denotes the output of the column-wise stage. A row-wise transformer with \([\text{CLS}]\) tokens compresses these \(m\) feature-group embeddings into a single row representation:
$$
h_i\in\mathbb{R}^d.
$$
Here \([\text{CLS}]\) denotes a learned summary token whose output is used as the compressed representation \(h_i\) of row \(i\). This is the main compression step: the model moves from \(n\times m\) feature tokens to \(n\) row tokens.

The third stage is dataset-wise in-context learning. The row embeddings
$$
h_1,\ldots,h_n
$$
become the context over which \(\text{TF}_\text{icl}\) operates. Training rows carry label information through the target-aware construction, while test rows do not. The ICL transformer lets test samples attend to training samples and produce predictions \(\hat{y}\), where \(\hat{y}\) denotes the predicted target.

This staged design is important because it separates two kinds of structure. Column-wise and row-wise processing learn feature and row representations; dataset-wise ICL performs the final train-test interaction. The model therefore avoids doing expensive full cell-level attention throughout the entire pipeline while still giving the prediction stage row-level context enriched by feature and target information.

## Summary

Compression then ICL separates feature processing from dataset-level prediction. TabICLv2 first embeds columns, then compresses each row into a fixed-dimensional representation, and finally lets test rows attend to labeled training rows through the ICL transformer. The next post covers query-aware scalable softmax, the attention-scaling mechanism TabICLv2 uses to preserve selective attention as the number of context samples grows.
