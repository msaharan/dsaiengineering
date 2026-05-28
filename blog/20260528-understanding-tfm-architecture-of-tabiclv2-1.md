[Mohit Saharan](https://linkedin.com/in/msaharan), P26, 20260528
___
# Understanding Tabular Foundation Models: the architecture of TabICLv2 - 1

Subtitle: Repeated feature grouping
___

This post starts a six-part miniseries on the architecture of TabICLv2. The goal of the series is to cover the architecture one subsection at a time, so each post can focus on the details needed to understand that component without making a single article too long. The reference for all posts in this miniseries is the TabICLv2 paper: https://arxiv.org/pdf/2602.11139.

The architecture of TabICLv2 is illustrated in the following figure. Given an input table \(X\in\mathbb{R}^{n\times m}\), where \(n\) is the number of rows and \(m\) is the number of features, repeated feature grouping first encodes columns into overlapping groups using circular shifts. Target-aware embedding then injects observed target information early in the network. \(\text{TF}_\text{col}\) embeds each grouped feature position through a set transformer, \(\text{TF}_\text{row}\) aggregates feature positions into row representations \(h\), and \(\text{TF}_\text{icl}\) performs in-context learning to predict test targets \(\hat{y}\). QASSMax, or query-aware scalable softmax, is applied in the part of \(\text{TF}_\text{col}\) where inducing points aggregate input information and in \(\text{TF}_\text{icl}\), where it helps mitigate attention fading in long contexts.

For this post, the important part is the first step: how raw columns are turned into grouped feature representations.

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-1.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

This first post covers repeated feature grouping, the mechanism TabICLv2 uses to give feature representations contextual views of other features while keeping \(m\) effective feature positions.

## Repeated feature grouping

The first architectural issue TabICLv2 addresses is how to represent features in a table. Let a dataset have feature random variables
$$
(X_1,\ldots,X_m)
$$
and a target random variable \(Y\). For a concrete dataset, write \(x_{ij}\) for the value of feature \(j\) in row \(i\), where \(i\in\{1,\ldots,n\}\) and \(j\in\{1,\ldots,m\}\). Write
$$
x_{\cdot j}=(x_{1j},\ldots,x_{nj})
$$
for the full \(j\)-th column.

In tabular data, two features can have similar marginal distributions,
$$
P_{X_a}\approx P_{X_b},
$$
where \(P_{X_j}\) denotes the marginal distribution of feature \(X_j\). Similar marginals do not imply similar predictive roles. For example, `days_since_signup` and `days_since_last_purchase` may both be positive, right-skewed variables, but their relationships to churn can be very different.

One way to express this difference is through the feature-specific conditional relationship with the target:
$$
P(Y\mid X_a)\neq P(Y\mid X_b).
$$
Here the notation is shorthand for two different conditional maps: the map from values of \(X_a\) to the distribution of \(Y\), and the map from values of \(X_b\) to the distribution of \(Y\). In a multivariate table, a feature's role is also shaped by its relationships with the other features. If \(X_{-j}\) denotes all features except \(X_j\), then the relevant context for feature \(j\) is not just \(P_{X_j}\), but how \(X_j\), \(X_{-j}\), and \(Y\) vary together.

This creates a representation problem: before the model can reason over feature interactions, its initial feature embeddings must preserve enough information to tell features apart. This matters because TabICL-style feature embedding can initially process each feature with the same encoder. A simplified way to write such an independent column encoder is
$$
\phi:\mathbb{R}^n\rightarrow\mathbb{R}^d,
$$
where \(d\) is the embedding dimension. The column vector \(x_{\cdot j}\) is mapped to a feature representation
$$
e_j=\phi(x_{\cdot j}).
$$
If \(\phi\) mostly sees each feature through its own values, then two columns with similar empirical distributions may be mapped to similar embeddings:
$$
\|e_a-e_b\|_2 \approx 0
\quad \text{or} \quad
\cos(e_a,e_b)\approx 1.
$$
Here \(\|\cdot\|_2\) is Euclidean distance and \(\cos(e_a,e_b)\) is cosine similarity. This is the representation-collapse problem: distinct features become nearly indistinguishable in representation space even though their semantics, correlations, or target relationships differ.

The problem is not that similar feature distributions are inherently bad. The problem is that a feature's role is not determined only by its marginal distribution \(P_{X_j}\). A feature is also characterized by its joint behavior with other features and with the target. Independent feature embedding can underuse this context.

The same issue can be viewed as a symmetry problem. If two columns \(a\) and \(b\) are processed by the same function \(\phi\) and have similar value distributions, the model has little information with which to break the symmetry
$$
x_{\cdot a}\leftrightarrow x_{\cdot b}.
$$
Downstream attention layers then receive nearly interchangeable tokens. Once this happens early, later layers may need to recover feature identity from weak signals.

TabPFNv2 and TabPFN-2.5 mitigate this collapse by grouping multiple columns into single tokens. Grouping gives each feature token some neighboring-feature context, but it also reduces the number of effective feature tokens, which can discard fine-grained feature information. TabICLv2 proposes repeated feature grouping to keep the contextualization benefit while preserving \(m\) effective feature positions.

For a table with \(m\) columns, TabICLv2 creates \(m\) groups. To make the wraparound indexing explicit, define
$$
\rho(t)=1+((t-1)\bmod m),
$$
so \(\rho(t)\) maps any integer \(t\) back into the column index set \(\{1,\ldots,m\}\). The group anchored at feature \(j\) contains columns
$$
\big(j,\rho(j+1),\rho(j+3)\big).
$$
Equivalently, the offset pattern relative to the anchor is \((0,1,3)\) with circular wraparound.

For row \(i\), define the grouped row input
$$
g_j(i)=\left(x_{i,j},x_{i,\rho(j+1)},x_{i,\rho(j+3)}\right).
$$
The vector \(g_j(i)\in\mathbb{R}^3\) contains three scalar feature values from the same row. Each group is encoded by a shared linear map
$$
\text{Lin}: \mathbb{R}^3\rightarrow\mathbb{R}^d,
$$
producing
$$
E_1[i,j]=\text{Lin}(g_j(i)).
$$
The resulting tensor \(E_1\in\mathbb{R}^{n\times m\times d}\) contains one \(d\)-dimensional embedding for each row \(i\) and each group position \(j\).

Now the representation at position \(j\) is no longer based only on \(x_{ij}\). It is based on a local multifeature context anchored at feature \(j\). When the shifted positions are distinct, as they are for \(m\geq4\), each original feature appears in three group positions: once as the anchor, once with offset \(+1\) from another anchor, and once with offset \(+3\) from another anchor. This is why the method is called repeated feature grouping.

If two features have similar marginal behavior but different relationships with their shifted companion features, the empirical distributions of their grouped inputs can differ. Writing \(\widehat{P}_{g_j}\) for the empirical distribution of the triples \(g_j(i)\) across rows,
$$
\widehat{P}_{g_a}\not\approx \widehat{P}_{g_b}
\quad \text{can lead to} \quad
E_1[\cdot,a]\not\approx E_1[\cdot,b].
$$
This is not a deterministic guarantee, because the learned linear map can still compress information. The point is that the model receives more context with which to distinguish otherwise similar columns.

Beyond preserving the number of positions, the particular offsets also control which feature pairs are seen together. The shift pattern \((0,1,3)\) has a useful combinatorial property: for \(m\geq7\) columns, no unordered pair of columns appears together in more than one group. This gives each feature several contextual views without repeatedly coupling the same feature pairs. For example, feature \(j\) is grouped with different companions across its repeated appearances instead of always being tied to the same neighboring column.

The result is a representation that helps break harmful feature symmetries while preserving \(m\) effective feature positions. Repeated feature grouping is therefore a small input-side change with a specific purpose: add feature context before the later column, row, and dataset-level transformer stages process the table.

## Summary

Repeated feature grouping addresses a core weakness of independently embedding tabular features: columns with similar value distributions can become hard to distinguish. TabICLv2 groups each feature with shifted companion features, giving the model multiple contextual views while keeping the number of effective feature positions unchanged. The next post covers target-aware embedding, the step where TabICLv2 injects observed targets into the feature representations of training rows.
