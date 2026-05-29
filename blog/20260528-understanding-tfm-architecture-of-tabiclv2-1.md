[Mohit Saharan](https://linkedin.com/in/msaharan), P26, 20260528, Draft
___
# Understanding Tabular Foundation Models: the architecture of TabICLv2 - 1

Subtitle: Repeated feature grouping
___

Two columns can look almost the same in isolation but play very different predictive roles. TabICLv2's first architectural step addresses that *before* any large transformer runs.

**What to watch for in this post**

- Why similar columns confuse independent encoders
- How circular-shift grouping adds local context
- Why TabICLv2 keeps \(m\) feature positions (unlike TabPFN grouping)

This post starts a six-part miniseries on the architecture of TabICLv2. The goal is to cover the architecture one subsection at a time, so each post can focus on the details needed to understand that component without making a single article too long. The reference for all posts in this miniseries is the [TabICLv2 paper (arXiv)](https://arxiv.org/pdf/2602.11139). This post covers **only** repeated feature grouping.

![TabICLv2 pipeline; this post covers only repeated feature grouping (leftmost block).](./20260528-understanding-tfm-architecture-of-tabiclv2-1.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

*TabICLv2 pipeline; this post covers only repeated feature grouping (leftmost block).*

Later posts cover target-aware embedding, column/row transformers, QASSMax, and the prediction heads.

**Hands-on companion:** I will use the local [NanoTabICL implementation](../../nanotabicl/model.py) as the code companion for this miniseries. It lives in `nanotabicl/`, with the architecture concentrated in `nanotabicl/model.py`. NanoTabICL is not the full production TabICLv2 repository; it is a compact implementation that makes the main architectural ideas easier to read in code.

## Repeated feature grouping

### The problem: similar columns, different roles

TabICLv2 first asks how to represent features in a table. Let a dataset have feature random variables
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
where \(P_{X_j}\) denotes the marginal distribution of feature \(X_j\). *In words: same shape of values across rows—not necessarily the same predictive role.* Similar marginals do not imply similar predictive roles. For example, `days_since_signup` and `days_since_last_purchase` may both be positive, right-skewed variables, but their relationships to churn can be very different.

One way to express this difference is through the feature-specific conditional relationship with the target:
$$
P(Y\mid X_a)\neq P(Y\mid X_b).
$$
Here the notation is shorthand for two different conditional maps. One map sends values of \(X_a\) to the distribution of \(Y\); the other sends values of \(X_b\) to the distribution of \(Y\). In a multivariate table, a feature's role also depends on how it co-varies with other columns and the target. Write \(X_{-j}\) for all features except \(X_j\). The relevant context for feature \(j\) is not just \(P_{X_j}\), but how \(X_j\), \(X_{-j}\), and \(Y\) vary together.

### Why independent column embedding collapses

This creates a representation problem. Before the model can reason over feature interactions, its initial feature embeddings must preserve enough information to tell features apart. TabICL-style feature embedding can initially process each feature with the same encoder. A simplified way to write such an independent column encoder is
$$
\phi:\mathbb{R}^n\rightarrow\mathbb{R}^d,
$$
where \(d\) is the embedding dimension. The column vector \(x_{\cdot j}\) is mapped to a feature representation
$$
e_j=\phi(x_{\cdot j}).
$$
*In words: each column is embedded on its own, without looking at neighboring columns.*

If \(\phi\) mostly sees each feature through its own values, then two columns with similar empirical distributions may be mapped to similar embeddings:
$$
\|e_a-e_b\|_2 \approx 0
\quad \text{or} \quad
\cos(e_a,e_b)\approx 1.
$$
*In words: two distinct features can land on nearly identical embeddings.*

Here \(\|\cdot\|_2\) is Euclidean distance and \(\cos(e_a,e_b)\) is cosine similarity. This is the representation-collapse problem: distinct features become nearly indistinguishable in representation space even though their semantics, correlations, or target relationships differ.

Similar marginals aren't the bug. The bug is treating a column as if its marginal distribution were its entire identity. A feature is also characterized by its joint behavior with other features and with the target. Independent feature embedding can underuse this context.

The same issue is a symmetry problem. Two columns \(a\) and \(b\) share the same encoder \(\phi\) and similar value distributions, so the model has little information with which to break the symmetry
$$
x_{\cdot a}\leftrightarrow x_{\cdot b}.
$$
Downstream attention layers then receive nearly interchangeable tokens. Once that happens early, later layers must recover feature identity from weak signals.

TabPFNv2 and TabPFN-2.5 mitigate this collapse by grouping multiple columns into single tokens. Grouping gives each feature token some neighboring-feature context, but it also reduces the number of effective feature tokens, which can discard fine-grained feature information. TabICLv2 proposes repeated feature grouping to keep the contextualization benefit while preserving \(m\) effective feature positions.

### TabICLv2's fix: local groups with circular shifts

For a table with \(m\) columns, TabICLv2 creates \(m\) groups. To make the wraparound indexing explicit, define
$$
\rho(t)=1+((t-1)\bmod m),
$$
so \(\rho(t)\) maps any integer \(t\) back into the column index set \(\{1,\ldots,m\}\). The group anchored at feature \(j\) contains columns
$$
\big(j,\rho(j+1),\rho(j+3)\big).
$$
Equivalently, the offset pattern relative to the anchor is \((0,1,3)\) with circular wraparound.

For example, with \(m=5\) columns:

| Group anchor \(j\) | Columns in group |
|---|---|
| 1 | (1, 2, 4) |
| 2 | (2, 3, 5) |
| 3 | (3, 4, 1) |
| 4 | (4, 5, 2) |
| 5 | (5, 1, 3) |

> Feature 1 appears as anchor in group 1, as offset \(+1\) in group 5, and as offset \(+3\) in group 3.

For row \(i\), define the grouped row input
$$
g_j(i)=\left(x_{i,j},x_{i,\rho(j+1)},x_{i,\rho(j+3)}\right).
$$
*In words: for row \(i\), group \(j\) takes three values from that row—column \(j\), \(j+1\), and \(j+3\), with wraparound.*

For example, \(g_1(i)=(x_{i,1},x_{i,2},x_{i,4})\). The vector \(g_j(i)\in\mathbb{R}^3\) contains three scalar feature values from the same row. Each group is encoded by a shared linear map
$$
\text{Lin}: \mathbb{R}^3\rightarrow\mathbb{R}^d,
$$
producing
$$
E_1[i,j]=\text{Lin}(g_j(i)).
$$
*In words: one shared linear map turns each 3-value group into a \(d\)-dimensional token.*

The resulting tensor \(E_1\in\mathbb{R}^{n\times m\times d}\) contains one \(d\)-dimensional embedding for each row \(i\) and each group position \(j\).

### Why the (0, 1, 3) pattern matters

The representation at position \(j\) is no longer based only on \(x_{ij}\). It is based on a local multifeature context anchored at feature \(j\). When the shifted positions are distinct, as they are for \(m\geq4\), each original feature appears in three group positions: once as the anchor, once with offset \(+1\) from another anchor, and once with offset \(+3\) from another anchor. This is why the method is called repeated feature grouping.

If two features have similar marginal behavior but different relationships with their shifted companion features, the empirical distributions of their grouped inputs can differ. Writing \(\widehat{P}_{g_j}\) for the empirical distribution of the triples \(g_j(i)\) across rows,
$$
\widehat{P}_{g_a}\not\approx \widehat{P}_{g_b}
\quad \text{can lead to} \quad
E_1[\cdot,a]\not\approx E_1[\cdot,b].
$$
This is not a deterministic guarantee, because the learned linear map can still compress information. The point is that the model receives more context with which to distinguish otherwise similar columns.

The offset pattern also controls **which feature pairs co-occur** in a group. With the shift pattern \((0,1,3)\), for **\(\geq 7\)** columns no pair of columns appears together in more than one group. This gives each feature several contextual views without repeatedly coupling the same feature pairs. For example, feature \(j\) is grouped with different companions across its repeated appearances instead of always being tied to the same neighboring column.

The result is a representation that helps break harmful feature symmetries while preserving \(m\) effective feature positions. Repeated feature grouping is therefore a small input-side change with a specific purpose: add feature context before the later column, row, and dataset-level transformer stages process the table.

### Implementation in NanoTabICL

In NanoTabICL, repeated feature grouping happens at the start of `NanoTabICLv2.forward`. The relevant model parameters are set during initialization:

```python
self.feature_group_size = feature_group_size
self.x_embed = nn.Linear(feature_group_size, embed_dim)
```

The default `feature_group_size` is 3, so `self.x_embed` is a shared linear map from a 3-value feature group into the token dimension. This corresponds to the mathematical map
$$
\text{Lin}: \mathbb{R}^3\rightarrow\mathbb{R}^d.
$$

The grouping itself is implemented by indexing shifted versions of the column axis:

```python
idxs = torch.arange(n_cols, dtype=torch.long, device=x.device)
x = torch.stack(
    [x[:, :, (idxs + (2 ** i - 1)) % n_cols]
     for i in range(self.feature_group_size)],
    dim=-1,
)
emb = self.x_embed(x)
```

The expression `(2 ** i - 1)` is the code version of the offset pattern. With `feature_group_size=3`, the loop uses:

| `i` | `(2 ** i - 1)` | Offset |
|---:|---:|---:|
| 0 | 0 | anchor column \(j\) |
| 1 | 1 | shifted column \(j+1\) |
| 2 | 3 | shifted column \(j+3\) |

The modulo operation `% n_cols` is the circular wraparound. If `idxs = [0, 1, 2, 3, 4]`, then the offset `3` gives `[3, 4, 0, 1, 2]`, so the last columns wrap back to the beginning.

The shape transition is the main thing to notice:

| Step | Shape | Meaning |
|---|---|---|
| input `x` | `(batch, rows, cols)` | one scalar per row and original feature |
| after `torch.stack(..., dim=-1)` | `(batch, rows, cols, 3)` | each feature position now holds a 3-column group |
| after `self.x_embed(x)` | `(batch, rows, cols, embed_dim)` | each group is a learned token |

So NanoTabICL keeps the same number of column positions, `cols`, but each position has already looked at a small circular group of neighboring columns. That is the implementation counterpart of preserving \(m\) effective feature slots while giving each slot local feature context.

#### Note on implementations

The TabICLv2 paper and NanoTabICL write the default offsets as \((0,1,3)\), implemented in NanoTabICL as `(idxs + (2**i - 1)) % n_cols`. The official `tabicl` repository uses the same circular family with a shifted anchor convention, stacking columns as `(idxs + 2**i) % m` for `i=0,1,2` (offsets \(1,2,4\) in column-index terms). The two patterns produce the same multiset of feature triples up to relabeling which output slot is called the anchor.

## Summary

**Takeaway:** Repeated feature grouping keeps \(m\) feature slots, but each slot sees a small neighborhood of columns.

Repeated feature grouping addresses a core weakness of independently embedding tabular features: columns with similar value distributions can become hard to distinguish. TabICLv2 groups each feature with shifted companion features, giving the model multiple contextual views while keeping the number of effective feature positions unchanged. The next post covers target-aware embedding, the step where TabICLv2 injects observed targets into the feature representations of training rows.
