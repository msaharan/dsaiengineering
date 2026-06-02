[Mohit Saharan](https://linkedin.com/in/msaharan), P27, 20260602, Draft

___
# Architecture of TabICLv2: target-aware embedding

Subtitle: 
___
This is the second post in the six-part miniseries on the architecture of TabICLv2. The following figure illustrates the architecture of TabICLv2.

![TabICLv2 architecture; this post covers target-aware embedding.](./20260602-understanding-tfm-architecture-of-tabiclv2-2.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

In the last post, TabICLv2 learned to tell similar-looking features apart by grouping each column with shifted neighbors. In this post, we add one more ingredient to training rows only: the observed target.

## Target-aware embedding

### Starting point: feature-only tokens \(E_1\)

The previous post showed that repeated feature grouping produces a tensor
$$
E_1\in\mathbb{R}^{n\times m\times d},
$$
where \(d\) is the token embedding dimension. After repeated feature grouping, \(m\) also denotes the number of grouped feature positions; in TabICLv2's default grouping pattern this equals the number of original features. The entry \(E_1[i,j]\in\mathbb{R}^d\) is the token for row \(i\in\{1,\ldots,n\}\) and grouped feature position \(j\in\{1,\ldots,m\}\). At this stage, the representation is still feature-only: it encodes the input table, but not the observed targets of the training rows.

### The operation: add a target embedding to every training-row token

Target-aware embedding changes this. It converts \(E_1\) into a target-aware tensor
$$
E_2\in\mathbb{R}^{n\times m\times d}
$$
by adding an embedding of the observed target \(y_i\) to each grouped feature token in training row \(i\). Operationally, target-aware embedding is vector addition in the same embedding space. To write the operation mathematically, let

$$
\mathcal{I}_\text{train}\subseteq \{1,\ldots,n\}
$$
be the set of rows whose targets are observed, and let
$$
\mathcal{I}_\text{test}=\{1,\ldots,n\}\setminus\mathcal{I}_\text{train}
$$
be the complementary set of test rows whose targets must be predicted. For \(i\in\mathcal{I}_\text{train}\), \(y_i\) denotes the observed target for row \(i\). Let us define a row-level target vector
$$
u_i=
\begin{cases}
\text{Embed}_\text{TAE}(y_i), & i\in \mathcal{I}_\text{train},\\
\mathbf{0}_d, & i\notin \mathcal{I}_\text{train},
\end{cases}
$$
where \(\text{Embed}_\text{TAE} \in\mathbb{R}^d \) is the target-aware embedding map and \(\mathbf{0}_d\in\mathbb{R}^d\) is the zero vector. Now, the target-aware representation is
$$
E_2[i,j]=E_1[i,j]+u_i,
\qquad i=1,\ldots,n,\quad j=1,\ldots,m,
$$
where for a training row,
$$
E_2[i,j]=E_1[i,j]+\text{Embed}_\text{TAE}(y_i),
\qquad i\in\mathcal{I}_\text{train},
$$
while for a test row,
$$
E_2[i,j]=E_1[i,j],
\qquad i\in\mathcal{I}_\text{test}.
$$
For each training row \(i\), TabICLv2 computes one target vector and adds it to every grouped feature token in that row. Test rows get zero instead. This masking condition is essential. For test rows, \(y_i\) is exactly what the model must predict, so adding \(\text{Embed}_\text{TAE}(y_i)\) would leak the answer. TabICLv2 injects target information only where labels are known.

### Classification vs regression implementations

The embedding map depends on the prediction task. For classification, it maps discrete class labels to label vectors; for regression, it maps a scalar target to the token space.

For classification with \(K\leq 10\) classes, where \(y_i\in\{0,\ldots,K-1\}\), \(\text{Embed}_\text{TAE}\) is a learned class-embedding interface. Mathematically, it can be written as a lookup table
$$
W_\text{cls}\in\mathbb{R}^{10\times d},
\qquad
\text{Embed}_\text{TAE}(y_i)=W_\text{cls}[y_i].
$$
Here \(W_\text{cls}\) stores one \(d\)-dimensional vector for each label supported by the pretrained label encoder. The <u>official implementation</u> realizes this through a one-hot-plus-linear layer, which is equivalent to selecting a learned class vector. The active task may use only the first \(K\) labels. Tasks with more than 10 classes use an additional label-handling step in TabICLv2 (outside this post's \(K\leq 10\) view).

For regression, where \(y_i\in\mathbb{R}\), the target embedding is a learned linear layer, which can be written as an affine map from the scalar target to the \(d\)-dimensional token space:
$$
\text{Embed}_\text{TAE}(y_i)=a y_i+b,
\qquad a,b\in\mathbb{R}^d,
$$
where \(a\) and \(b\) are learned vectors. In both cases, the target is converted into the same representation space as the feature tokens so the two can be added.

### Why add to tokens instead of appending a target column?

This design differs from appending the target as another column. Appending would change the number of tokens from \(m\) to \(m+1\). Target-aware addition keeps the shape fixed:
$$
\text{shape}(E_2)=\text{shape}(E_1)=n\times m\times d.
$$
The label information is therefore available at every grouped feature token before the column-wise and row-wise transformer stages, without introducing an extra target column token.

### Why this helps before the transformers run

Keeping the shape fixed is the computational benefit. The representational benefit connects back to representation collapse, but from a different angle than repeated feature grouping. As mentioned in the previous post (P26), two features, say \(X_a\) and \(X_b\), can have similar marginal distributions,
$$
P_{X_a}\approx P_{X_b},
$$
where \(P_{X_j}\) denotes the marginal distribution of feature \(X_j\), while having different relationships to the target:
$$
P(Y\mid X_a=x)\neq P(Y\mid X_b=x)
$$
even when both features take similar values. Repeated feature grouping helps by adding feature context: a feature is no longer encoded entirely in isolation. Target-aware embedding adds supervised context: during column-wise processing, feature tokens from training rows carry both feature information and the observed outcome for that row.

### What this does not do by itself

The important nuance is that target-aware embedding does not distinguish feature positions within the same row by itself. The same vector \(\text{Embed}_\text{TAE}(y_i)\) is added to every grouped feature token in row \(i\). This helps because the column-wise transformer later sees many rows labeled with different targets. For training rows \(i\) and \(r\) with different targets,
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
Here \(E_1[i,\cdot]\) and \(E_2[i,\cdot]\) denote all grouped feature tokens for row \(i\), while \(\phi\) and \(\psi\) are informal names for feature-only and feature-target representation functions. The feature-target encoding applies only to training rows; test rows still carry \(E_1\) only. This is the first target injection; TabICLv2 injects targets again later during dataset-wise ICL.

### Implementation in NanoTabICL

NanoTabICL implements target-aware embedding with two pieces: a task-dependent target embedder and one masked in-place addition to the training rows.

During initialization, the target embedder depends on whether the model is configured for classification or regression:

```python
self.y_embed_in = (
    ClassEmbedding(max_classes, embed_dim)
    if max_classes > 0
    else nn.Linear(1, embed_dim)
)
```

For classification, `ClassEmbedding` is a learnable lookup table:

```python
class ClassEmbedding(nn.Embedding):
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return super().forward(y.squeeze(-1).long())
```

The call to `long()` is the practical detail that turns labels such as `0`, `1`, or `2` into embedding-table indices. For regression, `nn.Linear(1, embed_dim)` implements the affine scalar-to-vector map \(a y_i+b\).

The actual target-aware update is one line in `forward`:

```python
emb[:, :n_train] += self.y_embed_in(y[:, :, None, None])
```

Before this line, `emb` has shape:

```text
(batch, rows, cols, embed_dim)
```

The slice `emb[:, :n_train]` selects only labeled context rows:

```text
(batch, n_train, cols, embed_dim)
```

The target tensor `y` starts as:

```text
(batch, n_train)
```

After `y[:, :, None, None]`, it has singleton feature and scalar dimensions:

```text
(batch, n_train, 1, 1)
```

The embedder maps this to a target vector:

```text
(batch, n_train, 1, embed_dim)
```

PyTorch broadcasting then adds the same target vector across all `cols` grouped feature positions in that training row. This is exactly the code version of:
$$
E_2[i,j]=E_1[i,j]+\text{Embed}_\text{TAE}(y_i),
\qquad i\in\mathcal{I}_\text{train}.
$$

The masking boundary is the important safety rule. NanoTabICL never indexes `emb[:, n_train:]` in this addition, so test rows remain feature-only at this stage:

```text
training rows: feature token + target embedding
test rows:     feature token only
```

That single slice, `:n_train`, is what prevents target leakage in the implementation.

## Summary

Target-aware embedding turns feature-only training-row representations into feature-target representations. By adding the target embedding to every feature token in a labeled row, TabICLv2 exposes outcome information early, before column-wise and row-wise processing, without increasing the number of feature tokens. The next post covers the compression-then-ICL pipeline, which turns target-aware feature tokens into row representations and then performs in-context learning over those rows.
