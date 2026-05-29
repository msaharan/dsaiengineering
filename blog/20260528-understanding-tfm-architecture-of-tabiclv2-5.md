[Mohit Saharan](https://linkedin.com/in/msaharan), P30, 20260528, Draft
___
# Understanding Tabular Foundation Models: the architecture of TabICLv2 - 5

Subtitle: Many-class classification
___
The previous post covered query-aware scalable softmax, which improves attention behavior when the context grows. This post covers many-class classification, where TabICLv2 extends a model pretrained with at most 10 classes to settings with many more labels.

**What to watch for in this post**

- Why a flat \(C\)-way classifier mismatches pretraining (\(C \gg 10\))
- Hierarchical classification: taxonomy intuition, tree, chain rule, inference steps
- Building the tree for \(C=57\) (same example throughout)
- Mixed-radix ensembling (MRE): digits, balanced bases, averaging \(\text{TF}_\text{col}\) runs
- How MRE (input / \(\text{TF}_\text{col}\)) and hierarchy (output / \(\text{TF}_\text{icl}\)) fit together

As a reminder, the full pipeline is below. **In this episode, focus on target-aware embedding** (how context labels are represented) **and \(\text{TF}_\text{icl}\)** (how test labels are predicted) when \(C \gg 10\). Later posts cover quantile predictions for regression. Given an input table \(X\in\mathbb{R}^{n\times m}\), repeated feature grouping and target-aware embedding prepare grouped feature tokens, \(\text{TF}_\text{col}\) and \(\text{TF}_\text{row}\) compress them into row representations \(h_i\), and \(\text{TF}_\text{icl}\) predicts test targets \(\hat{y}_i\). When \(C>10\), mixed-radix ensembling adapts label embeddings and column processing; hierarchical classification adapts the prediction step.

![TabICLv2 pipeline; many-class classification affects TAE and TF_icl.](./20260528-understanding-tfm-architecture-of-tabiclv2-5.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

*TabICLv2 pipeline; many-class classification affects target-aware embedding and in-context learning.*

## Many-class classification

TabICLv2 uses two related ideas to handle many classes. **Hierarchical classification** (in \(\text{TF}_\text{icl}\)) turns one large prediction problem into several smaller native prediction problems. **Mixed-radix ensembling** (before ICL, in target-aware embedding and \(\text{TF}_\text{col}\)) makes context labels compatible with more than 10 possible values. I start with the hierarchy because the many-class bottleneck is easiest to see at the output layer first; mixed-radix ensembling then solves the analogous label-embedding bottleneck on the input side.

### The bottleneck: more than 10 classes

Like many tabular foundation models, TabICLv2 is pretrained with classification tasks that have at most 10 classes. Let \(C\) be the number of downstream classes, \(x\) be the row or row representation being classified, and \(y\in\{0,\ldots,C-1\}\) be the true class label. A direct \(C\)-class classifier would predict
$$
p(y=c\mid x)=\frac{\exp(s_c(x))}{\sum_{r=0}^{C-1}\exp(s_r(x))},
$$
where \(c\) and \(r\) are class indices and \(s_c(x)\) is the score or logit for class \(c\). This is natural when \(C\leq10\), but it becomes mismatched when the downstream dataset has many more classes than the model saw during pretraining.

TabICLv2 does not train a new \(C\)-way head; it reuses the pretrained \(\leq 10\)-class interface in two places.

### Hierarchical classification

Let the full class set be
$$
\mathcal{Y}=\{0,1,\ldots,C-1\}.
$$
A partition splits \(\mathcal{Y}\) into disjoint groups
$$
\mathcal{Y}=\mathcal{G}_0\cup\mathcal{G}_1\cup\cdots\cup\mathcal{G}_{K-1},
\qquad
\mathcal{G}_a\cap\mathcal{G}_b=\varnothing \quad(a\ne b),
$$
where \(K\leq10\), each \(\mathcal{G}_k\) is a group of classes, and the groups are non-overlapping. The first classifier predicts which group contains the true class. If a group is still too large, it can be partitioned again. Repeating this process forms a hierarchy whose internal nodes each have at most 10 children. When a node contains at most 10 original classes, the native classifier predicts directly among those classes.

Once the class set has been organized into this hierarchy, each original class can be described by the sequence of local decisions needed to reach it: group choices at internal nodes, followed by a final class choice inside the leaf node. For a class \(c\), write \(\pi(c)\) for the **decision path to class \(c\)**, \(b_t(c)\) for the **local branch or class choice at depth \(t\)** along that path, and \(D(c)\) for the number of local decisions needed to identify class \(c\). Formally,
$$
\pi(c)=(b_0(c),b_1(c),\ldots,b_{D(c)-1}(c)).
$$
Each branch satisfies the local constraint
$$
b_t(c)\in\{0,\ldots,K_t-1\},
\qquad K_t\leq10,
$$
where \(K_t\) is the number of available local choices at depth \(t\) along the path for \(c\). So the model never has to solve a \(C\)-way decision directly. It solves a sequence of at-most-10-way decisions whose combination identifies one original class. For practitioners, this is analogous to replacing a flat product classifier with a taxonomy: first predict department, then category, then subcategory, then item.

At depth \(t\), the model predicts the next branch conditioned on the previous branch choices:
$$
p\left(b_t \mid x,b_0,\ldots,b_{t-1}\right).
$$
The chain rule of probability applied to the path \(\pi(c)\) that identifies class \(c\) gives
$$
p(y=c\mid x)
=
\prod_{t=0}^{D(c)-1}
p\left(b_t(c)\mid x,b_0(c),\ldots,b_{t-1}(c)\right).
$$
In words: the **probability of class \(c\)** equals the **product of branch probabilities along the path** \(\pi(c)\). In implementation, TabICLv2 never applies a single \(C\)-way softmax; it composes probabilities from several native at-most-10-way predictions instead.

### Building the tree in TabICLv2

TabICLv2 builds balanced groups from the sorted observed class labels. If a node contains \(N\) classes and \(N>10\), the number of child groups is
$$
K=\min\left(\left\lceil\frac{N}{10}\right\rceil,10\right),
$$
where \(N\) is the number of classes at the current node. The \(N\) classes are then split into \(K\) nearly equal contiguous groups. Any child group that still contains more than 10 classes is split again. This keeps every local prediction within the model's native class capacity.

For example, with \(C=57\), the root node uses
$$
K=\min(\lceil57/10\rceil,10)=6
$$
groups, with sizes close to \(57/6\). The first three groups contain 10 classes each, and the last three groups contain 9 classes each. Because every group already has at most 10 classes, the root predicts among six groups and each child node predicts directly among its 9 or 10 original classes. For a larger \(C\), some root groups would contain more than 10 classes and would be recursively split.

![Hierarchy for C=57: root splits into six groups of 9-10 contiguous classes; each child is handled by a native small-class prediction.](./20260528-understanding-tfm-architecture-of-tabiclv2-5.assets/tabiclv2-hierarchy-c57.png)

*Hierarchy for \(C=57\): one root group decision, then direct local classification inside each 9- or 10-class child node.*

This is the same 57-class setting used later for mixed-radix ensembling—here the labels are split by contiguous ranges, not by digits.

### Inference with the native ICL predictor

TabICLv2 applies the hierarchy at inference time by recursively calling the model's native small-class ICL predictor. The hierarchy is not a new \(C\)-class output head; it reuses the pretrained at-most-10-class predictor several times. Operationally:

1. **Partition** the class set at each node into at most 10 groups (balanced, from sorted observed labels; as in the previous subsection).
2. **At a node**, relabel training rows by their group index and run the native ICL classifier on the test row to obtain group probabilities.
3. **At leaf nodes**, run the native classifier directly among the original classes assigned to that node.
4. **Compose probabilities for all valid classes** by multiplying the group probabilities along the route with the leaf-level class probability; take the argmax for a hard prediction.

The final probability of a class is not produced by choosing one predicted group at each level. The official implementation recursively scores child nodes and combines probabilities, so every valid class receives a probability.

### Picking the predicted class

At inference time, the predicted class can be obtained by scoring valid leaves. For every valid class \(c<C\), let \(S(c)\) denote the path score: the product of the internal group probabilities and the final local class probability along \(\pi(c)\). For numerical stability, work in log space:
$$
\hat{y}
=
\arg\max_{0\leq c<C}
\sum_{t=0}^{D(c)-1}
\log p\left(b_t(c)\mid x,b_0(c),\ldots,b_{t-1}(c)\right).
$$

### Mixed-radix ensembling

Hierarchical classification fixes prediction at test time. Context rows still carry labels into target-aware embedding earlier—that is where **mixed-radix ensembling** enters.

Mixed-radix representation encodes a large class id as several small digits. Choose \(D\) bases, also called radices,
$$
[k_0,k_1,\ldots,k_{D-1}]
$$
such that
$$
2\leq k_i\leq10,
\qquad
\prod_{i=0}^{D-1}k_i\geq C.
$$
The product condition ensures that there are enough digit combinations to represent all \(C\) classes. TabICLv2 computes bases satisfying \(\prod_i k_i\geq C\); many choices are valid. The implementation often prefers **balanced bases** so digit views have similar cardinalities—not one huge digit and one tiny one. For example, with \(C=25\) and a native limit of 10 classes, \([5,5]\) is preferable to \([10,3]\) because both digits then have five possible values.

For a class label \(y\in\{0,\ldots,C-1\}\), define positional weights
$$
w_i=\prod_{j=i+1}^{D-1}k_j,
\qquad i=0,\ldots,D-1,
$$
with the convention that an empty product is \(1\), so \(w_{D-1}=1\). Here \(w_i\) is the place value of digit \(i\). The mixed-radix digits are
$$
y^{(i)}=\left\lfloor\frac{y}{w_i}\right\rfloor \bmod k_i,
\qquad i=0,\ldots,D-1.
$$
Each digit satisfies
$$
y^{(i)}\in\{0,\ldots,k_i-1\},
$$
so every digit is compatible with the at-most-10-class pretraining regime. The original label can be reconstructed from its digits:
$$
y=\sum_{i=0}^{D-1}y^{(i)}w_i,
$$
for represented labels \(y<C\). If \(\prod_i k_i>C\), some digit combinations correspond to no real class and are invalid during decoding.

For example, suppose \(C=57\). The following uses one balanced illustrative choice, \([8,8]\), because \(8\cdot8=64\geq57\); another valid decomposition is \([10,6]\), since \(10\cdot6=60\geq57\). With \([8,8]\), digit \(y^{(0)}\) is the **high place** and \(y^{(1)}\) is the **low place** (analogous to tens and ones in base 10, but with radix 8 at each position):
$$
y^{(0)}=\left\lfloor\frac{y}{8}\right\rfloor \bmod 8,
\qquad
y^{(1)}=y\bmod 8.
$$
Class \(y=42\) becomes \((y^{(0)},y^{(1)})=(5,2)\), because \(42=5\cdot8+2\). Class \(y=56\) becomes \((7,0)\). The combinations \((7,1)\) through \((7,7)\) represent \(57\) through \(63\), so they are unused when the true class set has only 57 classes. That is a second decomposition of the same 57-class set: hierarchy splits contiguous label ranges; mixed radix splits a class id into digits.

In TabICLv2, the mixed-radix digits provide several small-label views of the original class. Instead of embedding the large class id \(y\) directly, the model embeds one digit \(y^{(i)}\) at a time. In words: a **large label becomes several small digits**, and TabICLv2 **averages column-transformer outputs** across those digit views. Operationally, TabICLv2 creates several versions of the labeled context, one per digit, runs \(\text{TF}_\text{col}\) once per digit (\(D\) forward passes), and averages the resulting representations. Let \(E_1\) denote the feature-group representation before target-aware embedding, and let \(\text{Embed}_\text{TAE}(y^{(i)})\) denote the **target-aware embedding for digit \(i\)** on labeled context rows. Then
$$
O_\text{avg}
=\frac{1}{D}\sum_{i=0}^{D-1}
\text{TF}_\text{col}\left(E_1+\text{Embed}_\text{TAE}(y^{(i)})\right).
$$
Here \(O_\text{avg}\) is the averaged column-transformer representation across the \(D\) digit views. I'll call this **mixed-radix ensembling**, or **MRE**. It exposes information about a large class label through several small-label views, each compatible with the pretrained target-aware embedding interface.

### Implementation in NanoTabICL: the boundary

NanoTabICL exposes the native small-class classification interface, but it does **not** implement the full many-class machinery from TabICLv2. In particular, the compact repository does not include:

- the recursive hierarchical classification wrapper;
- mixed-radix digit construction;
- multiple \(\text{TF}_\text{col}\) passes over digit views;
- path-probability decoding for \(C>10\).

What it does show clearly is the native interface that the many-class methods reuse.

From the README, the standard classification setup is:

```python
model = NanoTabICLv2(max_classes=10, out_dim=10)
X_train_and_test = torch.randn(batch_size, n_train+n_test, n_cols)
y_train = torch.randint(10, size=(batch_size, n_train)).float()
y_test_pred_logits = model(X_train_and_test, y_train)
```

Here `max_classes=10` controls the label embeddings used for context rows, while `out_dim=10` controls the number of logits predicted for each test row. The expected output shape is:

```text
(batch, n_test, 10)
```

The embedding side is initialized like this:

```python
self.y_embed_in = ClassEmbedding(max_classes, embed_dim)
self.y_embed_icl = ClassEmbedding(max_classes, icl_dim)
```

Both are `ClassEmbedding` layers when `max_classes > 0`. The first one injects labels into feature tokens before \(\text{TF}_\text{col}\); the second injects labels into row tokens before \(\text{TF}_\text{icl}\). The output side is controlled separately:

```python
self.out_mlp = get_mlp(icl_dim, icl_dim * 2, out_dim)
```

So, in the small-class case, the implementation path is:

```text
training labels 0..9
    -> ClassEmbedding(max_classes, ...)
    -> target-aware feature and row tokens
    -> ICL over rows
    -> out_dim logits per test row
```

The many-class strategy described in this post can be understood as a set of wrappers around this native interface. Mixed-radix ensembling repeatedly feeds small digit labels into the target-aware embedding side. Hierarchical classification repeatedly asks the native small-class predictor to solve local branch decisions. NanoTabICL stops at the native interface, which is useful pedagogically because it shows the component that those wrappers depend on, without adding the production-level orchestration code.

### Putting it together

Mixed-radix ensembling and hierarchical classification solve different parts of the same many-class issue. MRE keeps label embeddings small inside \(\text{TF}_\text{col}\). Hierarchical classification keeps downstream predictions small inside \(\text{TF}_\text{icl}\). Together, they let TabICLv2 handle datasets with many more than 10 classes while reusing the model components trained on at-most-10-class tasks.

| Mechanism | Stage | Problem solved |
|-----------|--------|----------------|
| Mixed-radix ensembling (MRE) | \(\text{TF}_\text{col}\) / TAE | Embed context labels when \(C>10\) |
| Hierarchical classification | \(\text{TF}_\text{icl}\) | Predict test label when \(C>10\) |

## Summary

**Takeaway:** Many-class TabICLv2 decomposes large label spaces with mixed-radix ensembling on the input side and hierarchical classification on the output side, reusing the native at-most-10-class interface throughout.

Many-class classification in TabICLv2 decomposes a large label space into smaller classification problems: MRE on the input side (TAE and \(\text{TF}_\text{col}\)), hierarchical classification on the output side (\(\text{TF}_\text{icl}\)). The next post covers quantile predictions for regression, the regression strategy TabICLv2 uses to model predictive uncertainty without discretizing the target into classification bins.
