[Mohit Saharan](https://linkedin.com/in/msaharan), P30, 20260528
___
# Understanding Tabular Foundation Models: the architecture of TabICLv2 - 5

Subtitle: Many-class classification
___
The previous post covered query-aware scalable softmax, which improves attention behavior when the context grows. This post covers many-class classification, where TabICLv2 extends a model pretrained with at most 10 classes to settings with many more labels.

As a reminder, the architecture of TabICLv2 is illustrated in the following figure. Here, given an input \(X\in\mathbb{R}^{n\times m}\), where \(n\) is the number of rows and \(m\) is the number of features, repeated feature grouping encodes columns into grouped feature positions via circular shifts to break feature symmetries, and target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each grouped feature position through a set transformer, \(\text{TF}_\text{row}\) aggregates grouped feature embeddings into row representations \(h\), and \(\text{TF}_\text{icl}\) performs in-context learning to predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmax) is applied in part of \(\text{TF}_\text{col}\), where inducing points aggregate input information, and in \(\text{TF}_\text{icl}\) to mitigate attention fading and improve long-context generalization.

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-5.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Many-class classification

Like many tabular foundation models, TabICLv2 is pretrained with classification tasks that have at most 10 classes. Let \(C\) be the number of downstream classes, \(x\) be the row or row representation being classified, and \(y\in\{0,\ldots,C-1\}\) be the true class label. A direct \(C\)-class classifier would predict
$$
p(y=c\mid x)=\frac{\exp(s_c(x))}{\sum_{r=0}^{C-1}\exp(s_r(x))},
$$
where \(c\) and \(r\) are class indices and \(s_c(x)\) is the score or logit for class \(c\). This is natural when \(C\leq10\), but it becomes mismatched when the downstream dataset has many more classes than the model saw during pretraining.

TabICLv2 uses two related ideas to avoid that mismatch. The first is used before the ICL stage: mixed-radix ensembling makes target-aware embedding compatible with more than 10 possible labels. The second is used in the ICL stage: hierarchical classification turns one large prediction problem into several smaller native prediction problems. I will start with the hierarchy because it is easier to see the many-class bottleneck at the output layer first; mixed-radix ensembling then solves the analogous label-embedding bottleneck on the input side.

The general idea of hierarchical partitioning is to turn one large classification problem into several smaller ones. Let the full class set be
$$
\mathcal{Y}=\{0,1,\ldots,C-1\}.
$$
A partition splits \(\mathcal{Y}\) into disjoint groups
$$
\mathcal{Y}=\mathcal{G}_0\cup\mathcal{G}_1\cup\cdots\cup\mathcal{G}_{K-1},
\qquad
\mathcal{G}_a\cap\mathcal{G}_b=\varnothing \quad(a\ne b),
$$
where \(K\leq10\), each \(\mathcal{G}_k\) is a group of classes, and the groups are non-overlapping. The first classifier predicts which group contains the true class. If a group is still too large, it can be partitioned again. Repeating this process forms a tree whose leaves are the original classes and whose internal nodes each have at most 10 children.

Once the class set has been organized into this tree, each original class can be described by the sequence of branch choices needed to reach it.

For a class \(c\), let
$$
\pi(c)=(b_0(c),b_1(c),\ldots,b_{D(c)-1}(c))
$$
be the path from the root to the leaf for class \(c\), where \(D(c)\) is the number of branch decisions needed to reach class \(c\). Each \(b_t(c)\) is a branch index at depth \(t\), with the local constraint
$$
b_t(c)\in\{0,\ldots,K_t-1\},
\qquad K_t\leq10.
$$
Here \(K_t\) denotes the number of available branches at the node reached at depth \(t\) along the path for \(c\). So the model never has to solve a \(C\)-way decision directly. It solves a sequence of at-most-10-way decisions whose combination identifies one original class. For practitioners, this is analogous to replacing a flat product classifier with a taxonomy: first predict department, then category, then subcategory, then item.

At depth \(t\), the model predicts the next branch conditioned on the previous branch choices:
$$
p\left(b_t \mid x,b_0,\ldots,b_{t-1}\right).
$$
The class probability is then factorized as
$$
p(y=c\mid x)
=
\prod_{t=0}^{D(c)-1}
p\left(b_t(c)\mid x,b_0(c),\ldots,b_{t-1}(c)\right).
$$
This is the chain rule of probability applied to the path that identifies the class.

In TabICLv2 this hierarchy is used at inference time by recursively calling the model's native small-class ICL predictor. At a given node, the training examples belonging to that node are relabeled by their group index, and the model predicts group probabilities for the test row. At a leaf node, the remaining classes are few enough that the model predicts them directly. The final probability of a class is the product of the group probabilities along the route to that leaf. The hierarchy is therefore not a new \(C\)-class output head; it is a way to reuse the pretrained at-most-10-class predictor several times.

The remaining design question is how TabICLv2 chooses those groups in the first place.

TabICLv2 builds balanced groups from the sorted observed class labels. If a node contains \(N\) classes and \(N>10\), the number of child groups is
$$
K=\min\left(\left\lceil\frac{N}{10}\right\rceil,10\right),
$$
where \(N\) is the number of classes at the current node. The \(N\) classes are then split into \(K\) nearly equal contiguous groups. Any child group that still contains more than 10 classes is split again. This keeps every local prediction within the model's native class capacity.

For example, with \(C=57\), the root node uses
$$
K=\min(\lceil57/10\rceil,10)=6
$$
groups, with sizes close to \(57/6\). The first three groups contain 10 classes each, and the last three groups contain 9 classes each. Because every group already has at most 10 classes, the tree has one internal root level and leaf-level direct classification problems. For a larger \(C\), some root groups would contain more than 10 classes and would be recursively split.

At inference time, the predicted class can be obtained by scoring valid leaves. For every valid class \(c<C\), compute its path probability
$$
S(c)=\prod_{t=0}^{D(c)-1}
p\left(b_t(c)\mid x,b_0(c),\ldots,b_{t-1}(c)\right),
$$
and select
$$
\hat{y}=\arg\max_{0\leq c<C} S(c).
$$
Equivalently, one may work in log space for numerical stability:
$$
\hat{y}
=
\arg\max_{0\leq c<C}
\sum_{t=0}^{D(c)-1}
\log p\left(b_t(c)\mid x,b_0(c),\ldots,b_{t-1}(c)\right).
$$

The hierarchy fixes prediction after row representations have already been formed. But TabICLv2 also needs to represent the labels of context examples before that stage, inside target-aware embedding. If the original label \(y\) can take \(C>10\) values, directly embedding \(y\) would exceed the pretrained classification label range. TabICLv2 addresses this with mixed-radix ensembling.

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
The product condition ensures that there are enough digit combinations to represent all \(C\) classes. TabICLv2 chooses balanced bases so that the digit views have similar cardinalities. For example, with \(C=25\) and a native limit of 10 classes, it uses \([5,5]\), not \([10,3]\), because both digits then have five possible values.

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

For example, suppose \(C=57\). A balanced choice is \([8,8]\), because \(8\cdot8=64\geq57\). Then
$$
y^{(0)}=\left\lfloor\frac{y}{8}\right\rfloor \bmod 8,
\qquad
y^{(1)}=y\bmod 8.
$$
Class \(y=42\) becomes \((y^{(0)},y^{(1)})=(5,2)\), because \(42=5\cdot8+2\). Class \(y=56\) becomes \((7,0)\). The combinations \((7,1)\) through \((7,7)\) represent \(57\) through \(63\), so they are unused when the true class set has only 57 classes.

In TabICLv2, the mixed-radix digits provide several small-label views of the original class. Instead of embedding the large class id \(y\) directly, the model embeds one digit \(y^{(i)}\) at a time. Operationally, TabICLv2 creates several versions of the labeled context, one per digit, runs the column transformer on each version, and averages the resulting representations. Let \(E_1\) denote the feature-group representation before target-aware embedding, and let \(\text{Embed}_\text{TAE}(y^{(i)})\) denote the target-aware embedding vector for digit \(i\) on labeled context rows. The column transformer runs once per digit and averages the outputs:
$$
O_\text{avg}
=\frac{1}{D}\sum_{i=0}^{D-1}
\text{TF}_\text{col}\left(E_1+\text{Embed}_\text{TAE}(y^{(i)})\right).
$$
Here \(O_\text{avg}\) is the averaged column-transformer representation across the \(D\) digit views. This is mixed-radix ensembling. It exposes information about a large class label through several small-label views, each compatible with the pretrained target-aware embedding interface.

The key point is that mixed-radix ensembling and hierarchical classification solve different parts of the same many-class issue. Mixed-radix ensembling keeps label embeddings small inside \(\text{TF}_\text{col}\). Hierarchical classification keeps downstream predictions small inside \(\text{TF}_\text{icl}\). Together, they let TabICLv2 handle datasets with many more than 10 classes while reusing the model components trained on at-most-10-class tasks.

## Summary

Many-class classification in TabICLv2 decomposes a large label space into smaller classification problems. Mixed-radix ensembling makes target-aware embedding compatible with the pretrained class limit, while hierarchical classification composes small local decisions into predictions over many original classes. The next post covers quantile predictions for regression, the regression strategy TabICLv2 uses to model predictive uncertainty without discretizing the target into classification bins.
