[Mohit Saharan](https://linkedin.com/in/msaharan), P30, 20260528
___
# Understanding Tabular Foundation models: the architecture of TabICLv2 - 5

Subtitle: Many-class classification
___
The previous post covered query-aware scalable softmax, which improves attention behavior when the context grows. This post covers many-class classification, where TabICLv2 extends a model pretrained with at most 10 classes to settings with many more labels.

As a reminder, the architecture of TabICLv2 is illustrated in the following figure. Here, given an input \(X\in\mathbb{R}^{n\times m}\), repeated feature grouping encodes columns into multigroups via circular shifts to break feature symmetries, and target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each feature through a set transformer, \(\text{TF}_\text{row}\) aggregates features into row representations \(h\), and  \(\text{TF}_\text{icl}\) performs in-context learning tomorrow predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmaxx), is applied in part of  \(\text{TF}_\text{col}\) where inducing points aggregate input information and  \(\text{TF}_\text{icl}\) to mitigate attention fading and improve long-context generalization. 

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-2.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Many-class classification

Like many tabular foundation models, TabICLv2 is pretrained with classification tasks that have at most 10 classes. A direct \(C\)-class classifier would predict
$$
p(y=c\mid x)=\frac{\exp(s_c(x))}{\sum_{r=0}^{C-1}\exp(s_r(x))},
$$
where \(s_c(x)\) is the score or logit for class \(c\). This is natural when \(C\leq10\), but it becomes mismatched when the downstream dataset has many more classes than the model saw during pretraining.

The general solution is hierarchical partitioning: turn one large classification problem into several smaller ones. Let the full class set be
$$
\mathcal{Y}=\{0,1,\ldots,C-1\}.
$$
A partition splits \(\mathcal{Y}\) into disjoint groups
$$
\mathcal{Y}=\mathcal{G}_0\cup\mathcal{G}_1\cup\cdots\cup\mathcal{G}_{K-1},
\qquad
\mathcal{G}_a\cap\mathcal{G}_b=\varnothing \quad(a\ne b),
$$
where \(K\leq10\). The first classifier predicts which group contains the true class. If a group is still too large, it can be partitioned again. Repeating this process forms a tree whose leaves are the original classes and whose internal nodes each have at most 10 children.

For a class \(c\), let
$$
\pi(c)=(b_0(c),b_1(c),\ldots,b_{D(c)-1}(c))
$$
be the path from the root to the leaf for class \(c\). Each \(b_t(c)\) is a branch index at depth \(t\), with the local constraint
$$
b_t(c)\in\{0,\ldots,K_t-1\},
\qquad K_t\leq10.
$$
So the model never has to solve a \(C\)-way decision directly. It solves a sequence of at-most-10-way decisions whose combination identifies one original class. For practitioners, this is analogous to replacing a flat product classifier with a taxonomy: first predict department, then category, then subcategory, then item.

TabICLv2 has an additional complication. Target-aware embedding injects labels before hierarchical classification in the ICL stage. If the original label \(y\) can take \(C>10\) values, then directly embedding \(y\) would again exceed the pretrained class range. TabICLv2 addresses this with mixed-radix ensembling.

Mixed radix representation is a way to encode a large class id as several small digits. Choose bases
$$
[k_0,k_1,\ldots,k_{D-1}]
$$
such that
$$
2\leq k_i\leq10,
\qquad
\prod_{i=0}^{D-1}k_i\geq C.
$$
The product condition ensures that there are enough digit combinations to represent all \(C\) classes.

For a class label \(y\in\{0,\ldots,C-1\}\), define positional weights
$$
w_0=1,
\qquad
w_i=\prod_{r=0}^{i-1}k_r \quad \text{for } i\geq1.
$$
The mixed-radix digits are
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
for all represented labels \(y<C\). If \(\prod_i k_i>C\), some digit combinations correspond to no real class and are invalid during decoding.

For example, suppose \(C=57\). One valid choice is \([10,6]\), because \(10\cdot6=60\geq57\). Then
$$
y^{(0)}=y\bmod 10,
\qquad
y^{(1)}=\left\lfloor\frac{y}{10}\right\rfloor \bmod 6.
$$
Class \(y=42\) becomes \((y^{(0)},y^{(1)})=(2,4)\), because \(42=2+10\cdot4\). Class \(y=56\) becomes \((6,5)\). The combinations \((7,5)\), \((8,5)\), and \((9,5)\) represent \(57\), \(58\), and \(59\), so they are unused when the true class set has only \(57\) classes.

In TabICLv2, each digit defines a coarser grouping of the original classes. Instead of embedding the large class id \(y\) directly, the model embeds one digit \(y^{(i)}\) at a time. It runs \(\text{TF}_\text{col}\) once per digit and averages the outputs:
$$
O_\text{avg}
=\frac{1}{D}\sum_{i=0}^{D-1}
\text{TF}_\text{col}\left(E_1+\text{Embed}_\text{TAE}(y^{(i)})\right).
$$
This is mixed-radix ensembling. It exposes information about a large class label through several small-label views, each compatible with the pretrained target-aware embedding interface.

The ICL stage then uses hierarchical classification to compose these smaller decisions into a prediction over the original classes. If class \(c\) corresponds to path
$$
\pi(c)=(b_0(c),b_1(c),\ldots,b_{D(c)-1}(c)).
$$
then at depth \(t\), the model predicts the next branch conditioned on the previous branches:
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

The training loss for an example \((x,y)\) with path \(\pi(y)\) is the sum of local cross-entropies:
$$
\mathcal{L}(x,y)
=
-\sum_{t=0}^{D(y)-1}
\log p\left(b_t(y)\mid x,b_0(y),\ldots,b_{t-1}(y)\right).
$$
Each term is a small classification problem because each node has at most \(10\) children. This matches the model's pretrained class capacity even when \(C\) is much larger.

If mixed-radix digits are used as the hierarchy, the path can be taken as
$$
\pi(y)=(y^{(D-1)},y^{(D-2)},\ldots,y^{(0)})
$$
or another fixed digit order. The order determines whether the classifier first predicts coarse high-radix-place information or fine low-radix-place information. High-order digits usually define coarser groups because they select larger contiguous ranges of original class ids. Low-order digits distinguish classes within those groups.

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

The benefit is scalability. The local prediction heads remain small, while the number of representable final classes grows multiplicatively with depth. With bases \([k_0,\ldots,k_{D-1}]\), the hierarchy can represent up to
$$
\prod_{i=0}^{D-1}k_i
$$
classes even though no local decision has more than \(\max_i k_i\leq10\) choices.

Combined with heirarchical classification in \(\text{TF}_\text{icl}\), this enables TabICLv2 to handle an arbitrary number of classes.

## Summary

Many-class classification in TabICLv2 decomposes a large label space into smaller classification problems. Mixed-radix ensembling makes target-aware embedding compatible with the pretrained class limit, while hierarchical classification composes small local decisions into predictions over many original classes. The next post covers quantile predictions for regression, the regression strategy TabICLv2 uses to model predictive uncertainty without discretizing the target into classification bins.
