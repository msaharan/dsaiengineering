# Understanding Tabular Foundation models: the architecture of TabICLv2-5

Source: TabICLv2 paper. https://arxiv.org/pdf/2602.11139.

In the previous post, we covered query-aware scalable softmax, which improves attention behavior when the context grows. In this post, we cover many-class classification, where TabICLv2 extends a model pretrained with at most 10 classes to settings with many more labels.

## Illustration and summary

The architecture of TabICLv2 is illustrated in the following figure. Here, given an input \(X\in\mathbb{R}^{n\times m}\), repeated feature grouping encodes columns into multigroups via circular shifts to break feature symmetries, and target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each feature through a set transformer, \(\text{TF}_\text{row}\) aggregates features into row representations \(h\), and  \(\text{TF}_\text{icl}\) performs in-context learning tomorrow predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmaxx), is applied in part of  \(\text{TF}_\text{col}\) where inducing points aggregate input information and  \(\text{TF}_\text{icl}\) to mitigate attention fading and improve long-context generalization. 

The following subsections elaborate on the summary.

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-5.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Many-class classification

Like many TFMs, TabICLv2 is pretrained with up to 10 classes. It uses classification (Qu et al., 2025) at the ICL stage for more classes. However, target-aware embedding introduces labels before heirarchical partitioning. To address this, it proposes mixed-radix ensebling: for \(C>10\) classes, it computes bases \([k_0, \dots,k_{D-1}]\) with each \(k_i\leq10\) and \(\prod_{i} k_i \ge C \), then decompose each label \(y\) into \(D\) digits \(y^i\in\{0,\dots,k_i-1\}\) via mixed radix representation. Each digit defines a coarser grouping of the original classes. It runs \(\text{TF}_\text{col}\) once per digit and average the outputs:
$$
O_\text{avg} = \dfrac{1}{D}\sum_{i=0}^{D-1} \text{TF}_\text{col}(E_1 + \text{Embed}_{TAE}(y^{(i)})).
$$
Combined with heirarchical classification in \(\text{TF}_\text{icl}\), this enables TabICLv2 to handle an arbitrary number of classes.

## Summary

Many-class classification in TabICLv2 decomposes a large label space into smaller classification problems. Mixed-radix ensembling makes target-aware embedding compatible with the pretrained class limit, while hierarchical classification composes small local decisions into predictions over many original classes.

#  Appendix

### Heirarchical partitioning

Many-class classification becomes difficult when the model was pretrained to handle only a small maximum number of classes, such as \(10\). A direct \(C\)-class classifier predicts
$$
p(y=c\mid x)=\frac{\exp(s_c(x))}{\sum_{r=0}^{C-1}\exp(s_r(x))},
$$
where \(s_c(x)\) is the score or logit for class \(c\). If \(C\) is much larger than the class count seen during pretraining, the output space no longer matches the model's learned classification interface.

Hierarchical partitioning turns one large \(C\)-class problem into several smaller classification problems. Let the full class set be
$$
\mathcal{Y}=\{0,1,\ldots,C-1\}.
$$
A partition splits \(\mathcal{Y}\) into disjoint groups
$$
\mathcal{Y}=\mathcal{G}_0\cup\mathcal{G}_1\cup\cdots\cup\mathcal{G}_{K-1},
\qquad
\mathcal{G}_a\cap\mathcal{G}_b=\varnothing \quad(a\ne b),
$$
where \(K\leq 10\). The first classifier only predicts which group contains the true class. If a group is still too large, it can be partitioned again. Repeating this process forms a tree whose leaves are the original classes and whose internal nodes each have at most \(10\) children.

For a class \(c\), let
$$
\pi(c)=(b_0(c),b_1(c),\ldots,b_{D(c)-1}(c))
$$
be the path from the root to the leaf for class \(c\), where each \(b_t(c)\) is a branch index at depth \(t\). The key constraint is local:
$$
b_t(c)\in\{0,\ldots,K_t-1\},\qquad K_t\leq 10.
$$
So the model never has to solve a \(C\)-way decision directly. It solves a sequence of at-most-10-way decisions whose combination identifies one of many original classes.

For practitioners, this is analogous to replacing a flat product catalog classifier with a taxonomy: first predict department, then category, then subcategory, then item. The total number of items can be large, while each local decision remains small.

### Mixed radix representation

Mixed radix representation is a structured way to create such small decisions without manually designing a semantic hierarchy. A usual base-\(10\) number represents an integer by digits where every digit has radix \(10\). Mixed radix allows each digit position to have its own radix.

Choose bases
$$
[k_0,k_1,\ldots,k_{D-1}]
$$
with
$$
2\leq k_i\leq 10,\qquad \prod_{i=0}^{D-1}k_i\geq C.
$$
The product condition ensures that the digit system has enough unique codes to represent all \(C\) classes. For a class label \(y\in\{0,\ldots,C-1\}\), define the positional weights
$$
w_0=1,\qquad w_i=\prod_{r=0}^{i-1}k_r \quad \text{for } i\geq1.
$$
The mixed-radix digits are
$$
y^{(i)}=\left\lfloor\frac{y}{w_i}\right\rfloor \bmod k_i,
\qquad i=0,\ldots,D-1.
$$
Each digit satisfies \(y^{(i)}\in\{0,\ldots,k_i-1\}\), so each digit is an at-most-10-class label.

The original label can be reconstructed from its digits:
$$
y=\sum_{i=0}^{D-1}y^{(i)}w_i,
$$
for all represented labels \(y<C\). If \(\prod_i k_i>C\), some digit combinations correspond to no real class and should be treated as invalid or masked during final decoding.

As a concrete example, suppose \(C=57\). One valid choice is \([10,6]\), because \(10\cdot6=60\geq57\). Then
$$
y^{(0)}=y\bmod 10,\qquad
y^{(1)}=\left\lfloor\frac{y}{10}\right\rfloor \bmod 6.
$$
Class \(y=42\) becomes \((y^{(0)},y^{(1)})=(2,4)\), because \(42=2+10\cdot4\). Class \(y=56\) becomes \((6,5)\). The combinations \((7,5)\), \((8,5)\), and \((9,5)\) represent \(57\), \(58\), and \(59\), so they are unused when the true class set has only \(57\) classes.

In TabICLv2's many-class setting, these digits are useful because target-aware embedding was pretrained for small class counts. Instead of embedding a large class id \(y\) directly, the model embeds each digit \(y^{(i)}\), where each digit has at most \(10\) possible values. Mixed-radix ensembling then runs the target-aware part once per digit and averages the resulting representations:
$$
O_\text{avg}
=\frac{1}{D}\sum_{i=0}^{D-1}
\text{TF}_\text{col}\left(E_1+\text{Embed}_\text{TAE}(y^{(i)})\right).
$$
This lets the model expose information about a large class label through several small-label views, each compatible with the pretraining regime.

### Heirarchical classification

Hierarchical classification predicts a large class label by composing probabilities along a hierarchy. Let \(x\) denote the row or row representation being classified, and let class \(c\) correspond to path
$$
\pi(c)=(b_0(c),b_1(c),\ldots,b_{D(c)-1}(c)).
$$
At depth \(t\), the model predicts the next branch conditioned on the previous branches:
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
This is just the chain rule of probability applied to the path that identifies the class.

The training loss for an example \((x,y)\) with path \(\pi(y)\) is the sum of local cross-entropies:
$$
\mathcal{L}(x,y)
=
-\sum_{t=0}^{D(y)-1}
\log p\left(b_t(y)\mid x,b_0(y),\ldots,b_{t-1}(y)\right).
$$
Each term is a small classification problem, because each node has at most \(10\) children. This matches the model's pretrained class capacity even when the number of final classes \(C\) is much larger.

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

The benefit is scalability: the model's local prediction heads remain small, while the number of representable final classes grows multiplicatively with depth. With bases \([k_0,\ldots,k_{D-1}]\), the hierarchy can represent up to
$$
\prod_{i=0}^{D-1}k_i
$$
classes even though no local decision has more than \(\max_i k_i\leq10\) choices.
