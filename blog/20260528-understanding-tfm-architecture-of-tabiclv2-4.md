[Mohit Saharan](https://linkedin.com/in/msaharan), P29, 20260528
___
# Understanding Tabular Foundation models: the architecture of TabICLv2 - 4

Subtitle: Query-aware scalable softmax
___
The previous post covered how TabICLv2 compresses feature-level information into row representations and then performs in-context learning. This post covers query-aware scalable softmax, the attention-scaling mechanism TabICLv2 uses to preserve selective attention as the number of context samples grows.

As a reminder, the architecture of TabICLv2 is illustrated in the following figure. Here, given an input \(X\in\mathbb{R}^{n\times m}\), repeated feature grouping encodes columns into multigroups via circular shifts to break feature symmetries, and target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each feature through a set transformer, \(\text{TF}_\text{row}\) aggregates features into row representations \(h\), and  \(\text{TF}_\text{icl}\) performs in-context learning to predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmax) is applied in part of  \(\text{TF}_\text{col}\), where inducing points aggregate input information, and in \(\text{TF}_\text{icl}\) to mitigate attention fading and improve long-context generalization. 

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-4.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Query-aware scalable softmax

Query-aware scalable softmax, or QASSMax, is a modification to the attention softmax used in parts of \(\text{TF}_\text{col}\) and \(\text{TF}_\text{icl}\). Its purpose is to keep attention selective when the number of context samples becomes much larger than the sequence lengths seen during pretraining.

Start with standard scaled dot-product attention. For one attention head, a query \(q\in\mathbb{R}^{d_\text{head}}\) is compared with \(n\) keys \(k_1,\ldots,k_n\in\mathbb{R}^{d_\text{head}}\), where \(d_\text{head}\) is the dimension of that head. The unnormalized attention logit for key \(j\) is
$$
z_j=\frac{q^\top k_j}{\sqrt{d_\text{head}}},
$$
where \(j\in\{1,\ldots,n\}\). The attention weight assigned to key \(j\) is
$$
a_j=\text{softmax}(z)_j=\frac{\exp(z_j)}{\sum_{\ell=1}^{n}\exp(z_\ell)}.
$$
Here \(\ell\) is just the summation index over keys. The output is the weighted average \(\sum_{j=1}^n a_jv_j\), where \(v_j\) is the value vector associated with key \(j\). QASSMax does not replace this attention mechanism. It changes the logits before the softmax by rescaling the query.

If the query is rescaled as \(\bar{q}=\lambda q\), where \(\lambda>0\) is a scalar logit scale, then
$$
\bar{z}_j=\frac{\bar{q}^\top k_j}{\sqrt{d_\text{head}}}=\lambda z_j.
$$
So query scaling is mathematically equivalent to logit scaling. This is also equivalent to changing the softmax temperature. For a temperature \(\tau>0\),
$$
\text{softmax}_\tau(z)_j=\frac{\exp(z_j/\tau)}{\sum_{\ell=1}^{n}\exp(z_\ell/\tau)}.
$$
Writing \(\lambda=1/\tau\), we get
$$
\text{softmax}_\tau(z)=\text{softmax}(\lambda z).
$$
Lower temperature, or larger \(\lambda\), sharpens attention; higher temperature, or smaller \(\lambda\), spreads attention more broadly. The relative odds between keys \(j\) and \(\ell\) become
$$
\frac{a_j}{a_\ell}
=
\frac{\exp(\lambda z_j)}{\exp(\lambda z_\ell)}
=
\exp(\lambda(z_j-z_\ell)).
$$
Scaling therefore does not change which key has the highest logit. It changes how decisively the softmax concentrates probability mass on high-logit keys.

The need for scalable softmax comes from attention fading. Suppose one relevant key has logit \(z_\star\), and the other \(n-1\) distractors have a lower logit \(z_0\). Let \(\Delta=z_\star-z_0>0\) be the logit gap between the relevant key and a distractor. Under ordinary softmax,
$$
a_\star
=
\frac{\exp(z_\star)}{\exp(z_\star)+(n-1)\exp(z_0)}
=
\frac{1}{1+(n-1)\exp(-\Delta)}.
$$
For fixed \(\Delta\), \(a_\star\rightarrow0\) as \(n\rightarrow\infty\). Even if each distractor is individually less relevant, many distractors can collectively absorb the attention mass through the denominator.

Scalable Softmax (SSMax, Nakanishi 2025) addresses this by making the logit scale grow with context length. Let \(q_h=(q_{hi})\) be a query vector at head \(h\), with head dimension indexed by \(i\). SSMax rescales queries with a learnable per-head scalar \(s_h\):
$$
\bar{q}_{hi}=q_{hi}\cdot s_h\log n.
$$
This yields scaled logits \(\bar{z}_j=(s_h\log n)z_j\). In the one-relevant-key example,
$$
a_\star
=
\frac{1}{1+(n-1)\exp(-s_h\Delta\log n)}
\approx
\frac{1}{1+n^{1-s_h\Delta}}.
$$
This expression shows why \(\log n\) matters: to keep the relevant token visible as \(n\) grows, the relevant logit gap must effectively grow on the order of \(\log n\). The same \(n\) denotes the number of keys available to a query; in TabICLv2's ICL setting, this is closely tied to the number of context samples.

TabICLv2 extends this idea with query-aware scalable softmax. Instead of using one scalar \(s_h\) per head, QASSMax rescales each query element as
$$
\bar{q}_{hi}
=
q_{hi}
\underbrace{\cdot\text{MLP}_\text{base}(\log \,n)_{hi}}_\text{base scaling}
\cdot
\underbrace{(1+\tanh(\text{MLP}_\text{gate}(q_h)_i))}_\text{query-aware gating}.
$$
In vector notation, this is
$$
\bar{q}_h=q_h\odot B_h(n)\odot G_h(q_h),
$$
where
$$
B_h(n)=\text{MLP}_\text{base}(\log n)_h\in\mathbb{R}^{d_\text{head}},
$$
and
$$
G_h(q_h)=1+\tanh(\text{MLP}_\text{gate}(q_h))\in(0,2)^{d_\text{head}}.
$$
Here \(\odot\) denotes element-wise multiplication. For \(H\) attention heads, \(\text{MLP}_\text{base}: \mathbb{R}\rightarrow\mathbb{R}^{H\times d_\text{head}}\) takes \(\log n\) and outputs one base scale per head dimension, while \(\text{MLP}_\text{gate}:\mathbb{R}^{d_\text{head}} \rightarrow \mathbb{R}^{d_\text{head}}\) maps the current query to an element-wise gate. Both are two-layer MLPs with 64 hidden neurons and GELU activation.

GELU stands for Gaussian Error Linear Unit:
$$
\text{GELU}(x)=x\Phi(x),
$$
where \(\Phi(x)\) is the standard normal CDF. A common approximation is
$$
\text{GELU}(x)\approx
\frac{x}{2}\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right)\right)\right).
$$
In QASSMax, GELU simply gives the small MLPs a smooth nonlinearity so they can learn nontrivial functions of \(\log n\) and \(q_h\).

The base term \(B_h(n)\) handles the predictable effect of context length. This is inspired by scalable attention methods such as SSMax and ASEntmax. Entmax is a family of softmax alternatives that can produce sparse probability distributions by solving an entropy-regularized optimization problem over the probability simplex
$$
\mathcal{S}^n=\left\{p\in\mathbb{R}^n:\sum_{j=1}^n p_j=1,\ p_j\geq0\right\}.
$$
Here \(p_j\) is the probability assigned to key \(j\), and \(\mathcal{S}^n\) is the set of all valid probability vectors over \(n\) keys. At a high level,
$$
\text{entmax}_\alpha(z)
=
\arg\max_{p\in\mathcal{S}^n}\left(p^\top z + H_\alpha(p)\right),
$$
where \(H_\alpha\) is a Tsallis-entropy-style regularizer and \(\alpha\) controls sparsity. The relevant lesson for QASSMax is not sparsity itself, since QASSMax still uses softmax. The relevant lesson is that attention normalization can benefit from a learned function of context length, such as the ASEntmax form
$$
\delta+\beta(\log n)^\gamma.
$$
Here \(\delta\), \(\beta\), and \(\gamma\) are learned parameters controlling how the normalization changes with context length. QASSMax generalizes this idea through \(\text{MLP}_\text{base}(\log n)\).

The gate \(G_h(q_h)\) adds query awareness. This follows the principle behind selective attention: not every query should have the same attention sharpness. Some queries need to retrieve a highly specific row; others need to aggregate information more broadly. In a query-dependent temperature formulation, query row \(r\) might use
$$
a_{rj}
=
\frac{\exp(z_{rj}/\tau_r)}
{\sum_{\ell=1}^{n}\exp(z_{r\ell}/\tau_r)}.
$$
Here \(z_{rj}\) is the attention logit between query row \(r\) and key \(j\), \(a_{rj}\) is the resulting attention weight, and \(\tau_r>0\) is the temperature assigned to that query row.
QASSMax implements a related idea with a bounded element-wise gate. Because
$$
G_h(q_h)\in(0,2)^{d_\text{head}},
$$
the gate can reduce or increase the base scale, but it cannot grow without limit. This keeps the query-specific modulation from overwhelming the length-dependent trend.

The design is also related to gated attention. A generic gate applies a learned multiplicative control:
$$
\tilde{u}=g\odot u.
$$
Here \(u\) is a vector being modulated, \(g\) is a learned gate with the same dimension as \(u\), and \(\tilde{u}\) is the gated vector.
Some gated attention mechanisms apply the gate after attention, for example
$$
\tilde{o}=g(q)\odot \text{Attn}(q,K,V).
$$
In this expression, \(K\) is the matrix of keys, \(V\) is the matrix of values, \(\text{Attn}(q,K,V)\) is the ordinary attention output for query \(q\), and \(\tilde{o}\) is the gated output.
QASSMax applies the gate earlier, at the query-scaling stage. Because the gate changes \(\bar{q}_h\), it changes the logits before softmax:
$$
\bar{z}_j=
\frac{(q_h\odot B_h(n)\odot G_h(q_h))^\top k_j}
{\sqrt{d_\text{head}}}.
$$
So the gate affects the attention weights themselves, not only the post-attention output.

QASSMax was designed around four ideas: using \(\log n\) as the length variable counteracts the growth of the softmax denominator; \(\text{MLP}_\text{base}(\log n)\) allows a learned context-length scaling law; element-wise scaling is more expressive than a single per-head scalar; and bounded query-aware gating lets different queries adjust their attention sharpness without destabilizing the length scaling.

Applied to \(\text{TF}_\text{col}\) and \(\text{TF}_\text{icl}\), QASSMax substantially improves long-context behavior. In the paper's needle-in-haystack classification task, the model must focus on one anchor sample among many negative samples. Without scalable softmax, attention entropy rises and accuracy drops as the number of negatives grows. QASSMax maintains low entropy and 100% accuracy even with 15K negatives, outperforming SSMax at extreme scales.

## Summary

Query-aware scalable softmax modifies attention logits by rescaling queries with both a context-length-dependent base term and a bounded query-dependent gate. This helps TabICLv2 keep attention sharp in long contexts while allowing different queries and latent dimensions to adjust the amount of scaling they need. The next post covers many-class classification, where TabICLv2 extends a model pretrained with at most 10 classes to settings with many more labels.
