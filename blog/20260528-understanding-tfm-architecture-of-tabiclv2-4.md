[Mohit Saharan](https://linkedin.com/in/msaharan), P29, 20260528
___
# Understanding Tabular Foundation Models: the architecture of TabICLv2 - 4

Subtitle: Query-aware scalable softmax
___
The previous post covered how TabICLv2 compresses feature-level information into row representations and then performs in-context learning. This post covers Query-Aware Scalable Softmax (QASSMax), the attention-scaling mechanism TabICLv2 uses to preserve selective attention as the number of context samples grows.

As a reminder, the architecture of TabICLv2 is illustrated in the following figure. Write the input table as \(X\in\mathbb{R}^{n_\text{rows}\times m}\), where \(n_\text{rows}\) is the number of rows and \(m\) is the number of features. Repeated feature grouping encodes columns into grouped feature positions via circular shifts to break feature symmetries. Target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each grouped feature position through a set transformer, \(\text{TF}_\text{row}\) aggregates grouped feature embeddings into row representations \(h\), and \(\text{TF}_\text{icl}\) performs in-context learning to predict test targets \(\hat{y}\). QASSMax is applied in two places: in the first stage of the induced self-attention inside \(\text{TF}_\text{col}\), where inducing points aggregate input information, and in \(\text{TF}_\text{icl}\), where test rows attend to training rows.

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-4.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Query-aware scalable softmax

Query-aware scalable softmax, or QASSMax, modifies the softmax used inside attention. Its purpose is to keep attention selective when the number of context samples becomes much larger than the sequence lengths seen during pretraining.

To avoid overloading notation, I will use \(N\) for the number of keys in a generic attention calculation. Later, when discussing TabICLv2 and the paper's formula, I will use \(n\) for the training-set size.

Start with standard scaled dot-product attention. For one attention head, a query \(q\in\mathbb{R}^{d_\text{head}}\) is compared with \(N\) keys \(k_1,\ldots,k_N\in\mathbb{R}^{d_\text{head}}\), where \(d_\text{head}\) is the dimension of that head. The unnormalized attention logit for key \(j\) is
$$
z_j=\frac{q^\top k_j}{\sqrt{d_\text{head}}},
$$
where \(j\in\{1,\ldots,N\}\). The attention weight assigned to key \(j\) is
$$
a_j=\text{softmax}(z)_j=\frac{\exp(z_j)}{\sum_{\ell=1}^{N}\exp(z_\ell)}.
$$
Here \(z=(z_1,\ldots,z_N)\), \(a_j\) is the normalized attention weight, and \(\ell\) is only the summation index over keys. The output is the weighted average \(\sum_{j=1}^N a_jv_j\), where \(v_j\) is the value vector associated with key \(j\).

QASSMax does not replace attention. It changes the logits before the softmax by rescaling the query. In the simplest scalar case, if the query is rescaled as \(\tilde{q}=\lambda q\), where \(\lambda>0\) is a scalar logit scale, then
$$
\tilde{z}_j=\frac{\tilde{q}^\top k_j}{\sqrt{d_\text{head}}}=\lambda z_j.
$$
So scalar query scaling is mathematically equivalent to scalar logit scaling. It is also equivalent to changing the softmax temperature. For a temperature \(\tau>0\),
$$
\text{softmax}_\tau(z)_j=\frac{\exp(z_j/\tau)}{\sum_{\ell=1}^{N}\exp(z_\ell/\tau)}.
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

This matters because standard softmax can suffer from attention fading. Suppose one relevant key has logit \(z_\star\), and the other \(N-1\) distractors all have a lower logit \(z_0\). Let \(\Delta=z_\star-z_0>0\) be the logit gap between the relevant key and a distractor. Under ordinary softmax,
$$
a_\star
=
\frac{\exp(z_\star)}{\exp(z_\star)+(N-1)\exp(z_0)}
=
\frac{1}{1+(N-1)\exp(-\Delta)}.
$$
For fixed \(\Delta\), \(a_\star\rightarrow0\) as \(N\rightarrow\infty\). Even if each distractor is individually less relevant, many distractors can collectively absorb the attention mass through the denominator. The problem is not that softmax forgets the ranking of logits; the problem is that the denominator grows with the number of competing keys.

Scalable Softmax, or SSMax, addresses this by making the logit scale grow with context length. In the TabICLv2 paper's notation, let \(q_h=(q_{hi})\) be a query vector at attention head \(h\), with head dimension indexed by \(i\), and let \(n\) be the size of the training set. SSMax rescales queries with a learnable per-head scalar \(s_h\):
$$
\tilde{q}_{hi}=q_{hi}\cdot s_h\log n.
$$
This yields scaled logits \(\tilde{z}_j=(s_h\log n)z_j\). In the one-relevant-key example, replacing \(N\) by \(n\) for the training rows gives
$$
a_\star
=
\frac{1}{1+(n-1)\exp(-s_h\Delta\log n)}
\approx
\frac{1}{1+n^{1-s_h\Delta}}.
$$
The approximation uses \(n-1\approx n\) and \(\exp(-s_h\Delta\log n)=n^{-s_h\Delta}\). It shows why \(\log n\) matters: to keep the relevant token visible as \(n\) grows, the relevant logit gap must effectively grow on the order of \(\log n\). In this simplified example, the attention on the relevant key stays large when the learned scale is strong enough that \(s_h\Delta>1\).

TabICLv2 extends SSMax with query-aware scalable softmax. Instead of using one scalar \(s_h\) per head, QASSMax rescales each query element as
$$
\tilde{q}_{hi}
=
q_{hi}
\underbrace{\cdot\text{MLP}_\text{base}(\log n)_{hi}}_\text{base scaling}
\cdot
\underbrace{(1+\tanh(\text{MLP}_\text{gate}(q_h)_i))}_\text{query-aware gating}.
$$
In vector notation, this is
$$
\tilde{q}_h=q_h\odot B_h(n)\odot G_h(q_h),
$$
where
$$
B_h(n)=\text{MLP}_\text{base}(\log n)_h\in\mathbb{R}^{d_\text{head}},
$$
and
$$
G_h(q_h)=1+\tanh(\text{MLP}_\text{gate}(q_h))\in(0,2)^{d_\text{head}}.
$$
Here \(\odot\) denotes element-wise multiplication, \(B_h(n)\) is the length-dependent base vector for head \(h\), and \(G_h(q_h)\) is the query-dependent gate for that head. For \(H\) attention heads, \(\text{MLP}_\text{base}: \mathbb{R}\rightarrow\mathbb{R}^{H\times d_\text{head}}\) takes \(\log n\) and outputs one base value per head dimension, while \(\text{MLP}_\text{gate}:\mathbb{R}^{d_\text{head}} \rightarrow \mathbb{R}^{d_\text{head}}\) maps the current query to an element-wise gate. Both are two-layer MLPs with 64 hidden neurons and GELU activation. GELU stands for Gaussian Error Linear Unit; one common definition is \(\text{GELU}(x)=x\Phi(x)\), where \(\Phi(x)\) is the standard normal CDF. In the TabICLv2 implementation, the last layer of \(\text{MLP}_\text{gate}\) is initialized to zero, so the initial gate is \(G_h(q_h)=1\).

The base term \(B_h(n)\) handles the predictable effect of context length. This is inspired by scalable attention methods such as SSMax and ASEntmax. Entmax is a family of softmax alternatives that can produce sparse probability distributions by solving an entropy-regularized optimization problem over the probability simplex
$$
\mathcal{S}^N=\left\{p\in\mathbb{R}^N:\sum_{j=1}^N p_j=1,\ p_j\geq0\right\}.
$$
Here \(p=(p_1,\ldots,p_N)\) is a probability vector over \(N\) keys, \(p_j\) is the probability assigned to key \(j\), and \(\mathcal{S}^N\) is the set of all valid probability vectors over those keys. At a high level,
$$
\text{entmax}_\alpha(z)
=
\arg\max_{p\in\mathcal{S}^N}\left(p^\top z + H_\alpha(p)\right),
$$
where \(z\) is the vector of logits, \(H_\alpha\) is a Tsallis-entropy-style regularizer, and \(\alpha\) controls sparsity. The relevant lesson for QASSMax is not sparsity itself, because QASSMax still uses softmax. The relevant lesson is that attention normalization can benefit from a learned function of context length. ASEntmax uses a scaling form
$$
\delta+\beta(\log n)^\gamma.
$$
Here \(\delta\) is a length-independent offset, while \(\beta\) and \(\gamma\) are input-dependent quantities in ASEntmax. QASSMax generalizes the length-scaling part through \(\text{MLP}_\text{base}(\log n)\).

The gate \(G_h(q_h)\) adds query awareness. This follows the principle behind selective attention: not every query should have the same attention sharpness. Some queries need to retrieve a highly specific row; others need to aggregate information more broadly. In a query-dependent temperature formulation, query row \(r\) might use
$$
a_{rj}
=
\frac{\exp(z_{rj}/\tau_r)}
{\sum_{\ell=1}^{N}\exp(z_{r\ell}/\tau_r)}.
$$
Here \(r\) indexes the query row or query token, \(z_{rj}\) is the attention logit between query \(r\) and key \(j\), \(a_{rj}\) is the resulting attention weight, and \(\tau_r>0\) is the temperature assigned to that query. QASSMax implements a related idea with a bounded element-wise gate. Because
$$
G_h(q_h)\in(0,2)^{d_\text{head}},
$$
the gate can reduce or increase the base scale, but it cannot grow without limit. This keeps query-specific modulation from overwhelming the length-dependent trend.

The design is also related to gated attention. A generic gate applies a learned multiplicative control:
$$
\tilde{u}=g\odot u.
$$
Here \(u\) is a vector being modulated, \(g\) is a learned gate with the same dimension as \(u\), and \(\tilde{u}\) is the gated vector. Some gated attention mechanisms apply the gate after attention, for example
$$
\tilde{o}=g(q)\odot \text{Attn}(q,K,V).
$$
In this expression, \(K\) is the matrix of keys, \(V\) is the matrix of values, \(\text{Attn}(q,K,V)\) is the ordinary attention output for query \(q\), and \(\tilde{o}\) is the gated output. QASSMax applies the gate earlier, at the query-scaling stage. Because the gate changes \(\tilde{q}_h\), it changes the logits before softmax:
$$
\tilde{z}_j=
\frac{(q_h\odot B_h(n)\odot G_h(q_h))^\top k_j}
{\sqrt{d_\text{head}}}.
$$
So the gate affects the attention weights themselves, not only the post-attention output. Unlike the earlier scalar-temperature example, QASSMax is not generally equivalent to one scalar temperature per query, because \(B_h(n)\) and \(G_h(q_h)\) scale different head dimensions differently. The scalar-temperature view is still useful as intuition: QASSMax changes attention sharpness by changing the logits before softmax.

QASSMax was designed around four ideas. First, using \(\log n\) as the length variable counteracts the growth of the softmax denominator. Second, \(\text{MLP}_\text{base}(\log n)\) allows a learned context-length scaling law. Third, element-wise scaling is more expressive than a single per-head scalar. Fourth, bounded query-aware gating lets different queries adjust their attention sharpness without destabilizing the length scaling.

Applied to \(\text{TF}_\text{col}\) and \(\text{TF}_\text{icl}\), QASSMax improves long-context behavior. In the paper's needle-in-haystack classification task, the model must focus on one anchor sample among many negative samples. Without scalable softmax, attention entropy rises and accuracy drops as the number of negatives grows. QASSMax maintains low entropy and 100% accuracy even with 15K negatives, outperforming SSMax at extreme scales.

## Summary

Query-aware scalable softmax modifies attention logits by rescaling queries with both a context-length-dependent base term and a bounded query-dependent gate. This helps TabICLv2 keep attention sharp in long contexts while allowing different queries and latent dimensions to adjust the amount of scaling they need. The next post covers many-class classification, where TabICLv2 extends a model pretrained with at most 10 classes to settings with many more labels.
