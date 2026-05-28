# Understanding Tabular Foundation models: the architecture of TabICLv2-4

Source: TabICLv2 paper. https://arxiv.org/pdf/2602.11139.

In the previous post, we covered how TabICLv2 compresses feature-level information into row representations and then performs in-context learning. In this post, we cover query-aware scalable softmax, the attention-scaling mechanism TabICLv2 uses to preserve selective attention as the number of context samples grows.

## Illustration and summary

The architecture of TabICLv2 is illustrated in the following figure. Here, given an input \(X\in\mathbb{R}^{n\times m}\), repeated feature grouping encodes columns into multigroups via circular shifts to break feature symmetries, and target-aware embedding injects target information from the beginning. \(\text{TF}_\text{col}\) embeds each feature through a set transformer, \(\text{TF}_\text{row}\) aggregates features into row representations \(h\), and  \(\text{TF}_\text{icl}\) performs in-context learning tomorrow predict test targets \(\hat{y}\). QASSMax (query-aware scalable softmaxx), is applied in part of  \(\text{TF}_\text{col}\) where inducing points aggregate input information and  \(\text{TF}_\text{icl}\) to mitigate attention fading and improve long-context generalization. 

The following subsections elaborate on the summary.

![Screenshot 2026-05-28 at 17.29.16](./20260528-understanding-tfm-architecture-of-tabiclv2-4.assets/Screenshot%202026-05-28%20at%2017.29.16.png)

## Query-aware scalable softmax

To improve generalization to larger datasets, the Scalable Softmax (SSMax, Nakanishi 2025) was extended, which is a temperature scaling method that sharpens attention distributions by rescaling queries before computing logits. Let \(q_h = (q_{hi})\) be a query vector at head \(h\) with head dimension indexed by \(i\), and let \(n\) be the size of the training set. SSMax rescales queries with a learnable per-head scalar $s_h$:
$$
\bar{q}_{hi} = q_{hi}\cdot s_h\log n.
$$
TabICLv2 proposes quesy-aware scalable softmax (QASSMax), which rescales each query element as:
$$
\bar{q}_{hi} = q_{hi}\underbrace{\cdot\text{MLP}_\text{base}(\log \,n)_{hi}}_\text{base scaling}\cdot\underbrace{(1+\tanh(\text{MLP}_\text{gate}(q_h)_i))}_\text{query-aware gating}
$$
where for \(H\) attention heads, \(\text{MLP}_\text{base}: \mathbb{R}\rightarrow\mathbb{R}^{H\times d_\text{head}}\) and \(\text{MLP}_\text{gate}:R^{d_\text{head}} \rightarrow \mathbb{R^{d_\text{head}}}\) are two-layer MLPs with 64 hidden neurons and GELU activation.

The QASSMax was designed based on the following rationale:

1. The \(\log n\) factor is critical as it counteracts the linear growth of the softmax denominator with respect to \(n\) (Nakanishi, 2025; Chen et al., 2025b); 
2. ASEntmax (Vastlenko et al., 2025) uses learnable \(\delta + \beta(\log n)^\gamma\), inspiring us to generalize to \(\text{MLP}_\text{base}(\log n)\);
3. Element-wise scaling increases expressiveness beyond per-head scalars;
4. Selective Attention (Zhang et al., 2024) introduces query-awareness in temperature scaling, which motivated in TabICLv2 the use of  bounded query-gating \(\in (0,2)\) that modulates the base scaling without dominating the \(\log n\) trend. In addition, the gating design of TabICLv2 shares similar insights with Gated Attention (Qiu et al., 2025), which applies gating to attention outputs and finds query-dependent, element-wise gating most effective.

QASSMax applied to \(\text{TF}_\text{col}\) and \(\text{TF}_\text{icl}\) yields substantial performance improvements. To study its effect on attention fading, a toy neddle-in-haystack classification task was designed, as shown below: the model must focus on a single anchor sample (the needle) among increasing negative samples (the haystack). Without scalable softmax, attention entropy rises and accuracy drops. However, QASSMax maintains low entropy and 100% accuracy even with 15K negatives, outperforming SSMax, which largely degrades at extreme scales.

## Summary

Query-aware scalable softmax modifies attention logits by rescaling queries with both a context-length-dependent base term and a bounded query-dependent gate. This helps TabICLv2 keep attention sharp in long contexts while allowing different queries and latent dimensions to adjust the amount of scaling they need.

#  Appendix

### Query-aware scalable softmax

In standard scaled dot-product attention, a query \(q\in\mathbb{R}^{d_\text{head}}\) is compared with keys \(k_1,\ldots,k_n\in\mathbb{R}^{d_\text{head}}\). The unnormalized attention logit for key \(j\) is
$$
z_j=\frac{q^\top k_j}{\sqrt{d_\text{head}}},
$$
and the attention weight is
$$
a_j=\text{softmax}(z)_j=\frac{\exp(z_j)}{\sum_{\ell=1}^{n}\exp(z_\ell)}.
$$
The output is the weighted average \(\sum_{j=1}^n a_jv_j\), where \(v_j\) is the value vector associated with key \(j\). In this view, QASSMax does not replace attention itself. It changes the logits before the softmax by rescaling the query. If \(\bar{q}=\alpha q\), then
$$
\bar{z}_j=\frac{\bar{q}^\top k_j}{\sqrt{d_\text{head}}}=\alpha z_j.
$$
Thus query scaling is mathematically equivalent to logit scaling. This is why the QASSMax formula can be understood as a learned, length-aware, query-aware temperature mechanism.

#### Temperature scaling

Temperature scaling modifies the sharpness of a softmax distribution. For a temperature \(\tau>0\),
$$
\text{softmax}_\tau(z)_j=\frac{\exp(z_j/\tau)}{\sum_{\ell=1}^{n}\exp(z_\ell/\tau)}.
$$
Lower temperature \(\tau<1\) sharpens the distribution: large logits become more dominant and small logits receive less probability mass. Higher temperature \(\tau>1\) flattens the distribution: probability mass is spread more evenly.

Equivalently, one may multiply logits by a scale \(\alpha=1/\tau\):
$$
\text{softmax}_\tau(z)=\text{softmax}(\alpha z).
$$
In attention, multiplying the query by \(\alpha\) multiplies every logit in that query row by \(\alpha\). This leaves the ranking of keys unchanged but changes how decisive the softmax is. The relative odds between keys \(j\) and \(\ell\) become
$$
\frac{a_j}{a_\ell}=\frac{\exp(\alpha z_j)}{\exp(\alpha z_\ell)}
=\exp(\alpha(z_j-z_\ell)).
$$
So scaling does not invent new similarities; it amplifies or dampens the similarities already encoded by \(q^\top k_j\).

#### Scalable Softmax (SSMax)

The attention fading problem appears when the number of candidate keys \(n\) grows while the logit gaps remain roughly fixed. Suppose one relevant key has logit \(z_\star\), and for simplicity the other \(n-1\) keys have a lower logit \(z_0\). Let \(\Delta=z_\star-z_0>0\). Under ordinary softmax,
$$
a_\star
=\frac{\exp(z_\star)}{\exp(z_\star)+(n-1)\exp(z_0)}
=\frac{1}{1+(n-1)\exp(-\Delta)}.
$$
For fixed \(\Delta\), \(a_\star\rightarrow 0\) as \(n\rightarrow\infty\). This is the basic denominator effect: even if each distractor is individually less relevant, many distractors can collectively absorb the attention mass.

Scalable Softmax addresses this by making the logit scale grow with context length. In the form used in the main text, SSMax rescales the query at head \(h\) by a learned scalar times \(\log n\):
$$
\bar{q}_h = q_h\cdot s_h\log n.
$$
This yields scaled logits \(\bar{z}_j=(s_h\log n)z_j\). In the simplified one-relevant-key setting,
$$
a_\star
=\frac{1}{1+(n-1)\exp(-s_h\Delta\log n)}
\approx
\frac{1}{1+n^{1-s_h\Delta}}.
$$
This expression shows the role of \(\log n\). To keep the relevant token visible as \(n\) grows, the relevant logit gap must effectively grow on the order of \(\log n\). SSMax builds that length dependence into the attention temperature.

QASSMax keeps this length-aware idea but generalizes the scaling. Instead of one scalar \(s_h\) per head, it uses an element-wise base scale
$$
B_h(n)=\text{MLP}_\text{base}(\log n)_h\in\mathbb{R}^{d_\text{head}},
$$
and a query-dependent gate
$$
G_h(q_h)=1+\tanh(\text{MLP}_\text{gate}(q_h))\in(0,2)^{d_\text{head}}.
$$
Using element-wise multiplication \(\odot\), the rescaled query is
$$
\bar{q}_h=q_h\odot B_h(n)\odot G_h(q_h).
$$
The base term handles the predictable effect of context length; the gate lets different queries and dimensions adjust that base scale without removing the dominant \(\log n\) dependence.

#### GELU activation

GELU stands for Gaussian Error Linear Unit. It is a smooth activation function commonly used in transformer MLPs:
$$
\text{GELU}(x)=x\Phi(x),
$$
where \(\Phi(x)\) is the cumulative distribution function of the standard normal distribution. A widely used approximation is
$$
\text{GELU}(x)\approx
\frac{x}{2}\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right)\right)\right).
$$
Compared with ReLU, which is exactly zero for \(x<0\), GELU smoothly downweights negative values instead of hard-thresholding them. In QASSMax, GELU is used inside the small MLPs that compute scaling factors. Its role is not specific to softmax; it gives those MLPs a smooth nonlinearity so they can learn nontrivial functions of \(\log n\) and \(q_h\).

#### ASEntmax

Entmax is a family of alternatives to softmax that can produce sparse probability distributions. Softmax assigns strictly positive probability to every key:
$$
\text{softmax}(z)_j>0 \quad \text{for all } j.
$$
Sparse attention mechanisms instead allow some attention weights to become exactly zero, so the model can ignore irrelevant keys more decisively.

One way to describe entmax is as a regularized optimization problem over the probability simplex
$$
\Delta^n=\left\{p\in\mathbb{R}^n:\sum_{j=1}^n p_j=1,\ p_j\geq0\right\}.
$$
At a high level, entmax chooses
$$
\text{entmax}_\alpha(z)
=\arg\max_{p\in\Delta^n}\left(p^\top z + H_\alpha(p)\right),
$$
where \(H_\alpha\) is a Tsallis-entropy-style regularizer. The parameter \(\alpha\) controls sparsity. The softmax case corresponds to the dense entropy-regularized limit, while larger \(\alpha\) values can yield exact zeros.

The main text mentions ASEntmax because it uses a learnable context-length-dependent form such as
$$
\delta+\beta(\log n)^\gamma.
$$
The relevant idea for QASSMax is not the sparse entmax transformation itself. The relevant idea is that attention normalization can benefit from a learned function of \(\log n\), rather than a fixed hand-chosen scale. QASSMax adopts this lesson through \(\text{MLP}_\text{base}(\log n)\), while keeping the final normalization as softmax.

#### Selective attention

Selective attention refers to making the attention temperature depend on the current query, position, or token context. In ordinary attention, one softmax temperature is shared across all queries in a head. In a selective attention layer, query row \(r\) may have its own temperature \(\tau_r\):
$$
a_{rj}=
\frac{\exp(z_{rj}/\tau_r)}
{\sum_{\ell=1}^{n}\exp(z_{r\ell}/\tau_r)}.
$$
If \(\tau_r\) is small, query \(r\) attends selectively to a few high-logit keys. If \(\tau_r\) is large, the same query spreads attention more broadly. This is useful because not every query should have the same attention sharpness. Some examples require retrieving one highly specific row; others require aggregating evidence across many rows.

QASSMax uses the same broad principle: attention sharpness should depend on the query. Its gate
$$
G_h(q_h)=1+\tanh(\text{MLP}_\text{gate}(q_h))
$$
is query-aware and element-wise. The \(\tanh\) makes the gate bounded in \((0,2)\), so the gate can reduce or increase the base scale but cannot grow without limit. This matters because the model should adapt to the query while still respecting the length-scaling trend supplied by \(\text{MLP}_\text{base}(\log n)\).

#### Gated attention

A gate is a learned multiplicative control. Given a representation \(u\) and a gate \(g\), a gated representation often has the form
$$
\tilde{u}=g\odot u,
$$
where \(\odot\) denotes element-wise multiplication. If \(g_i\approx0\), dimension \(i\) is suppressed; if \(g_i\approx1\), it passes through; if the gate range permits values above \(1\), the dimension can be amplified.

In many gated attention variants, the gate is applied to attention outputs. For example, if \(\text{Attn}(q,K,V)\in\mathbb{R}^{d_\text{head}}\) is the usual attention output, one may compute
$$
\tilde{o}=g(q)\odot \text{Attn}(q,K,V).
$$
Here the attention distribution is computed first, and the gate modulates the resulting vector.

QASSMax applies the gating idea earlier, at the query-scaling stage:
$$
\bar{q}_h=q_h\odot B_h(n)\odot G_h(q_h).
$$
Because the gate changes \(\bar{q}_h\), it changes the logits before softmax:
$$
\bar{z}_j=\frac{(q_h\odot B_h(n)\odot G_h(q_h))^\top k_j}{\sqrt{d_\text{head}}}.
$$
This means the gate affects the attention weights themselves, not only the post-attention output. The design is query-dependent, element-wise, and bounded. Those three properties are important: query-dependent because different prediction contexts may need different sharpness; element-wise because different latent dimensions may encode different matching signals; bounded because the gate should modulate the length-aware base scale rather than overwhelm it.
