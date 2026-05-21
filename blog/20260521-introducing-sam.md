[Mohit Saharan](https://linkedin.com/in/msaharan), P23, 20260521

---

Over the past month, I developed the series into something I wanted: an engineering-first exploration of the frontier of tabular ML with a focus on quantitative finance for real-world use-cases. 

In my posts, I tried my best to make them as close to the reality as possible because I am interested in applying these lessons, conceptually or programming-wise, to real-world problems. 

I am happy with the depth my posts are covering, and I have several more planned, but I took a strategic break after P22 to reflect on the post and code quality to improve future posts. One of the things I wanted to do was to extract the reusable concepts and save them in a self-contained page for future reference.

But concepts by themselves are of limited use. That led me to think that it would better to have a place where I could exercise them regularly and solve a real problem at the same time. This week I took a step towards that and initiated the development of a quantitative research and engineering platform to analyze financial markets continuously in an increasingly sophisticated manner over time. I am calling it SAM, and it’s available with Apache-2.0 license.

![Screenshot 2026-05-21 at 10.53.03](./20260521-introducing-sam.assets/Screenshot%202026-05-21%20at%2010.53.03.png)

> README:
>
> SAM is an open-core personal quant research and engineering platform and is a companion to the [DSAIEngineering Newsletter](https://newsletter.dsaiengineering.com/). The workflows and primitives described in the newsletter are implemented in SAM. Currently, it focuses on production-style US-listed ETF allocation, volatility/risk scoring, and future US cross-sectional equity ranking workflows. More functionality will be integrated from the newsletter into SAM to make it more capable over time.

The idea for creating this platform has been on my mind since last year. I always envisioned to use it in combination with Perplexity Finance such that this platform could do comprehensive data analysis, modeling, simulations, etc., Perplexity Finance could do comprehensive market and geopolitical analysis, and together they could be used to develop an investment/trading strategies that a person could test with paper money to learn the process and test their understanding of the field.  

Currently, SAM is capable of generating a report daily with the following contents:

```markdown
# Daily SAM Brief - 2026-05-21

Research only. Not investment advice.

## Data Freshness and Validation

- Price last date: 2026-05-20
- Data freshness days: 1
- Validation failures: 0
- Model status: reused

## SPY Risk Regime

- Risk level: normal
- Realized volatility 20d: 0.1058
- Threshold: 0.2074
- Volatility / threshold: 0.5103
- Drawdown 20d: -0.0106
- MA distance 20d: 0.0165

## ETF Ranking

| score_rank | symbol | prediction | selected |
| --- | --- | --- | --- |
| 1 | SLV | 0.1835 | True |
| 2 | GLD | 0.0254 | True |
| 3 | SPY | 0.0237 | True |
| 4 | TLT | 0.0200 | True |
| 5 | EFA | 0.0187 | True |
| 6 | XLU | 0.0164 | False |
| 7 | DIA | 0.0146 | False |
| 8 | XLF | 0.0126 | False |
| 9 | LQD | 0.0123 | False |
| 10 | HYG | 0.0098 | False |

## Target Research Weights

| symbol | weight | score | score_rank |
| --- | --- | --- | --- |
| SLV | 0.2000 | 0.1835 | 1 |
| GLD | 0.2000 | 0.0254 | 2 |
| SPY | 0.2000 | 0.0237 | 3 |
| TLT | 0.2000 | 0.0200 | 4 |
| EFA | 0.2000 | 0.0187 | 5 |

## Turnover and Cost Diagnostics

- Previous weight date: 2026-05-20 00:00:00
- Turnover: 0.4000
- Estimated cost: 0.0002
- Gross exposure: 1.0000
- Net exposure: 1.0000
- Turnover breach: False

## Limitations

- Public market data can be revised and may differ from institutional data.
- Costs are simple basis-point estimates, not a market-impact model.
- The allocation view is a daily research snapshot for a monthly-horizon ETF workflow.
- TabPFN and TabICL are not required for this CPU-first daily brief.

```

Over time, I want it to do more sophisticated analyses using traditional quantitative finance methods and ML methods.

If you are interested in playing with it, I invite you to check out the repository: https://github.com/msaharan/sam, and let me know what you think.
