[Mohit Saharan](https://linkedin.com/in/msaharan), P25, 20260525
___
# [P25] SAM Platform: Orchestration + ML for Trading PyPI Libraries

Developing a production-oriented trading system on top of ML4T libraries.
___

In [P23](https://newsletter.dsaiengineering.com/p/p23-introducing-sam), I introduced SAM as a personal quantitative research and engineering platform to apply the ML and quantitative finance concepts discussed in the newsletter. In P23, I made SAM work to generate a simple daily brief. After that, I tried implementing in SAM the volatility-regime forecasting workflow discussed in [P19](https://newsletter.dsaiengineering.com/p/p19-volatility-regime-forecasting-tabpfn-tabicl-classical-ml-models-2), but I wasn't satisfied with the results. I felt I lacked the domain knowledge to make it work in a way that's realistic.

Yesterday, I [came across](https://www.linkedin.com/posts/msaharan_machinelearning-quantitativefinance-algorithmictrading-activity-7464368821412282368-mwld?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g) the [ML for Trading book](https://ml4trading.io/) by [Stefan Jansen](https://www.linkedin.com/in/applied-ai/) through a note posted by [him on Substack](http://substack.com/@ml4trading). It caught my attention because I had been looking for something like it to learn the field and was also trying to find coding resources to build SAM upon. I found both in this one.

The following picture shows a screenshot from the official website of the book. It covers much of the classical ML material that I was planning to discuss in this newsletter already in addition to tabular foundation models. Moreover, it covers many of the production and workflow pieces I wanted to build into SAM to make it a quant research and engineering platform for personal and educational use.

![Screenshot 2026-05-25 at 22.03.09](./20260525-sam-ml-for-trading.assets/Screenshot%202026-05-25%20at%2022.03.09.png)

The book also comes with several more add-ons that are highly relevant to today's AI-dominant workflows.

![Screenshot 2026-05-25 at 22.13.12](./20260525-sam-ml-for-trading.assets/Screenshot%202026-05-25%20at%2022.13.12.png)

The part I found immediately useful was the following set of [libraries](https://github.com/orgs/ml4t/repositories). The core libraries I looked at are MIT licensed and already contain a lot of wiring that I would otherwise have to do on my own, and I would certainly either get them wrong due to lack of domain knowledge or make a lot of mistakes.

![Screenshot 2026-05-25 at 22.55.43](./20260525-sam-ml-for-trading.assets/Screenshot%202026-05-25%20at%2022.55.43.png)

It's likely to be a good idea to not reinvent the wheel. So, today, I rebuilt SAM around ML4T libraries.

The implementation details will change as SAM develops further, but the architectural decision is the durable part. For now, it's enough to know that SAM is a thin orchestration layer on top of ML4T: SAM orchestrates; ML4T computes, validates, backtests, and executes. SAM owns the repeatable workflow layer: configuration, CLI routing, strategy selection, pipeline sequencing, manifests, promotion gates, and operator controls. ML4T owns the specialized trading machinery: market data access, feed specifications, feature engineering, signal diagnostics, model training surfaces, backtesting, live execution, broker wrappers, and risk guards.

If you are interested in looking at the code and playing with it, you can find today's version [here](https://github.com/msaharan/sam/tree/109c2ec15bb39fe3b8953dd480d78f3fdc23fe28). The current repo is still small, but the important change is visible in the structure: configs, pipelines, manifests, live runners, ops tools, and tests now sit around ML4T rather than replacing it. The screenshot below is SAM after the rebuild.

![Screenshot 2026-05-25 at 22.48.30](./20260525-sam-ml-for-trading.assets/Screenshot%202026-05-25%20at%2022.48.30.png)

This version is still early, but it gives SAM the right shape for further development. Check out the repo and let me know what you think.
