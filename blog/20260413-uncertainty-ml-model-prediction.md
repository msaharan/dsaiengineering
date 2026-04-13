# Reporting ML model performance with confidence intervals

When reporting model performance, a single number (like accuracy) is not enough. We should also report the uncertainty on the performance to quantify the potential variation in the performance.

For this example, let's assume we are reporting classification accuracy, which is defined as the proportion of correct predictions from all predictions made.
$$
\text{classification accuracy} = \dfrac{\text{correct predictions}}{\text{total predictions  }}
$$


Here, classification error can quantify how prone a model is to making mistakes.
$$
\text{classification error} = \dfrac{\text{incorrect predictions}}{\text{total predictions  }}
$$


After the final model has been trained and finalized, it can be used to make predictions on the test dataset. Those predictions are used to calculate classification accuracy or classification error. Rather than presenting just a single error score, a confidence interval can be calculated and presented as part of the model performance.

The confidence interval is comprised of its range and probability. The range defines the lower and upper bounds within which the true model performance is expected to lie. The confidence level (e.g., 95%) indicates how often this method would produce intervals that contain the true performance, under repeated sampling.

The confidence interval can be calculated in various ways, where each has its own assumptions and conditions. In this example, we will talk about the Wald interval (normal approximation), which is a simple but less robust method compared to alternatives like the Wilson interval. It works well when the sample size is sufficiently large (e.g., when both $n$ and $n$ are not too small, $\geq5-10$ as a rule of thumb) and assumes the observations were drawn from the domain independently (e.g., they are independent and identically distributed). 

In this case, the confidence interval for classification error can be calculated as:
$$
\text{CI} = \text{error} \pm \text{const} \times \sqrt{(\text{error} \times (1 - \text{error}))/n)}
$$


where, $\text{error}$ is the classification error, $n$ is the number of observations used to evaluate the model, and $\text{const}$ is the constant for confidence interval:

- 1.64 (90%)
- 1.96 (95%)
- 2.33 (98%)
- 2.58 (99%)

Following is a worked out example:

Consider a model with an $\text{error}$ of 0.02 ($\text{error} = 0.02$) on a test set with 50 examples ($$).

We can calculate the 95% confidence interval (const = 1.96) as follows:

- $0.02 \pm 1.96 \times \sqrt{(0.02 \times (1 - 0.02)) / 50}$
- $0.02 \pm 1.96 \times \sqrt{0.0196 / 50}$
- $0.02 \pm 1.96 \times 0.0197$
- $0.02 \pm 0.0388$

Or, in other words, we are 95% confident that the interval [0.0, 0.0588] contains the true classification error of the model on unseen data. Note that the confidence intervals on the classification error must be clipped to the values 0.0 and 1.0. It is impossible to have a negative error (e.g. less than 0.0) or an error more than 1.0.

Reference: [Machine Learning Mastery](https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/?__cf_chl_tk=LAByLD8qIqQeNJXx_ZKLGXxwj.IaUZMYPERJw3cWr_4-1776082309-1.0.1.1-WvaTC3NnWdOamHRMAi0zhqfje1hUBJCwzSMR0Q_gwPQ).

---

[Mohit Saharan](https://linkedin.com/in/msaharan), 20260413, P#1. Reading an academic document that discussed the performance of a ML model inspired this post.