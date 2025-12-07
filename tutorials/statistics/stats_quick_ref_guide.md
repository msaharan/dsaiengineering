## Statistics Quick Reference Guide

###  How to Use This in Practice

When you’re about to compute a quantity, run this quick checklist:

1. **What question am I answering?** (center, spread, shape, relationship, uncertainty?)
2. **What assumptions does my statistic make?** (normality, linearity, independence, large n?)
3. **Do my data roughly meet those assumptions?** (if not, can I transform, bin, or choose a robust / non‑parametric alternative?)
4. **How will I explain this to a stakeholder?** (maybe report CI + effect size instead of only p‑values; median + percentiles instead of only mean)

## 1. Central Tendency

### 1.1 Mean (Average)

**What it answers:**
“What’s the typical value if I balance everything out?”

**Works well**

1. **Modeling sensor noise**

   * You average repeated temperature readings from a stable sensor. Errors are small and symmetric.
2. **Overall app engagement for similar users**

   * You compute mean daily active minutes for a homogeneous cohort (e.g., paying users in a single country, no huge whales).

**Breaks**

1. **Income distribution with a few ultra‑rich customers**

   * Mean customer revenue is dominated by a few big enterprise clients; it doesn’t represent most customers at all.
2. **Session duration with many 0’s**

   * Many users bounce immediately (0–5 sec) while a few stay for hours. Mean session time looks healthy but hides the bounce problem.

---

### 1.2 Median

**What it answers:**
“What value sits right in the middle if I sort everyone?”

**Works well**

1. **Typical home price**

   * Housing prices are skewed by luxury homes; median price describes the “typical” buyer better than the mean.
2. **Typical latency**

   * You want “usual” backend response time. Median latency is robust to rare but huge spikes.

**Breaks**

1. **Cost modeling for capacity planning**

   * Median request size won’t tell you about large requests that drive infrastructure cost; you need the tail.
2. **Highly multimodal data**

   * User ages clustered at 18–25 and 45–55; median (say 32) doesn’t represent any real group.

---

### 1.3 Mode

**What it answers:**
“What’s the most common category or value?”

**Works well**

1. **Most common device type**

   * Mode of device (mobile / desktop / tablet) is meaningful for prioritizing UI work.
2. **Most frequent product size**

   * Mode of clothing size (M, L, XL) helps inventory decisions.

**Breaks**

1. **Continuous data with no repeats**

   * Mode of floating‑point transaction amounts is meaningless; almost all values are unique.
2. **Flat or noisy categorical distribution**

   * Five categories each around 20%; mode is arbitrary and misleadingly “dominant”.

---

## 2. Spread / Variability

### 2.1 Range

**What it answers:**
“What’s the span from smallest to largest value?”

**Works well**

1. **Sanity checks on input**

   * Check that ages fall between 0 and 120; any violation flags data issues.
2. **Spec limits for hardware**

   * Range of temperature or voltage must stay within design bounds.

**Breaks**

1. **Single outlier dominates**

   * One corrupted log entry with latency = 1e9 ms makes range useless as a summary.
2. **Comparing variability across groups**

   * Two samples with the same range but very different internal distributions; range says they’re “equally variable” when they’re not.

---

### 2.2 Variance & Standard Deviation (SD)

**What they answer:**
“How much do values spread around the mean?”
Variance = average squared deviation; SD = its square root (same units as data).

**Works well**

1. **Gaussian‑like measurement noise**

   * Modeling error terms in linear regression; SD is a natural scale.
2. **Stable process control**

   * Monitoring production metrics (e.g., weight of packaged goods) where deviations cluster around a stable mean.

**Breaks**

1. **Heavy‑tailed distributions**

   * SD of transaction size in fraud‑prone data is dominated by rare huge values; doesn’t reflect “typical” variation.
2. **Non‑centered or multi‑peak distributions**

   * Bimodal distribution of user activity (night owls vs day users); SD is hard to interpret.

---

### 2.3 Quartiles, Percentiles & IQR

* **Quartiles / percentiles:** values at specific ranks (25th, 50th, 75th, 95th, …).
* **IQR = Q3 − Q1:** spread of the middle 50%.

**Works well**

1. **Latency SLOs**

   * 95th / 99th percentile latency is directly relevant to user experience; IQR summarizes typical variability.
2. **Robust spread in skewed data**

   * Comparing wage distributions across regions; IQR less affected by extreme salaries.

**Breaks**

1. **Very small samples**

   * With n = 8 customers, your 95th percentile is unstable; small changes in data flip it dramatically.
2. **When you need model‑based variance**

   * For parametric models assuming normal errors, percentiles/IQR are less useful than SD in fitting and diagnostics.

---

## 3. Shape of Distribution

### 3.1 Skewness

**What it answers:**
“Is the distribution lopsided? Which tail is longer?”

**Works well**

1. **Deciding transformations**

   * Detecting positive skew in revenue to justify log transforms before regression.
2. **Risk analysis**

   * Strong right skew in claim size indicates rare but costly events needing special handling.

**Breaks**

1. **Heavily multimodal data**

   * Two clear peaks can produce low skewness but still be complex; skewness misses this.
2. **Outlier‑driven skew**

   * A few data‑entry errors create huge skew; skewness reflects bad data, not true shape.

---

### 3.2 Kurtosis

**What it answers:**
“How heavy‑tailed or peak‑y is the distribution compared to normal?”

**Works well**

1. **Detecting fat tails in finance**

   * High kurtosis in returns suggests more extreme moves than a normal model would predict.
2. **Outlier sensitivity check**

   * Comparing kurtosis across manufacturing lines to see where extreme deviations are more common.

**Breaks**

1. **Small sample kurtosis**

   * With n = 30, kurtosis is very noisy; interpretations are unreliable.
2. **Over‑interpreting for non‑normal shapes**

   * Weird shapes (e.g., uniform, box‑shaped) can give kurtosis values that are hard to translate into practical meaning.

---

## 4. Standardized Scores & Test Statistics

### 4.1 z‑Score (for reference)

**What it answers:**
“How many standard deviations is this value from the mean?”

**Works well**

1. **Anomaly detection in near‑normal metrics**

   * CPU usage or temperature around a stable mean; z > 3 suggests anomalies.
2. **Standardizing features**

   * Scaling continuous features to mean 0, SD 1 for many ML models.

**Breaks**

1. **Non‑normal, heavy‑tailed data**

   * For power‑law traffic spikes, “3 SDs away” isn’t reliably “rare”.
2. **Very small n using sample SD**

   * z‑scores based on n = 10 are shaky; sampling error in SD is large.

---

### 4.2 t‑Score / t‑Statistic

**What it answers:**
“Given sample data and estimated SD, how extreme is our observed difference/mean relative to the null?”

**Works well**

1. **A/B test with moderate n**

   * Comparing mean conversion rates for 2 variants with ~100–500 users each.
2. **Estimated mean of a small sample with unknown population SD**

   * Average time‑to‑resolve for 20 support tickets.

**Breaks**

1. **Highly non‑normal with tiny samples**

   * Response times with severe skew and n = 10; t‑test assumptions are badly violated.
2. **Binary or ordinal outcomes**

   * t‑test on “yes/no” outcomes is inferior to logistic regression or proportion tests.

---

### 4.3 Percentile Rank

**What it answers:**
“For a value x, what percentage of observations are ≤ x?”

**Works well**

1. **Scoring model predictions vs a population**

   * A user’s risk score is at the 95th percentile → high risk relative to others.
2. **Customer ranking**

   * Revenue percentile for each customer to target top 5% by spend.

**Breaks**

1. **Comparing across distributions with different shapes**

   * 90th percentile engagement in region A vs B; underlying distributions differ, percentiles alone can mislead.
2. **Modeling raw relationships**

   * Replacing a continuous variable with percentile rank can destroy meaningful scale information for regression.

---

### 4.4 Scaled Scores (T‑Score, Stanine, etc.)

**What they answer:**
“Where is this observation on a standardized reporting scale (e.g., mean 50 SD 10, or 1–9 stanines)?”

**Works well**

1. **Reporting to non‑technical stakeholders**

   * Presenting user satisfaction scores as stanines (“Your team is in band 8/9”).
2. **Combining scores from different tests**

   * Convert raw scores to a common scale, then compare or aggregate.

**Breaks**

1. **Upstream modeling**

   * Using stanines instead of raw values in regression loses information.
2. **Non‑normal underlying scores**

   * If raw scores are highly skewed, scaled scores can give a false impression of equal intervals.

---

### 4.5 χ² (Chi‑Square) Statistic

**What it answers:**
“How far is the observed frequency table from what we would expect under the null?”

**Works well**

1. **Feature vs label independence**

   * Checking if device type is associated with churn (categorical × categorical).
2. **Goodness‑of‑fit for discrete distributions**

   * Testing if click counts by hour match a Poisson model.

**Breaks**

1. **Low expected counts**

   * Many cells with expected < 5 → χ² approximation is poor; need exact tests or regrouping.
2. **Strong dependence with huge n**

   * With millions of rows, χ² is almost always “significant,” even for trivial effects; p‑value becomes less informative.

---

### 4.6 F Statistic

**What it answers:**
“Is the variance explained by a model or group differences big relative to residual noise?”

**Works well**

1. **ANOVA across multiple variants**

   * Comparing mean engagement for A/B/C/D variants in an experiment.
2. **Overall regression significance**

   * Testing if your set of predictors jointly explains meaningful variation vs a null model.

**Breaks**

1. **Highly non‑normal residuals with small n**

   * Heavy‑tailed errors violate assumptions; F‑test results can mislead.
2. **Comparing many irrelevant predictors**

   * With lots of noisy features, F can be significant just by chance (multiple testing issue).

---

## 5. Association & Regression

### 5.1 Correlation Coefficient (Pearson r)

**What it answers:**
“How strong and what direction is the *linear* relationship between two variables?”

**Works well**

1. **Linear relationship in continuous data**

   * Height vs weight in a population; roughly linear, correlation makes sense.
2. **Feature pre‑screening**

   * Checking correlation between a feature and target as a quick filter.

**Breaks**

1. **Non‑linear relationships**

   * U‑shaped relationship between age and app usage; r ≈ 0 but strong pattern exists.
2. **Outlier‑sensitive**

   * One extreme point can inflate r from 0.1 to 0.8; robust correlation or visualization is needed.

---

### 5.2 Coefficient of Determination (R²)

**What it answers:**
“What fraction of variance in the target does our model explain?”

**Works well**

1. **Simple regression model comparison**

   * Comparing linear models predicting house price from size, then from size+location.
2. **Model fit in continuous outcome regression**

   * Tracking improvement across model iterations for a regression task.

**Breaks**

1. **Non‑linear / mis‑specified models**

   * A linear model on truly non‑linear data may still have a “decent” R² but miss structure.
2. **Classification problems**

   * R² has analogs (pseudo‑R²), but plain R² on probabilities or logits is often misinterpreted.

---

### 5.3 Regression Coefficients (β’s, including Intercept)

**What they answer:**
“How does the expected outcome change when a predictor increases by one unit (holding others fixed)?”

**Works well**

1. **Interpretable linear relationships**

   * In linear regression, “+1 bedroom increases price by $X on average, holding sqft constant.”
2. **Logistic regression log‑odds interpretation**

   * “+1 in risk score multiplies odds of churn by exp(β).”

**Breaks**

1. **Highly collinear predictors**

   * With correlated features (e.g., sqft and number of rooms), coefficients become unstable and hard to interpret.
2. **Non‑linear effects**

   * Effect of age on churn is not linear; a single β for age is misleading.

---

### 5.4 Residuals

**What they answer:**
“For each observation, how far is the predicted value from the actual value?”

**Works well**

1. **Model diagnostics**

   * Residual vs fitted plots to detect non‑linearity or heteroskedasticity.
2. **Error analysis**

   * Inspecting large residuals to find patterns where the model underperforms (e.g., certain user segments).

**Breaks**

1. **Using raw residuals as a feature without care**

   * Feeding residuals back into the same model can leak information or create circularity.
2. **Ignoring structure in residuals**

   * Time‑series with autocorrelated residuals; treating them as independent errors is wrong.

---

## 6. Inference & Effect Size

### 6.1 p‑Value

**What it answers (precisely):**
“Under the null hypothesis, what is the probability of observing data this extreme or more?”

**Works well**

1. **Screening for promising effects**

   * In controlled A/B tests, p‑values flag variants worth deeper look.
2. **Comparing with pre‑defined thresholds**

   * Product team agrees “we roll out only if p < 0.01”; p‑value operationalizes that.

**Breaks**

1. **Interpreted as P(H₀ is true)**

   * p = 0.04 ≠ “4% chance the null is true”; it’s about data rarity under H₀.
2. **Fishing / multiple testing**

   * Running hundreds of tests and reporting only significant ones gives meaningless p‑values without correction.

---

### 6.2 Confidence Interval (CI)

**What it answers (practical view):**
“A range of plausible values for a parameter, given data and model assumptions.”

**Works well**

1. **Reporting A/B test outcomes**

   * “Variant B improves conversion by 1.2% (95% CI: 0.5% to 1.9%).” Stakeholders see effect size and uncertainty.
2. **Communicating estimates to non‑statisticians**

   * CIs around forecasts provide understandable uncertainty bounds.

**Breaks**

1. **Misread as a probability about the parameter**

   * “95% chance the true value lies in this interval” is not strictly correct (frequentist view).
2. **Model mis‑specification**

   * CIs from a wrong model (e.g., ignoring clustering or time dependence) give over‑confident ranges.

---

### 6.3 Effect Size (e.g., Cohen’s d)

**What it answers:**
“How big is the difference, in standardized units, independent of sample size?”

* Cohen’s d ≈ (mean₁ − mean₂) / pooled SD.

**Works well**

1. **Comparing strength of effects**

   * Evaluating multiple experiments: which change had the largest meaningful impact?
2. **Planning sample size**

   * You assume a target effect size to calculate needed n for future experiments.

**Breaks**

1. **Non‑normal or highly skewed outcomes**

   * d relies on SD; skewed distributions make interpretation tricky.
2. **Different SDs across groups**

   * Heterogeneous variances can distort standardized difference.

---

### 6.4 Odds Ratio & Relative Risk

* **Odds ratio (OR):** ratio of odds of an event between groups.
* **Relative risk (RR):** ratio of probabilities between groups.

**Works well**

1. **Logistic regression interpretation**

   * Each unit change in predictor multiplies odds by exp(β); OR is the natural effect measure.
2. **Risk communication when baseline risk known**

   * “Treatment halves risk of churn” (RR = 0.5).

**Breaks**

1. **Confusing odds with probabilities**

   * OR of 4 at low baseline risk is modest; at high baseline risk it can look huge if misread as a 4× probability.
2. **When outcome is common**

   * OR deviates strongly from RR; OR can exaggerate effect compared to risk ratio.

