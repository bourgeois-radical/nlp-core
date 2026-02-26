# Experimentation & Causal Inference for GenAI

> **Purpose**: Design, run, and analyze experiments to prove GenAI features work  
> **Key Insight**: "A/B testing is the gold standard, but with GenAI's stochastic nature, you need additional techniques to get reliable signals"

## Table of Contents
1. [The Experimentation Challenge in GenAI](#1-the-experimentation-challenge-in-genai)
2. [A/B Testing Fundamentals](#2-ab-testing-fundamentals)
3. [Statistical Foundations](#3-statistical-foundations)
4. [Variance Reduction Techniques](#4-variance-reduction-techniques)
5. [Causal Inference Without Randomization](#5-causal-inference-without-randomization)
6. [GenAI-Specific Experiment Design](#6-genai-specific-experiment-design)
7. [Common Pitfalls and How to Avoid Them](#7-common-pitfalls-and-how-to-avoid-them)
8. [Your Interview Story: Fixing the A/B Pipeline](#8-your-interview-story-fixing-the-ab-pipeline)

---

## 1. The Experimentation Challenge in GenAI

### Why GenAI Experiments Are Different

Traditional A/B tests compare deterministic systems: button A vs button B, algorithm X vs algorithm Y. The output is the same every time for the same input. GenAI introduces **stochasticity** — the same prompt can produce different outputs on different runs.

```
┌─────────────────────────────────────────────────────────────────────┐
│                 TRADITIONAL A/B TEST                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: "Show me hotels in Paris"                                   │
│                                                                     │
│  Control: [Algorithm A] → Always returns: Hotel 1, Hotel 2, Hotel 3 │
│  Treatment: [Algorithm B] → Always returns: Hotel 2, Hotel 1, Hotel 4│
│                                                                     │
│  Same input → Same output → Clean comparison                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    GENAI A/B TEST                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: "Show me hotels in Paris"                                   │
│                                                                     │
│  Control: [GPT-4] → Run 1: "Here are top hotels..."                 │
│                   → Run 2: "I found these options..."               │
│                   → Run 3: "For Paris, I recommend..."              │
│                                                                     │
│  Treatment: [Claude] → Run 1: "Paris has wonderful hotels..."       │
│                      → Run 2: "Let me show you the best..."         │
│                      → Run 3: "Here are my recommendations..."      │
│                                                                     │
│  Same input → Different outputs → Noisy comparison                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The idealo Context

At idealo, you're evaluating GenAI features for a **price comparison platform**. The metrics that matter are:

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Click-through to merchant (CTR)** | User engagement | Primary conversion signal |
| **Purchase completion** | End-to-end success | Ultimate business goal |
| **Return visits** | Long-term satisfaction | User retention |
| **Query reformulation rate** | Search quality | Lower is better |
| **Time to first click** | UX efficiency | Faster is better |

### GitHub's Insight on Inverse Relationships

From GitHub's Copilot evaluation:

> "There are sometimes inverse relationships between metrics. Higher latency might actually lead to a higher acceptance rate because users might see fewer suggestions total."

This is crucial: **don't optimize a single metric in isolation**. A GenAI feature that increases CTR by 5% but doubles page load time might hurt overall conversion.

---

## 2. A/B Testing Fundamentals

### The Basic Setup

An A/B test splits users into two groups:
- **Control (A)**: Sees the existing experience
- **Treatment (B)**: Sees the new GenAI feature

The goal: determine if the treatment causes a statistically significant improvement in your target metric.

### Randomization Unit

**Critical decision**: What gets randomized?

| Unit | Pros | Cons | Use When |
|------|------|------|----------|
| **Request-level** | Maximum sample size | Same user sees both variants → inconsistent experience | Never for UX tests |
| **Session-level** | Good balance | User may see different variants across sessions | Short-term features |
| **User-level** | Consistent experience | Smaller effective sample size | Most A/B tests ✓ |
| **Geo/Time-level** | Simple implementation | Confounders (region differences) | When user-level impossible |

**idealo recommendation**: User-level randomization. A user searching for "best price iPhone 15" should consistently see either the GenAI-enhanced results or the standard results, not a mix.

### Assignment Mechanism

```python
import hashlib

def assign_variant(user_id: str, experiment_id: str, traffic_fraction: float = 0.5) -> str:
    """
    Deterministic assignment based on user_id hash.
    Same user always gets same variant.
    """
    hash_input = f"{user_id}:{experiment_id}"
    hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    
    # Normalize to [0, 1)
    normalized = hash_value / (2 ** 128)
    
    if normalized < traffic_fraction:
        return "treatment"
    else:
        return "control"
```

### Sample Size Calculation

Before running an experiment, calculate the required sample size:

**Minimum Detectable Effect (MDE):**

$$n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \sigma^2}{\delta^2}$$

Where:
- $n$ = sample size per group
- $z_{\alpha/2}$ = z-score for significance level (1.96 for 95% confidence)
- $z_{\beta}$ = z-score for power (0.84 for 80% power)
- $\sigma^2$ = variance of the metric
- $\delta$ = minimum detectable effect

**Practical Example:**

```python
from scipy import stats
import numpy as np

def sample_size_calculator(
    baseline_rate: float,
    mde_relative: float,  # e.g., 0.05 for 5% lift
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Calculate required sample size for a conversion rate experiment.
    """
    p1 = baseline_rate  # Control conversion rate
    p2 = baseline_rate * (1 + mde_relative)  # Expected treatment rate
    
    # Pooled variance
    p_pooled = (p1 + p2) / 2
    variance = 2 * p_pooled * (1 - p_pooled)
    
    # Effect size
    effect = abs(p2 - p1)
    
    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    # Sample size per group
    n = (z_alpha + z_beta) ** 2 * variance / (effect ** 2)
    
    return int(np.ceil(n))

# Example: idealo wants to detect 5% lift in CTR (baseline 10%)
n = sample_size_calculator(baseline_rate=0.10, mde_relative=0.05)
print(f"Required sample size per group: {n:,}")
# Output: Required sample size per group: 31,234
```

---

## 3. Statistical Foundations

### Hypothesis Testing Framework

**Null Hypothesis ($H_0$)**: Treatment has no effect  
$$\mu_T - \mu_C = 0$$

**Alternative Hypothesis ($H_1$)**: Treatment has an effect  
$$\mu_T - \mu_C \neq 0$$

### The t-Test for A/B Results

For comparing two means:

$$t = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}}}$$

Where:
- $\bar{X}_T, \bar{X}_C$ = sample means
- $s_T^2, s_C^2$ = sample variances
- $n_T, n_C$ = sample sizes

**Python Implementation:**

```python
from scipy import stats
import numpy as np

def analyze_ab_test(control: np.ndarray, treatment: np.ndarray) -> dict:
    """
    Complete A/B test analysis with effect size and confidence intervals.
    """
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(control, treatment)
    
    # Means and standard errors
    mean_c, mean_t = control.mean(), treatment.mean()
    se_c = control.std() / np.sqrt(len(control))
    se_t = treatment.std() / np.sqrt(len(treatment))
    
    # Lift
    lift = (mean_t - mean_c) / mean_c
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((control.var() + treatment.var()) / 2)
    cohens_d = (mean_t - mean_c) / pooled_std
    
    # 95% Confidence interval for the difference
    se_diff = np.sqrt(se_c**2 + se_t**2)
    diff = mean_t - mean_c
    ci_lower = diff - 1.96 * se_diff
    ci_upper = diff + 1.96 * se_diff
    
    return {
        'control_mean': mean_c,
        'treatment_mean': mean_t,
        'lift': lift,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'ci_95': (ci_lower, ci_upper)
    }
```

### Effect Size Interpretation

Cohen's d provides standardized effect size:

| Cohen's d | Interpretation |
|-----------|----------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

**Note**: In tech A/B tests, a 2-3% lift in conversion is often significant even with small Cohen's d because of large sample sizes.

### Multiple Testing Correction

When running multiple comparisons (e.g., testing CTR, conversion rate, and time on page simultaneously), you risk **false discoveries**.

**Bonferroni Correction (Conservative):**

$$\alpha_{adjusted} = \frac{\alpha}{m}$$

Where $m$ = number of comparisons.

**Benjamini-Hochberg (FDR Control):**

```python
from scipy.stats import false_discovery_control

def adjust_pvalues_bh(p_values: list) -> np.ndarray:
    """
    Benjamini-Hochberg procedure for FDR control.
    """
    return false_discovery_control(p_values, method='bh')

# Example
p_values = [0.001, 0.008, 0.039, 0.041, 0.042]
adjusted = adjust_pvalues_bh(p_values)
print(f"Original: {p_values}")
print(f"Adjusted: {adjusted}")
```

---

## 4. Variance Reduction Techniques

### Why Variance Reduction Matters

Higher variance → larger sample sizes → longer experiments → slower iteration.

From DoorDash's AutoEval:
> "AutoEval has reduced relevance judgment turnaround time by 98% compared to human evaluation, unlocking a nine-fold increase in capacity."

Variance reduction achieves similar speedups for A/B tests.

### CUPED (Controlled-experiment Using Pre-Experiment Data)

The most powerful variance reduction technique, used by Microsoft, Netflix, Booking.com.

**Core Idea**: Use pre-experiment data to predict what the metric *would have been* without treatment, then analyze residuals.

**Mathematical Foundation:**

Let $Y$ be the metric of interest, and $X$ be a covariate measured before the experiment.

The CUPED-adjusted metric:

$$\hat{Y}_{CUPED} = Y - \theta (X - \bar{X})$$

Where:

$$\theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

**Variance Reduction Factor:**

$$\text{Var}(\hat{Y}_{CUPED}) = \text{Var}(Y)(1 - \rho^2)$$

Where $\rho$ is the correlation between $X$ and $Y$.

**Practical Implication**: If pre-experiment behavior ($X$) is 70% correlated with post-experiment behavior ($Y$), variance is reduced by $1 - 0.7^2 = 51\%$, effectively doubling your sample size.

**Implementation:**

```python
import numpy as np
from scipy import stats

def cuped_adjustment(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    x_control: np.ndarray,  # Pre-experiment metric for control
    x_treatment: np.ndarray  # Pre-experiment metric for treatment
) -> tuple:
    """
    Apply CUPED adjustment to reduce variance.
    
    y = post-experiment metric
    x = pre-experiment covariate (e.g., last month's CTR)
    """
    # Combine for theta estimation
    y_all = np.concatenate([y_control, y_treatment])
    x_all = np.concatenate([x_control, x_treatment])
    
    # Calculate theta (regression coefficient)
    cov_xy = np.cov(x_all, y_all)[0, 1]
    var_x = np.var(x_all)
    theta = cov_xy / var_x
    
    # Global mean of covariate
    x_bar = x_all.mean()
    
    # Adjusted metrics
    y_control_adj = y_control - theta * (x_control - x_bar)
    y_treatment_adj = y_treatment - theta * (x_treatment - x_bar)
    
    # Compare variance reduction
    original_var = y_all.var()
    adjusted_var = np.concatenate([y_control_adj, y_treatment_adj]).var()
    reduction = 1 - (adjusted_var / original_var)
    
    print(f"Variance reduced by: {reduction:.1%}")
    
    return y_control_adj, y_treatment_adj
```

### Stratified Sampling

Group users by known characteristics, analyze within strata.

```python
def stratified_analysis(df, treatment_col, metric_col, strata_col):
    """
    Analyze A/B test results within strata, then combine.
    """
    results = []
    
    for stratum in df[strata_col].unique():
        stratum_df = df[df[strata_col] == stratum]
        
        control = stratum_df[stratum_df[treatment_col] == 0][metric_col]
        treatment = stratum_df[stratum_df[treatment_col] == 1][metric_col]
        
        effect = treatment.mean() - control.mean()
        weight = len(stratum_df) / len(df)
        
        results.append({
            'stratum': stratum,
            'effect': effect,
            'weight': weight,
            'n': len(stratum_df)
        })
    
    # Weighted average effect
    weighted_effect = sum(r['effect'] * r['weight'] for r in results)
    
    return weighted_effect, results
```

---

## 5. Causal Inference Without Randomization

Sometimes you can't run a proper A/B test:
- Feature already shipped (need post-hoc analysis)
- Business refuses to hold out control group
- Technical constraints prevent randomization

### Difference-in-Differences (DiD)

Compare changes over time between treatment and control groups.

**Setup:**
- **Treatment Group**: Users who saw the new GenAI feature
- **Control Group**: Users who didn't (different region, time period, etc.)
- **Before Period**: Before feature launch
- **After Period**: After feature launch

**The DiD Estimator:**

$$\hat{\tau}_{DiD} = (\bar{Y}_{T,after} - \bar{Y}_{T,before}) - (\bar{Y}_{C,after} - \bar{Y}_{C,before})$$

This removes:
- Time trends (both groups experience them)
- Group differences (captured in the "before" period)

```python
import pandas as pd
import statsmodels.formula.api as smf

def difference_in_differences(df):
    """
    Estimate treatment effect using DiD.
    
    df should have columns: metric, treatment (0/1), post (0/1)
    """
    # Interaction term captures the DiD effect
    model = smf.ols('metric ~ treatment * post', data=df).fit()
    
    # The coefficient on treatment:post is the DiD estimate
    did_effect = model.params['treatment:post']
    did_se = model.bse['treatment:post']
    did_pvalue = model.pvalues['treatment:post']
    
    return {
        'did_effect': did_effect,
        'std_error': did_se,
        'p_value': did_pvalue,
        'ci_95': (did_effect - 1.96*did_se, did_effect + 1.96*did_se)
    }
```

**Key Assumption**: Parallel trends — in the absence of treatment, both groups would have evolved similarly.

### Propensity Score Matching

When treatment assignment is non-random, match treatment users to similar control users.

**Propensity Score:**

$$e(X) = P(\text{Treatment} = 1 | X)$$

The probability of receiving treatment given covariates $X$.

**Process:**
1. Fit a logistic regression to predict treatment assignment
2. For each treated user, find control user(s) with similar propensity score
3. Analyze outcomes among matched pairs

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np

def propensity_score_matching(df, treatment_col, covariates, metric_col, n_neighbors=1):
    """
    Match treatment users to control users with similar propensity scores.
    """
    X = df[covariates]
    treatment = df[treatment_col]
    
    # Fit propensity model
    ps_model = LogisticRegression()
    ps_model.fit(X, treatment)
    propensity_scores = ps_model.predict_proba(X)[:, 1]
    
    df = df.copy()
    df['propensity'] = propensity_scores
    
    # Separate treatment and control
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    # Match using nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(control[['propensity']])
    
    matched_indices = []
    for idx, row in treated.iterrows():
        distances, indices = nn.kneighbors([[row['propensity']]])
        matched_indices.extend(control.iloc[indices[0]].index.tolist())
    
    # Create matched dataset
    matched_control = control.loc[matched_indices]
    
    # Compare outcomes
    ate = treated[metric_col].mean() - matched_control[metric_col].mean()
    
    return {
        'ate': ate,
        'n_treated': len(treated),
        'n_matched': len(matched_control)
    }
```

### Regression Discontinuity

When treatment is assigned based on a threshold:
- Users with score > 50 see GenAI feature
- Users with score ≤ 50 don't

Compare users just above vs just below the threshold.

---

## 6. GenAI-Specific Experiment Design

### Handling Stochasticity

**Problem**: Same prompt → different outputs → noisy metrics.

**Solutions:**

**1. Fix Temperature to 0**
```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[...],
    temperature=0  # Deterministic output
)
```
Trade-off: Reproducible but potentially less creative outputs.

**2. Increase Sample Size**
Budget for 2-3x more samples than deterministic systems.

**3. Aggregate at Session Level**
Instead of evaluating individual responses, evaluate entire sessions.

### Paired Comparison Design

Run both systems on the **same inputs**, then compare outputs.

```python
def paired_evaluation(queries: list, model_a, model_b):
    """
    Evaluate models on identical inputs for cleaner comparison.
    """
    results = []
    
    for query in queries:
        # Same query to both models
        output_a = model_a.generate(query, temperature=0)
        output_b = model_b.generate(query, temperature=0)
        
        # Evaluate both
        score_a = evaluate(output_a)
        score_b = evaluate(output_b)
        
        results.append({
            'query': query,
            'score_a': score_a,
            'score_b': score_b,
            'diff': score_b - score_a
        })
    
    # Paired t-test (more powerful than independent t-test)
    diffs = [r['diff'] for r in results]
    t_stat, p_value = stats.ttest_1samp(diffs, 0)
    
    return {
        'mean_diff': np.mean(diffs),
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### Interleaving Experiments

Show results from both systems to the same user, measure which they prefer.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTERLEAVING DESIGN                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User searches: "best price for iPhone 15"                          │
│                                                                     │
│  Results shown (interleaved):                                       │
│  1. [Model A] Amazon - €849                                         │
│  2. [Model B] MediaMarkt - €859                                     │
│  3. [Model B] Saturn - €869                                         │
│  4. [Model A] eBay - €879                                           │
│  5. [Model A] Otto - €889                                           │
│                                                                     │
│  User clicks: Position 1 (Model A) and Position 3 (Model B)         │
│                                                                     │
│  Score: Model A = 1, Model B = 1                                    │
│                                                                     │
│  Aggregate across thousands of sessions to determine winner         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Advantages**:
- Removes position bias (both models appear at all positions)
- Higher sensitivity than traditional A/B
- Faster convergence

---

## 7. Common Pitfalls and How to Avoid Them

### Pitfall 1: Sample Ratio Mismatch (SRM)

**Problem**: Expected 50/50 split, but observed 48/52.

**Why It Happens**:
- Buggy assignment code
- Bot traffic filtered inconsistently
- Some users not tracked properly

**Detection**:

```python
from scipy.stats import chisquare

def srm_check(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> dict:
    """
    Check for Sample Ratio Mismatch using chi-square test.
    """
    total = n_control + n_treatment
    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)
    
    chi2, p_value = chisquare(
        [n_control, n_treatment],
        [expected_control, expected_treatment]
    )
    
    return {
        'observed_ratio': n_control / total,
        'expected_ratio': expected_ratio,
        'chi2': chi2,
        'p_value': p_value,
        'srm_detected': p_value < 0.001  # Very strict threshold
    }

# Example
result = srm_check(48000, 52000)
if result['srm_detected']:
    print("⚠️ SRM DETECTED! Do not trust results.")
```

### Pitfall 2: Novelty Effects

**Problem**: New features get more engagement simply because they're new.

**Solution**:
- Run experiments for at least 2 weeks
- Analyze week 1 vs week 2 separately
- If week 1 >> week 2, novelty effect is present

```python
def detect_novelty_effect(df, date_col, metric_col, treatment_col):
    """
    Compare treatment effect in week 1 vs week 2.
    """
    df['week'] = (df[date_col] - df[date_col].min()).dt.days // 7 + 1
    
    results = {}
    for week in [1, 2]:
        week_df = df[df['week'] == week]
        control = week_df[week_df[treatment_col] == 0][metric_col]
        treatment = week_df[week_df[treatment_col] == 1][metric_col]
        results[f'week_{week}_lift'] = (treatment.mean() - control.mean()) / control.mean()
    
    novelty_detected = results['week_1_lift'] > 2 * results['week_2_lift']
    
    return results, novelty_detected
```

### Pitfall 3: Peeking

**Problem**: Looking at results daily and stopping when significant.

**Why It's Bad**: Inflates false positive rate. If you check every day, you'll eventually see p < 0.05 by chance.

**Solution**:
- Pre-register experiment duration
- Use sequential testing methods (group sequential, always-valid p-values)

### Pitfall 4: Metric Definition Drift

**Problem**: Someone changes how "conversion" is calculated mid-experiment.

**Solution**: Lock metric definitions at experiment start.

```python
@dataclass
class ExperimentConfig:
    experiment_id: str
    start_date: datetime
    end_date: datetime
    metric_definitions: dict  # Frozen at creation
    
    def __post_init__(self):
        # Hash definitions to detect changes
        self.definition_hash = hashlib.md5(
            json.dumps(self.metric_definitions, sort_keys=True).encode()
        ).hexdigest()
```

### Pitfall 5: Simpson's Paradox

**Problem**: Treatment wins overall, but loses in every subgroup.

**Example**:
```
Desktop users: Control 10% CTR, Treatment 9% CTR  → Control wins
Mobile users:  Control 5% CTR, Treatment 4% CTR   → Control wins
Overall:       Control 7% CTR, Treatment 8% CTR   → Treatment wins

How? Treatment had more mobile traffic, which has lower CTR but higher volume.
```

**Solution**: Always segment by key dimensions (device, country, user tenure).

---

## 8. Your Interview Story: Fixing the A/B Pipeline

### The Problem

At Schwarz Group, the A/B testing pipeline was producing **unreliable results**. Stakeholders couldn't trust the data, leading to either:
- Shipping features without proper validation
- Paralysis — not shipping anything due to uncertainty

### Your Investigation

You analyzed historical experiments and found:

**Issue 1: Sample Ratio Mismatch**
- Expected 50/50, observed 47/53 in multiple experiments
- Root cause: Assignment hash was using a field that could be null
- Fix: Added validation and fallback logic

**Issue 2: No Novelty Effect Correction**
- Experiments showed strong week 1 effects that vanished by week 2
- Stakeholders were making decisions on week 1 data
- Fix: Implemented mandatory burn-in period (first 7 days excluded from analysis)

**Issue 3: Metric Definition Changes**
- "Conversion rate" was redefined mid-experiment
- Numerator changed, denominator stayed the same
- Fix: Locked metric definitions at experiment start, stored hash

### Your Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IMPROVED A/B PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. EXPERIMENT SETUP                                                │
│     ├── Define metrics (locked)                                     │
│     ├── Calculate required sample size                              │
│     ├── Set expected duration                                       │
│     └── Pre-register hypothesis                                     │
│                                                                     │
│  2. AUTOMATED CHECKS (Daily)                                        │
│     ├── SRM check (χ² test)                                         │
│     ├── Assignment consistency                                      │
│     └── Data quality validation                                     │
│                                                                     │
│  3. ANALYSIS                                                        │
│     ├── Burn-in period exclusion (7 days)                           │
│     ├── Week-over-week comparison                                   │
│     ├── Segment analysis                                            │
│     └── CUPED adjustment                                            │
│                                                                     │
│  4. REPORTING                                                       │
│     ├── Effect size with confidence intervals                       │
│     ├── Statistical significance                                    │
│     └── Business impact projection                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The Result

- Stakeholder trust restored
- Experiment turnaround time reduced (proper power analysis)
- False positive rate reduced from ~15% to expected 5%

### How to Tell This Story in the Interview

> "At Schwarz, I inherited an A/B testing pipeline that was producing inconsistent results. I did a deep dive and found three issues: SRM due to hash function edge cases, novelty effects being ignored, and metric definitions changing mid-experiment.
>
> For SRM, I added automated chi-square checks that would alert if the ratio deviated. For novelty effects, I implemented a mandatory 7-day burn-in period and separate week-over-week analysis. For metric drift, I locked definitions at experiment creation and stored a hash for validation.
>
> The result: stakeholders could trust the data again, and we reduced our false positive rate from about 15% to the expected 5%."

---

## Summary: Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTATION CHECKLIST                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BEFORE EXPERIMENT:                                                 │
│  ☐ Calculate required sample size                                   │
│  ☐ Define metrics (and lock them)                                   │
│  ☐ Set experiment duration                                          │
│  ☐ Pre-register hypothesis                                          │
│                                                                     │
│  DURING EXPERIMENT:                                                 │
│  ☐ Daily SRM checks                                                 │
│  ☐ Monitor data quality                                             │
│  ☐ Don't peek (or use sequential testing)                           │
│                                                                     │
│  AFTER EXPERIMENT:                                                  │
│  ☐ Exclude burn-in period                                           │
│  ☐ Check for novelty effects                                        │
│  ☐ Segment analysis                                                 │
│  ☐ Apply variance reduction (CUPED)                                 │
│  ☐ Multiple testing correction if needed                            │
│  ☐ Report effect size AND significance                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Next: [03_RAG_EVALUATION.md](03_RAG_EVALUATION.md)*
