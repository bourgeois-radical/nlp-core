# Interview Questions & Model Answers

> **Purpose**: Prepare for likely questions with structured, impactful answers  
> **Strategy**: STAR format (Situation, Task, Action, Result) for behavioral; structured frameworks for technical

## Table of Contents
1. [The 2-Minute Pitch](#1-the-2-minute-pitch)
2. [LLM Evaluation Questions](#2-llm-evaluation-questions)
3. [A/B Testing & Experimentation](#3-ab-testing--experimentation)
4. [Technical Deep Dives](#4-technical-deep-dives)
5. [Behavioral Questions](#5-behavioral-questions)
6. [Questions to Ask Them](#6-questions-to-ask-them)
7. [Quick Reference Cheat Sheet](#7-quick-reference-cheat-sheet)

---

## 1. The 2-Minute Pitch

### "Tell me about yourself"

**Structure**: Present → Past → Future (why idealo)

```
"I'm an ML Engineer at Schwarz Group, working on production credit scoring 
and customer review summarization. My credit scoring system serves 250,000+ 
monthly users.

What's most relevant for this role: I built an LLM-as-Judge system for 
evaluating review summaries. The challenge was reliability — single LLM 
calls have variance. My solution used majority voting across multiple 
runs, with statistical validation to flag uncertain cases for human review. 
This gave us reliable pseudo-labels at scale.

I also fixed our A/B testing pipeline that was producing inconsistent 
results — issues with sample ratio mismatch, novelty effects, and metric 
drift.

I'm drawn to idealo's AI Booster Team because it's exactly the intersection 
I want: evaluation, experimentation, and turning GenAI pilots into production. 
I want to be the person who proves whether AI features actually work."
```

**Key Points**:
- Lead with current role and scale
- Highlight **evaluation experience** (core to this role)
- Mention **experimentation** (second core skill)
- End with genuine interest in the specific role

---

## 2. LLM Evaluation Questions

### Q: "Tell me about your experience with LLM evaluation"

**Framework**: Problem → Approach → Innovation → Result

```
"At Schwarz, I built an LLM-as-Judge system for evaluating customer review 
summaries. 

The Problem: We needed to evaluate thousands of summaries but couldn't 
afford human annotation at scale. And single LLM evaluations were 
unreliable — run the same evaluation twice, get different scores.

My Approach: 
1. Multiple LLM calls per evaluation — 5 to 11 runs
2. Majority voting for the final score
3. Statistical validation: analyze the score distribution

The Innovation: Beyond simple majority, I checked the entropy of the 
score distribution. If scores were bimodal (like 2, 2, 5, 5, 5), that 
indicates uncertainty even with a majority. These cases get flagged 
for human review.

Result: We got reliable pseudo-labels that matched human judgment 
about 90% of the time, with uncertain cases properly escalated."
```

### Q: "How would you evaluate a GenAI feature for idealo?"

**Framework**: Offline → Online → Decision

```
"Let me walk through evaluating an AI-generated product summary feature.

OFFLINE EVALUATION:
First, I'd create a golden test set — 500+ products with expert-written 
summaries or expert relevance judgments.

Metrics I'd measure:
- BERTScore for semantic similarity to reference summaries
- Factuality: extract claims, verify against product specs
- LLM-as-Judge for subjective quality (helpfulness, clarity)
- Cost analysis: tokens per summary, latency

I'd run this across product categories — electronics, fashion, home goods — 
because performance varies by domain.

ONLINE EVALUATION:
Once offline metrics look good, I'd design an A/B test:
- 50% users see AI summaries
- 50% see the current experience

Primary metric: Click-through to merchant
Secondary: Time on page, return visits, conversion rate

I'd segment by category and user type to catch heterogeneous effects.

DECISION FRAMEWORK:
Ship if:
- CTR increases by 2%+ (statistically significant)
- Hallucination rate below 1%
- Cost per summary under €0.01
- Latency under 500ms for summary generation

I'd also set up monitoring for post-launch — daily sampling with automated 
evaluation to catch degradation."
```

### Q: "How do you compare different LLMs for a task?"

**Framework**: Criteria → Benchmark → Analysis → Recommendation

```
"I follow a systematic process:

1. DEFINE CRITERIA
   - Quality (accuracy, relevance, safety)
   - Cost ($ per 1K tokens)
   - Latency (p50, p99)
   - Context window needs

2. BUILD EVALUATION SET
   Representative queries from production logs — stratified by:
   - Query type (simple, complex, edge cases)
   - Product category
   - Language (German, English)

3. RUN BENCHMARK
   Same prompts across all models
   Multiple temperatures (0, 0.3, 0.7)
   Measure all criteria

4. PARETO ANALYSIS
   Plot quality vs. cost
   Identify efficient frontier
   
   Example findings:
   - GPT-4: Quality 4.5/5, $0.06/query, 900ms
   - Claude 3.5: Quality 4.4/5, $0.02/query, 600ms
   - GPT-3.5: Quality 3.5/5, $0.003/query, 300ms

5. RECOMMENDATION
   For idealo's scale, Claude 3.5 might win on cost-quality tradeoff.
   But if the task is safety-critical, GPT-4's 0.1 quality edge might 
   justify 3x the cost.

This is how GitHub evaluates models for Copilot — they found that 
different models excel at different tasks."
```

### Q: "How would you measure hallucination rate?"

```
"Three approaches depending on context:

1. CLAIM EXTRACTION + VERIFICATION (most rigorous)
   - Use an LLM to extract factual claims from the output
   - Check each claim against the source documents or knowledge base
   - Hallucination = claims not supported by source
   
   For idealo product summaries, this works well because product 
   specs are structured data — easy to verify.

2. NLI-BASED (scalable)
   - Use an NLI model (Natural Language Inference)
   - Check: does source ENTAIL the output?
   - Score: percentage of output sentences entailed
   
   Faster but less precise.

3. LLM-AS-JUDGE (flexible)
   - Prompt GPT-4: 'Does this output contain facts not supported 
     by the source document?'
   - Calibrate against human labels
   
   Most flexible, but requires careful prompt engineering.

For idealo, I'd start with claim extraction because product specs 
are well-structured. Target hallucination rate: under 1% for 
customer-facing features."
```

---

## 3. A/B Testing & Experimentation

### Q: "Describe an A/B test you ran or fixed"

**STAR Format**

```
SITUATION:
At Schwarz, our A/B testing pipeline was producing inconsistent results. 
Stakeholders couldn't trust the data, which caused either:
- Shipping features without proper validation, or
- Analysis paralysis — not shipping anything

TASK:
I was asked to investigate and fix the pipeline reliability issues.

ACTION:
I analyzed historical experiments and found three problems:

1. Sample Ratio Mismatch (SRM)
   - Expected 50/50 splits, observed 47/53
   - Root cause: Hash function using a nullable field
   - Fix: Added validation and fallback logic, plus automated 
     chi-square SRM checks

2. Novelty Effects
   - Week 1 effects were inflated, vanished by week 2
   - Stakeholders were deciding based on week 1 data
   - Fix: Mandatory 7-day burn-in period, week-over-week analysis

3. Metric Definition Drift
   - "Conversion rate" definition changed mid-experiment
   - Fix: Locked metric definitions at experiment start, stored hash

I also implemented CUPED for variance reduction, using pre-experiment 
user behavior to reduce noise.

RESULT:
- Stakeholder trust restored
- False positive rate dropped from ~15% to expected 5%
- Experiment turnaround time improved (proper power analysis meant 
  we weren't running experiments longer than needed)
```

### Q: "How do you handle multiple testing correction?"

```
"When testing multiple metrics simultaneously — say CTR, conversion rate, 
and time on page — you risk inflating the false positive rate.

Two approaches:

1. BONFERRONI (conservative)
   Divide alpha by number of tests.
   If testing 5 metrics at 5% significance: use 1% per test.
   
   Pro: Simple, guarantees family-wise error rate.
   Con: Very conservative, may miss real effects.

2. BENJAMINI-HOCHBERG (FDR control)
   Controls false discovery rate, not family-wise error.
   Sort p-values, compare to adjusted thresholds.
   
   Pro: More power than Bonferroni.
   Con: Some false positives expected (controlled rate).

For idealo, I'd recommend:
- Primary metric (e.g., CTR): No correction, this is the decision driver
- Secondary metrics: FDR control with BH procedure
- Document the decision upfront in the experiment plan

The key is pre-registration — decide correction approach BEFORE 
seeing results, not after."
```

### Q: "What is CUPED and when would you use it?"

```
"CUPED — Controlled-experiment Using Pre-Experiment Data — is a 
variance reduction technique.

THE IDEA:
Use pre-experiment behavior to predict what the metric WOULD have been 
without treatment. Analyze residuals instead of raw values.

MATH:
Y_adjusted = Y - θ(X - X̄)

Where:
- Y = post-experiment metric
- X = pre-experiment covariate (e.g., last month's CTR)
- θ = regression coefficient (Cov(X,Y) / Var(X))

IMPACT:
Variance is reduced by (1 - ρ²), where ρ is the correlation between 
pre and post metrics.

If pre-experiment CTR correlates 0.7 with post-experiment CTR:
Variance reduction = 1 - 0.49 = 51%
This effectively DOUBLES your sample size.

WHEN TO USE:
- Long-running experiments where you can't just add more users
- Metrics with high user-level variance
- When you have reliable pre-experiment data

At idealo, for a search quality experiment, I'd use:
- Pre-experiment covariate: User's CTR over the past 30 days
- This captures user-level baseline engagement

Microsoft, Netflix, and Booking.com all use CUPED extensively."
```

---

## 4. Technical Deep Dives

### Q: "Explain how you'd build an LLM-as-Judge pipeline"

```
"I'd build it in layers:

LAYER 1: PROMPT ENGINEERING
- Clear criteria definition
- Discrete scale (1-5 or binary)
- Chain-of-thought reasoning
- Few-shot examples of each score level

Example prompt structure:
```
# Task
Evaluate this product summary for accuracy.

# Scale
1 = Major factual errors
2 = Minor factual errors  
3 = Accurate but incomplete
4 = Accurate and mostly complete
5 = Accurate and comprehensive

# Context
Product specs: {specs}
Generated summary: {summary}

# Instructions
1. List each claim in the summary
2. Check each against the specs
3. Note any discrepancies
4. Assign a score

# Output
Score: 
Reasoning:
```

LAYER 2: VARIANCE REDUCTION
- Multiple runs (5-11) per evaluation
- Temperature = 0 for reproducibility
- Majority voting for final score

LAYER 3: CONFIDENCE ESTIMATION
- Analyze score distribution entropy
- Low entropy = high confidence
- High entropy = flag for human review

LAYER 4: HUMAN CALIBRATION
- Golden dataset with expert labels
- Regular auditing (sample 5% of automated judgments)
- Recalibrate prompts when alignment drops

This is essentially what Instacart built with LACE, and what 
Segment built for their CustomerAI evaluation."
```

### Q: "What metrics would you track for a RAG system?"

```
"I'd track metrics at each pipeline stage:

RETRIEVAL METRICS:
- Recall@K: Did we find the relevant documents?
- MRR: How high is the first relevant doc?
- NDCG: Quality of the full ranking
- Empty result rate: How often do we find nothing?
- Latency: Retrieval time (target: <100ms)

GENERATION METRICS:
- Faithfulness: Is the answer grounded in retrieved docs?
- Hallucination rate: Claims not in source
- Answer relevance: Does it address the query?
- Completeness: Are key points covered?
- Latency: Generation time (target: <1s)

END-TO-END METRICS:
- Task success rate: Did user accomplish their goal?
- User satisfaction: Thumbs up/down
- Reformulation rate: Did user have to re-query?

COST METRICS:
- Cost per query (embedding + LLM)
- Token usage

For idealo, I'd add domain-specific metrics:
- Price accuracy: Is the 'best price' actually best?
- Merchant attribution: Did we correctly identify the source?

DoorDash's WPR metric is relevant here too — for idealo's product 
grids, I'd weight results by visual prominence."
```

### Q: "Walk me through your SQL skills"

```python
# Example: Experiment analysis query

"""
-- Calculate conversion rate by variant with confidence intervals
WITH daily_metrics AS (
    SELECT 
        variant,
        DATE(timestamp) as date,
        COUNT(DISTINCT user_id) as users,
        SUM(converted) as conversions
    FROM experiment_events
    WHERE experiment_id = 'genai_search_v1'
      AND timestamp BETWEEN '2025-01-15' AND '2025-01-29'
      AND timestamp >= '2025-01-22'  -- Burn-in period
    GROUP BY variant, DATE(timestamp)
),

variant_stats AS (
    SELECT 
        variant,
        SUM(users) as total_users,
        SUM(conversions) as total_conversions,
        SUM(conversions) * 1.0 / SUM(users) as conversion_rate,
        -- Standard error for proportion
        SQRT(
            (SUM(conversions) * 1.0 / SUM(users)) * 
            (1 - SUM(conversions) * 1.0 / SUM(users)) / 
            SUM(users)
        ) as se
    FROM daily_metrics
    GROUP BY variant
)

SELECT 
    variant,
    total_users,
    total_conversions,
    ROUND(conversion_rate * 100, 2) as conversion_pct,
    -- 95% confidence interval
    ROUND((conversion_rate - 1.96 * se) * 100, 2) as ci_lower,
    ROUND((conversion_rate + 1.96 * se) * 100, 2) as ci_upper
FROM variant_stats
ORDER BY variant;
"""
```

---

## 5. Behavioral Questions

### Q: "Tell me about a time you had to influence stakeholders"

```
SITUATION:
At Schwarz, after fixing the A/B pipeline, I discovered that a 
feature the team was about to ship showed a novelty effect — 
strong week 1 results that vanished by week 2.

TASK:
I needed to convince the product manager and engineering lead 
to delay launch and extend the experiment.

ACTION:
- Prepared a clear visualization: week-over-week trend showing 
  the declining effect
- Calculated what the true effect was (removing novelty bias)
- Showed examples from industry (LinkedIn, Netflix) where novelty 
  effects led to bad launches
- Proposed a solution: extend experiment 2 more weeks, implement 
  novelty monitoring

RESULT:
They agreed to extend. Final analysis showed the true lift was 
0.5%, not the 3% we saw in week 1. We still shipped, but with 
realistic expectations and proper monitoring in place.

LEARNING:
Data visualization was key. Showing the trend over time was more 
convincing than just telling them the number was wrong.
```

### Q: "How do you handle disagreement with colleagues?"

```
"My approach is to separate the person from the problem.

Recent example: A colleague insisted we use a 1-100 scale for our 
LLM-as-Judge evaluation. I'd read research showing LLMs struggle 
with continuous scales — they cluster around certain values.

Instead of just disagreeing, I:
1. Acknowledged their reasoning (more granularity seems better)
2. Proposed an experiment: run both approaches on the same data
3. Compared alignment with human judgments

Results showed the 1-5 scale had higher agreement with humans. 
The evidence resolved the disagreement, and we implemented 
discrete scoring.

The key is making disagreements about evidence, not preferences."
```

### Q: "What's your biggest weakness?"

```
"I sometimes over-engineer solutions. My first instinct with the 
LLM-as-Judge system was to build a complex multi-agent architecture 
with debate and reflection.

I've learned to start simpler. Now I ask: 'What's the simplest 
thing that could possibly work?' Build that first, measure it, 
then add complexity only where needed.

For the evaluation system, I started with simple majority voting. 
When that worked 85% of the time, I added entropy-based confidence 
only for the uncertain 15%. That was enough — I didn't need the 
full debate architecture for our use case."
```

---

## 6. Questions to Ask Them

### About the Role

1. **"What's a recent GenAI feature the AI Booster Team helped ship? How did you measure success?"**
   - Shows you understand the role's purpose

2. **"What's the biggest challenge in evaluating LLMs at idealo's scale?"**
   - Probes for real problems you'll face

3. **"How does the team balance speed (shipping fast) vs rigor (thorough evaluation)?"**
   - Shows you understand the tension

### About the Technical Stack

4. **"What does the tech stack look like for evaluation pipelines? Bedrock? Open source?"**
   - Practical information you need

5. **"How do product teams currently request AI Booster support?"**
   - Understand the operating model

### About Growth

6. **"What would success look like for this role in the first 6 months?"**
   - Sets clear expectations

7. **"How does the AI Booster Team's roadmap connect to idealo's broader strategy?"**
   - Shows strategic thinking

---

## 7. Quick Reference Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTERVIEW CHEAT SHEET                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  YOUR KILLER POINTS:                                                │
│  ✓ LLM-as-Judge with majority voting + statistical validation       │
│  ✓ Fixed broken A/B pipeline (SRM, novelty, metric drift)           │
│  ✓ Production ML at scale (250K users/month)                        │
│  ✓ Statistical foundation (MLE, distributions, inference)           │
│                                                                     │
│  LLM EVAL METRICS:                                                  │
│  • BLEU/ROUGE — n-gram overlap                                      │
│  • BERTScore — semantic similarity                                  │
│  • LLM-as-Judge — GPT-4 rates 1-5                                   │
│  • Hallucination rate — claim verification                          │
│                                                                     │
│  A/B TESTING CHECKS:                                                │
│  • SRM (sample ratio mismatch) — chi-square test                    │
│  • Novelty effects — week-over-week comparison                      │
│  • Multiple testing — Bonferroni or BH correction                   │
│  • CUPED — variance reduction using pre-experiment data             │
│                                                                     │
│  RAG METRICS:                                                       │
│  • Retrieval: Recall@K, MRR, NDCG                                   │
│  • Generation: Faithfulness, Relevance, Hallucination               │
│  • E2E: Task success, User satisfaction                             │
│                                                                     │
│  COST AWARENESS:                                                    │
│  • GPT-4: ~$0.03/1K input, $0.06/1K output                          │
│  • Claude 3.5: ~$0.003/1K input, $0.015/1K output                   │
│  • Fine-tuned small model can be 10-100x cheaper                    │
│                                                                     │
│  CASE STUDY REFERENCES:                                             │
│  • GitHub: 4,000+ offline tests, model proxy for A/B                │
│  • DoorDash: WPR metric, fine-tuned judge beats crowd               │
│  • Segment: 90% human alignment, discrete scales                    │
│  • Instacart: Binary scoring, debate-style evaluation               │
│  • trivago: 90% adoption, 16 days saved/person/year                 │
│                                                                     │
│  WHY IDEALO:                                                        │
│  "I want to be the person who proves GenAI works — not just builds  │
│  it. The AI Booster Team's mission of turning pilots into           │
│  production through rigorous evaluation is exactly what I want."    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Next: [06_IDEALO_SPECIFIC.md](06_IDEALO_SPECIFIC.md)*
