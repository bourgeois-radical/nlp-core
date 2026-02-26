# idealo Business Context & GenAI Use Cases

> **Purpose**: Understand idealo's business to propose relevant GenAI applications  
> **Key Insight**: idealo is a **price comparison platform** — different challenges than e-commerce marketplaces

## Table of Contents
1. [Understanding idealo](#1-understanding-idealo)
2. [The AI Booster Team's Mission](#2-the-ai-booster-teams-mission)
3. [GenAI Opportunities at idealo](#3-genai-opportunities-at-idealo)
4. [Evaluation Framework for idealo](#4-evaluation-framework-for-idealo)
5. [Relevant Industry Parallels](#5-relevant-industry-parallels)
6. [Technical Stack Expectations](#6-technical-stack-expectations)
7. [Interview-Ready Examples](#7-interview-ready-examples)

---

## 1. Understanding idealo

### The Business Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IDEALO BUSINESS MODEL                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WHAT IDEALO IS:                                                    │
│  ├── Price comparison platform (like Google Shopping)              │
│  ├── 600+ million product offers                                   │
│  ├── 50,000+ merchants                                             │
│  ├── 2.5 million page views/day                                    │
│  └── Owned by Axel Springer SE                                     │
│                                                                     │
│  HOW IT MAKES MONEY:                                                │
│  ├── CPC (Cost Per Click): Merchants pay when users click through  │
│  ├── CPA (Cost Per Acquisition): Pay for conversions               │
│  └── Premium listings and merchant services                        │
│                                                                     │
│  KEY DIFFERENCE FROM MARKETPLACES:                                  │
│  ├── idealo doesn't sell products                                  │
│  ├── idealo sends users TO merchants                               │
│  └── Success = clicks to merchants + user satisfaction             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The User Journey

```
User searches                    idealo shows                  User clicks
"iPhone 15 128GB"    →    price comparison grid    →    to cheapest/best merchant
                          with multiple offers
                          
┌──────────────────────────────────────────────────────────────────────┐
│                         idealo Product Page                          │
├──────────────────────────────────────────────────────────────────────┤
│  iPhone 15 128GB                                                     │
│  ★★★★☆ (4.5) · 1,234 reviews                                        │
│                                                                      │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐           │
│  │ Amazon         │ │ MediaMarkt     │ │ Saturn         │           │
│  │ €799.00       │ │ €819.00       │ │ €829.00       │           │
│  │ [Go to shop →]│ │ [Go to shop →]│ │ [Go to shop →]│           │
│  └────────────────┘ └────────────────┘ └────────────────┘           │
│                                                                      │
│  Price history graph | Specifications | Reviews                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Metrics for idealo

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **CTR to Merchant** | % of users who click to a shop | Direct revenue driver |
| **Search Success Rate** | % of searches resulting in click | User satisfaction |
| **Price Accuracy** | Is displayed price correct? | Trust and compliance |
| **Pages per Session** | Engagement depth | User experience |
| **Return Visit Rate** | Users coming back | Long-term value |
| **Time to Click** | Speed of user decision | UX quality |

### Competitive Landscape

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRICE COMPARISON LANDSCAPE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DIRECT COMPETITORS:                                                │
│  ├── Google Shopping (dominant, integrated in search)              │
│  ├── Geizhals (strong in DACH region)                               │
│  ├── PriceRunner (Nordic focus)                                     │
│  └── Kelkoo (European)                                              │
│                                                                     │
│  INDIRECT COMPETITORS:                                              │
│  ├── Amazon (users start searches there)                            │
│  ├── Direct-to-merchant searches                                    │
│  └── Social commerce (TikTok Shop, Instagram)                       │
│                                                                     │
│  IDEALO'S DIFFERENTIATION:                                          │
│  ├── Deep product data and specifications                           │
│  ├── Price history and alerts                                       │
│  ├── Independent reviews                                            │
│  └── European focus, GDPR-compliant                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. The AI Booster Team's Mission

### From the Job Description

> "At idealo, Generative AI (GenAI) is becoming a multiplier across every team. The AI Booster Team is our internal technical competence center: we pair with product teams, build reusable GenAI building blocks and share best practices company-wide."

### The Operating Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI BOOSTER TEAM STRUCTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ROLE 1: COMPETENCE CENTER                                          │
│  ├── Share GenAI best practices                                     │
│  ├── Build reusable components                                      │
│  └── Coach teams on evaluation                                      │
│                                                                     │
│  ROLE 2: VALIDATION PARTNER                                         │
│  ├── "Validate AI business cases through data"                      │
│  ├── Run experiments                                                │
│  └── Ship evaluation frameworks                                     │
│                                                                     │
│  ROLE 3: PILOT-TO-PRODUCTION                                        │
│  ├── "Turn pilots into production"                                  │
│  ├── Define quality thresholds                                      │
│  └── Monitor deployed GenAI features                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Similar to trivago's AI Ambassadors

From trivago's case study:
> "The AI Ambassadors are at the heart of trivago's AI evolution. A cross-functional group that meets bi-weekly to share knowledge, benchmark solutions, and ensure that AI adoption is both strategic and inclusive."

**Parallel to idealo AI Booster Team:**
- Cross-functional (work with product teams)
- Knowledge sharing (best practices company-wide)
- Benchmarking solutions (model selection)
- Strategic adoption (validate business cases)

---

## 3. GenAI Opportunities at idealo

### Opportunity Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GENAI OPPORTUNITIES AT IDEALO                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. SEARCH & DISCOVERY                                              │
│     ├── Natural language product search                             │
│     │   "Show me budget laptops good for video editing"             │
│     ├── Query understanding & expansion                             │
│     │   "iPhone" → "iPhone 15, 14, 13, SE, Pro Max..."              │
│     ├── Conversational product finding                              │
│     │   Chatbot guides user to right product                        │
│     └── Related product recommendations                             │
│                                                                     │
│  2. PRODUCT CONTENT                                                 │
│     ├── Auto-generate product descriptions                          │
│     │   From specs → readable summary                               │
│     ├── Summarize product reviews                                   │
│     │   "Users love battery life, complain about camera"            │
│     ├── Attribute extraction from merchant data                     │
│     │   Unstructured text → structured specs                        │
│     └── Product categorization                                      │
│         Classify products into idealo taxonomy                      │
│                                                                     │
│  3. USER EXPERIENCE                                                 │
│     ├── Personalized deal alerts                                    │
│     │   "Price dropped on item you viewed"                          │
│     ├── Price trend explanations                                    │
│     │   "Why is this product expensive right now?"                  │
│     ├── Comparison summaries                                        │
│     │   "Product A vs B: key differences"                           │
│     └── Purchase decision assistance                                │
│                                                                     │
│  4. INTERNAL OPERATIONS                                             │
│     ├── Merchant data quality checks                                │
│     │   Detect incorrect prices, descriptions                       │
│     ├── Fraud/scam detection                                        │
│     │   Suspicious merchant listings                                │
│     ├── Customer support automation                                 │
│     │   Answer user questions at scale                              │
│     └── Content moderation                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Prioritization Matrix

| Use Case | Business Impact | Technical Complexity | Data Available? |
|----------|-----------------|---------------------|-----------------|
| Review summarization | High (UX) | Medium | ✓ Reviews exist |
| Natural language search | High (differentiation) | High | ✓ Queries logged |
| Product descriptions | Medium (SEO) | Low | ✓ Specs available |
| Attribute extraction | High (data quality) | Medium | ✓ Merchant data |
| Price explanations | Medium (UX) | Medium | ✓ Price history |
| Query expansion | High (coverage) | Medium | ✓ Search logs |
| Chatbot | High (engagement) | High | Partial |

### Deep Dive: Review Summarization

**Why it's high value:**
- idealo has millions of product reviews
- Users don't read all reviews
- Summary increases decision confidence → higher CTR

**Evaluation approach (your expertise!):**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REVIEW SUMMARY EVALUATION                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  OFFLINE METRICS:                                                   │
│  ├── Coverage: Are key themes from reviews captured?                │
│  ├── Accuracy: Does summary reflect actual sentiment?               │
│  ├── Factuality: No hallucinated features or issues                 │
│  └── Conciseness: Summary length vs info density                    │
│                                                                     │
│  LLM-AS-JUDGE CRITERIA:                                             │
│  ├── "Does the summary accurately represent the reviews?"           │
│  ├── "Are there any claims not supported by reviews?"               │
│  ├── "Would this summary help a user make a purchase decision?"     │
│  └── Binary scoring (Yes/No) per criterion                          │
│                                                                     │
│  ONLINE A/B TEST:                                                   │
│  ├── Control: Show reviews only                                     │
│  ├── Treatment: Show reviews + AI summary                           │
│  ├── Primary metric: CTR to merchant                                │
│  └── Secondary: Time to click, page engagement                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Deep Dive: Natural Language Search

**The opportunity:**
- Current: keyword search ("iphone 15 128gb black")
- Future: conversational ("best phone for photography under €800")

**DoorDash parallel:**
From their case study:
> "WPR measures multiple content blocks arranged spatially on the screen, including stores, dishes, and items."

**idealo equivalent: Whole-Page Relevance for product grids**
- Weight products by visual position
- Measure grid-level relevance, not just top result
- Account for price sorting, filters

**Evaluation approach:**

```python
def idealo_page_relevance(query, results, llm_judge):
    """
    Whole-Page Relevance for idealo search results.
    Inspired by DoorDash's WPR metric.
    """
    # Position weights (grid layout)
    position_weights = {
        0: 1.0,   # Top-left (most valuable)
        1: 0.9,   # Top-center
        2: 0.8,   # Top-right
        3: 0.5,   # Second row
        4: 0.5,
        5: 0.5,
        # ... diminishing weights
    }
    
    weighted_relevance = 0.0
    
    for position, product in enumerate(results):
        # LLM judges relevance of this product to query
        relevance_score = llm_judge.evaluate(
            query=query,
            product=product,
            criteria="Does this product satisfy the user's search intent?"
        )
        
        weight = position_weights.get(position, 0.1)
        weighted_relevance += relevance_score * weight
    
    # Normalize by sum of weights used
    total_weight = sum(position_weights.get(i, 0.1) for i in range(len(results)))
    
    return weighted_relevance / total_weight
```

---

## 4. Evaluation Framework for idealo

### Adapting Industry Best Practices

**From GitHub:**
> "We can test a new model without changing the product code. We have a proxy server built into our infrastructure."

**idealo equivalent:**
- Model abstraction layer for evaluation
- A/B test models without code changes
- Rapid iteration on prompts

**From DoorDash:**
> "AutoEval has reduced relevance judgment turnaround time by 98%."

**idealo equivalent:**
- Automated evaluation pipeline
- Near-real-time quality monitoring
- Free up experts for edge cases

**From Segment:**
> "LLMs struggle with continuous scores... we used a discrete 1-5 scale."

**idealo equivalent:**
- Binary or 1-5 scoring for quality metrics
- Chain-of-thought for explainability
- 90%+ alignment with human judgment target

### Proposed Evaluation Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IDEALO EVALUATION STACK                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DATA LAYER                                                         │
│  ├── Query logs (search behavior)                                   │
│  ├── Product catalog (specs, prices)                                │
│  ├── Review corpus (user opinions)                                  │
│  └── Click streams (user journeys)                                  │
│                                                                     │
│  EVALUATION LAYER                                                   │
│  ├── Offline benchmarks                                             │
│  │   ├── Golden test sets (expert-labeled)                          │
│  │   ├── LLM-as-Judge (automated scoring)                           │
│  │   └── Reference metrics (ROUGE, BERTScore)                       │
│  │                                                                   │
│  └── Online experiments                                             │
│      ├── A/B test framework                                         │
│      ├── User engagement metrics                                    │
│      └── Business metrics (CTR, revenue)                            │
│                                                                     │
│  TOOLING (AWS)                                                      │
│  ├── Bedrock: Foundation models                                     │
│  ├── SageMaker: Experiment tracking                                 │
│  ├── Athena: Query logs analysis                                    │
│  └── Lambda/Step Functions: Pipeline orchestration                  │
│                                                                     │
│  MONITORING                                                         │
│  ├── Quality dashboards                                             │
│  ├── Cost tracking                                                  │
│  └── Drift detection                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Quality Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Hallucination rate | < 1% | Trust is critical for price comparison |
| LLM-as-Judge accuracy | > 4.0/5.0 | Must be genuinely helpful |
| Latency | < 500ms | User experience |
| Cost per query | < €0.01 | Scale to millions of queries |
| Human alignment | > 90% | Match expert judgment |

---

## 5. Relevant Industry Parallels

### trivago (Closest Parallel!)

Both are **comparison platforms** (hotels vs products).

**Key insights from trivago:**
> "16 days saved per person per year on average, thanks to AI-driven efficiencies."

| trivago Metric | Value | idealo Opportunity |
|----------------|-------|-------------------|
| AI adoption rate | >90% | Target for internal tools |
| Days saved/person/year | 16 | Efficiency measurement |
| IT Support chatbot resolution | 35% | Customer support target |

**trivago's AI Radar approach:**
> "A live dashboard categorising all tested and approved AI tools by function and adoption status, currently with 42 tools mapped."

**idealo AI Booster could build:**
- Internal AI tool registry
- Approval workflow for new tools
- Usage and ROI tracking

### DoorDash (Search Relevance)

Both need to **rank results** (restaurants vs products).

**Applicable learnings:**
- Whole-Page Relevance metric
- Fine-tuned judge models
- Stratified sampling (head, torso, tail queries)

**From DoorDash:**
> "Fine-tuned LLMs consistently match or outperform external raters in key relevance tasks."

### Segment (Structured Output Evaluation)

Both generate **structured data** (ASTs vs product attributes).

**Applicable learnings:**
- Semantic equivalence checking
- Synthetic test data generation
- LLM-as-Judge for code/schema validation

---

## 6. Technical Stack Expectations

### AWS Focus (from Job Description)

> "Proficiency in AWS analytics & MLOps: SageMaker Experiments / Pipelines, Bedrock, Athena, Lambda, Step Functions"

| Service | Use Case at idealo |
|---------|-------------------|
| **Bedrock** | Foundation models (Claude, Titan) for evaluation |
| **SageMaker Experiments** | Track model comparisons |
| **SageMaker Pipelines** | Orchestrate evaluation workflows |
| **Athena** | Query experiment logs at scale |
| **Lambda** | Trigger evaluations, webhooks |
| **Step Functions** | Multi-step evaluation pipelines |

### Evaluation Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │ Query       │     │ Sample      │     │ Evaluate    │           │
│  │ Sampling    │ ──▶ │ Preparation │ ──▶ │ (Bedrock)   │           │
│  │ (Athena)    │     │ (Lambda)    │     │             │           │
│  └─────────────┘     └─────────────┘     └──────┬──────┘           │
│                                                  │                  │
│                                                  ▼                  │
│                                          ┌─────────────┐           │
│                                          │ Aggregate   │           │
│                                          │ (Lambda)    │           │
│                                          └──────┬──────┘           │
│                                                  │                  │
│              ┌──────────────────┬────────────────┤                  │
│              ▼                  ▼                ▼                  │
│       ┌─────────────┐   ┌─────────────┐  ┌─────────────┐           │
│       │ Dashboard   │   │ Alerts      │  │ Store       │           │
│       │ (QuickSight)│   │ (SNS)       │  │ (S3)        │           │
│       └─────────────┘   └─────────────┘  └─────────────┘           │
│                                                                     │
│  ORCHESTRATION: Step Functions                                      │
│  TRACKING: SageMaker Experiments                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Interview-Ready Examples

### Example 1: Search Quality Evaluation

**Scenario:** idealo wants to add AI-powered query expansion.

**Your approach:**

```
"I'd evaluate this in three stages:

OFFLINE:
1. Sample queries from logs — stratified by frequency (head/torso/tail)
2. For each query, generate expanded queries with the AI
3. Measure coverage increase: how many more relevant products returned?
4. Use LLM-as-Judge to score relevance of expansions
   - 'Does [expanded query] capture the user's intent?'

ONLINE A/B TEST:
- Control: current search
- Treatment: AI-expanded search
- Primary metric: CTR to merchant
- Segment by query type (product, category, brand)

DECISION:
Ship if:
- CTR increases 2%+
- No decrease in search success rate
- Latency impact < 50ms

MONITORING:
- Daily sampling with automated evaluation
- Alert if quality drops below threshold"
```

### Example 2: Review Summarization Quality

**Scenario:** Launching AI-generated review summaries.

**Your approach:**

```
"This is similar to what I built at Schwarz. Here's my evaluation plan:

GOLDEN DATASET:
- 500 products across categories
- Expert-written reference summaries
- Annotated for coverage, accuracy, helpfulness

AUTOMATED METRICS:
- BERTScore vs reference summaries
- Claim extraction: verify each claim against source reviews
- LLM-as-Judge with 5-run majority voting

CRITERIA:
1. Coverage: Does summary mention major themes? (Binary)
2. Accuracy: Does sentiment match reviews? (Binary)
3. Factuality: Any hallucinated features? (Binary)
4. Helpfulness: Would this help purchase decision? (1-5)

A/B TEST:
- 2-week experiment with burn-in
- Primary: CTR to merchant for products with summaries
- Secondary: Time on page, return visits

THRESHOLD:
- Hallucination rate < 0.5% (stricter for customer-facing)
- LLM-Judge average > 4.2/5.0
- CTR lift > 1%"
```

### Example 3: Model Selection

**Scenario:** Choose between GPT-4 and Claude for product descriptions.

**Your approach:**

```
"I'd run a systematic comparison:

EVALUATION SET:
- 1000 products across categories
- Include edge cases (multilingual, technical specs, fashion)

METRICS:
- Quality: LLM-as-Judge (accuracy, style, completeness)
- Cost: $ per description
- Latency: p50, p99 generation time

PROCESS:
1. Same prompts to both models
2. 3 temperatures (0, 0.3, 0.7)
3. Blind human evaluation on subset (100 products)
4. LLM-as-Judge on full set

ANALYSIS:
Plot quality vs cost (Pareto frontier)

Example findings:
- GPT-4: Quality 4.6/5, €0.02/desc, 800ms
- Claude 3.5: Quality 4.5/5, €0.008/desc, 500ms
- Claude wins for idealo's scale (millions of products)

RECOMMENDATION:
Use Claude for bulk generation, GPT-4 for complex categories 
where 0.1 quality difference matters (electronics with specs)"
```

### Why idealo Script

```
"I'm drawn to idealo for three reasons:

1. THE ROLE
   The AI Booster Team is exactly what excites me — being the 
   person who proves GenAI works, not just builds it. Turning 
   pilots into production through rigorous evaluation.

2. THE CHALLENGE
   Price comparison at scale is fascinating. You have structured 
   data (prices, specs) mixed with unstructured (reviews, queries). 
   Evaluating GenAI here needs both automated metrics and business 
   outcome measurement.

3. THE STAGE
   idealo is at the perfect point — established enough to have 
   data and scale, but early enough in GenAI adoption that I can 
   shape how evaluation is done. Building reusable frameworks that 
   become company-wide standards.

My experience with LLM-as-Judge and A/B testing is directly 
applicable. I want to bring the rigor I've built at Schwarz to 
idealo's GenAI initiatives."
```

---

## Summary: Key Talking Points

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IDEALO INTERVIEW TALKING POINTS                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  UNDERSTAND THE BUSINESS:                                           │
│  ├── Price comparison, not marketplace                              │
│  ├── Revenue = clicks to merchants                                  │
│  ├── Primary metric: CTR                                            │
│  └── Trust is critical (price accuracy, no hallucinations)          │
│                                                                     │
│  AI BOOSTER TEAM ROLE:                                              │
│  ├── Internal competence center                                     │
│  ├── Validate business cases through data                           │
│  ├── Turn pilots into production                                    │
│  └── Similar to trivago's AI Ambassadors                            │
│                                                                     │
│  TOP GENAI OPPORTUNITIES:                                           │
│  ├── Review summarization (your exact experience!)                  │
│  ├── Natural language search                                        │
│  ├── Product attribute extraction                                   │
│  └── Query understanding/expansion                                  │
│                                                                     │
│  EVALUATION APPROACH:                                               │
│  ├── Offline: LLM-as-Judge, BERTScore, claim verification           │
│  ├── Online: A/B tests with CTR as primary metric                   │
│  ├── Thresholds: <1% hallucination, >4.0/5.0 quality                │
│  └── Similar to DoorDash's AutoEval, GitHub's 4000+ tests           │
│                                                                     │
│  INDUSTRY PARALLELS:                                                │
│  ├── trivago: Same business model (comparison platform)             │
│  ├── DoorDash: Search relevance, WPR metric                         │
│  ├── Segment: Structured output evaluation                          │
│  └── GitHub: Model selection, offline testing                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Good luck! Remember: They want someone who **measures and validates**, not just builds. Lead with your evaluation experience.*
