# Industry Case Studies — LLM Evaluation in Production

> **Purpose**: Learn from companies that have solved LLM evaluation at scale  
> **Key Insight**: Every successful system combines automated evaluation with human calibration

## Table of Contents
1. [GitHub Copilot — Model Evaluation at Scale](#1-github-copilot--model-evaluation-at-scale)
2. [DoorDash AutoEval — LLM-Powered Search Evaluation](#2-doordash-autoeval--llm-powered-search-evaluation)
3. [Segment — LLM-as-Judge for Code Generation](#3-segment--llm-as-judge-for-code-generation)
4. [Instacart LACE — Customer Support Chatbot Evaluation](#4-instacart-lace--customer-support-chatbot-evaluation)
5. [trivago — AI Adoption at Company Scale](#5-trivago--ai-adoption-at-company-scale)
6. [Synthesis: Common Patterns](#6-synthesis-common-patterns)
7. [Applying These Lessons to idealo](#7-applying-these-lessons-to-idealo)

---

## 1. GitHub Copilot — Model Evaluation at Scale

### Company Context

- **Product**: AI-powered code completion and chat
- **Scale**: Millions of developers using Copilot daily
- **Challenge**: Evaluate multiple models (GPT-4, Claude, Gemini) for quality, performance, and safety

### Key Quote

> "Just because a model is newer doesn't mean it will perform better for your use case."

This is the core insight for idealo: **don't chase the newest model — validate it rigorously for your specific task**.

### Evaluation Infrastructure

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GITHUB'S EVALUATION STACK                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INFRASTRUCTURE:                                                    │
│  ├── Custom platform built with GitHub Actions                      │
│  ├── Apache Kafka for data streaming                                │
│  ├── Microsoft Azure for compute                                    │
│  └── Dashboards for result exploration                              │
│                                                                     │
│  SCALE: > 4,000 offline tests                                       │
│                                                                     │
│  KEY CAPABILITY:                                                    │
│  "We can test a new model without changing the product code.        │
│   We have a proxy server that can switch API endpoints easily."     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Evaluation Methods

**1. Code Completion: Unit Test Pass Rate**

GitHub has ~100 containerized repositories with passing CI tests. They:
1. Deliberately break the code
2. Ask the model to fix it
3. Measure if CI tests pass again

**Why This Matters**: It's a **functional evaluation** — the output either works or it doesn't. No subjective judgment needed.

**2. Chat Quality: LLM-as-Judge**

For more subjective questions (explanations, suggestions), they use another LLM to evaluate:

> "We use a model with known good performance for these purposes to ensure consistent evaluations across our work. We also routinely audit the outputs of this LLM in evaluation scenarios to make sure it's working correctly."

**3. Responsible AI Testing**

> "Regardless of the model used, GitHub Copilot tests both prompts and responses for relevance, such as non-code related questions, and toxic language such as hate speech, sexual content, violence, and evidence of self-harm."

### What They Measure

| Metric | For | Why |
|--------|-----|-----|
| **Unit test pass rate** | Code completions | Functional correctness |
| **Similarity to original** | Code quality | Baseline comparison |
| **Answer accuracy** | Chat | Factual correctness |
| **Token usage** | All | Efficiency / cost |

### Inverse Relationships Insight

> "There are sometimes inverse relationships between metrics. Higher latency might actually lead to a higher acceptance rate because users might see fewer suggestions total."

**Lesson**: Don't optimize a single metric in isolation. A faster model might show more suggestions, but users accept fewer of them.

### Application to idealo

| GitHub Practice | idealo Equivalent |
|-----------------|-------------------|
| Unit test pass rate | Does product recommendation lead to click? |
| Code similarity to original | Answer similarity to ground truth |
| Proxy server for model switching | Model gateway for A/B testing |
| Daily production monitoring | Continuous relevance monitoring |

---

## 2. DoorDash AutoEval — LLM-Powered Search Evaluation

### Company Context

- **Product**: Food delivery search (restaurants, dishes, items)
- **Scale**: Millions of search queries daily across restaurants, retail, grocery, pharmacy
- **Challenge**: Traditional human annotation couldn't scale — too slow, inconsistent, expensive

### The Problem They Solved

> "It isn't feasible to manually assess millions of query-document pairs, especially as search evolves daily. Human annotation cycles can take days or weeks, slowing iteration speed for search improvements."

**Before AutoEval**:
- Human raters evaluated search results
- Days/weeks of turnaround
- Inconsistent ratings
- Limited to high-frequency ("head") queries, missing issues in "tail" queries

**After AutoEval**:
- **98% reduction** in turnaround time
- **9x increase** in evaluation capacity
- Coverage across head, torso, and tail queries

### The WPR Metric (Whole-Page Relevance)

This is DoorDash's most important contribution. Unlike NDCG (which evaluates a vertical list), WPR measures the **entire search page**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WPR CONCEPT                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Traditional NDCG: Linear list                                      │
│  Position 1: Weight = 1.0                                           │
│  Position 2: Weight = 0.63                                          │
│  Position 3: Weight = 0.50                                          │
│  ...                                                                │
│                                                                     │
│  DoorDash WPR: 2D spatial layout                                    │
│  ┌─────────────────────────────────┐                                │
│  │ HERO STORE                      │ Weight = 3.0                   │
│  └─────────────────────────────────┘                                │
│  ┌───────────┐  ┌───────────┐                                       │
│  │ STORE 2   │  │ STORE 3   │        Weight = 1.5 each              │
│  └───────────┘  └───────────┘                                       │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                        │
│  │ DISH 1    │  │ DISH 2    │  │ DISH 3    │ Weight = 0.5 each      │
│  └───────────┘  └───────────┘  └───────────┘                        │
│                                                                     │
│  WPR = Σ (relevance_i × position_weight_i) / Σ position_weight_i    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Why This Matters for idealo**:

idealo's search results aren't a simple list — they have:
- Hero product cards
- Price comparison tables
- Merchant listings
- Related products

WPR accounts for this spatial arrangement.

### AutoEval Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTOEVAL PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. QUERY SAMPLING                                                  │
│     └── Stratified by intent, frequency, geography, daypart         │
│                                                                     │
│  2. PROMPT CONSTRUCTION                                             │
│     └── Structured context (store, menu, metadata)                  │
│     └── Task-specific templates (dish search, cuisine search)       │
│                                                                     │
│  3. LLM INFERENCE                                                   │
│     └── Base or fine-tuned GPT-4o                                   │
│                                                                     │
│  4. WPR AGGREGATION                                                 │
│     └── Individual judgments → page-level score                     │
│                                                                     │
│  5. AUDITING                                                        │
│     └── Human review of sampled judgments                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Prompt Engineering Insights

> "We experimented with various prompting strategies, including zero-shot, few-shot, and structured templates. We found that task-specific structured prompts paired with rule-based logic and domain-specific examples offered the most consistent, interpretable, and human-aligned results."

**Their Techniques**:

1. **Chain-of-thought reasoning**: Break rating into steps (exact match → substitute → off-target)
2. **Contextual grounding**: Include rich metadata in prompts
3. **Embedded guidelines**: Put evaluation criteria directly in prompts

### Fine-Tuning Success

> "Fine-tuned GPT-4o consistently match or outperform external raters in key relevance tasks."

They created a **golden dataset** with expert annotations, then fine-tuned the judge model on it.

**Result**: The fine-tuned model was **more accurate than crowd annotators**.

### Human-in-the-Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEEDBACK LOOP                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Internal experts generate golden data                           │
│                    ↓                                                │
│  2. Model is fine-tuned and evaluated                               │
│                    ↓                                                │
│  3. External raters audit outputs                                   │
│                    ↓                                                │
│  4. Experts analyze flagged outputs and refine prompts              │
│                    ↓                                                │
│  Loop continues...                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Application to idealo

| DoorDash Practice | idealo Equivalent |
|-------------------|-------------------|
| WPR for search pages | Weighted relevance for product comparison pages |
| Store/dish/cuisine evaluation | Product/price/merchant evaluation |
| Fine-tuned GPT-4o judge | Fine-tuned judge for product relevance |
| Stratified sampling | Sample across categories, price ranges, query types |

---

## 3. Segment — LLM-as-Judge for Code Generation

### Company Context

- **Product**: Customer Data Platform (CDP)
- **Feature**: CustomerAI Audiences — natural language to audience definitions
- **Challenge**: Multiple valid AST (Abstract Syntax Tree) representations for the same audience

### The Core Challenge

> "The fundamental challenge Segment faced was how to evaluate a generative AI system when there can be an unbounded set of 'right answers.'"

**Example**:

```
User input: "Customers who purchased at least 1 time"

Valid outputs (semantically equivalent):
- purchase_count >= 1
- purchase_count > 0
- purchase_count >= 1 AND purchase_count <= ∞
- NOT(purchase_count == 0)
```

Traditional evaluation (exact match) would fail here. This is identical to idealo's challenge: "best price for iPhone" can be answered in many valid ways.

### LLM-as-Judge Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SEGMENT'S MULTI-AGENT SYSTEM                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐                                            │
│  │ GROUND TRUTH AST    │ (from UI, known correct)                   │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │ QUESTION GENERATOR  │ Generate: "Customers who purchased..."    │
│  │ AGENT               │                                            │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │ AST GENERATOR       │ Generate AST from natural language         │
│  │ AGENT               │ (This is the system being evaluated)       │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │ JUDGE AGENT         │ Compare generated AST to ground truth      │
│  │ (GPT-4)             │ Score: 1-5                                 │
│  └─────────────────────┘                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Synthetic Data Generation

Clever approach: They had ASTs from the UI but no corresponding natural language prompts.

**Solution**: Use an LLM to **reverse engineer** prompts from ASTs.

```
AST: {"condition": {"field": "purchase_count", "operator": ">=", "value": 1}}

LLM generates prompt: "Customers who have made at least one purchase"
```

This created a large evaluation dataset without expensive human annotation.

### Scoring Insights

> "LLMs struggle with continuous scores. When asked to provide scores from 0 to 100, models tend to output only discrete values like 0 and 100."

**Their Solution**: Use 1-5 discrete scale.

| Score | Meaning |
|-------|---------|
| 1 | Very bad - completely wrong |
| 2 | Bad - major issues |
| 3 | Acceptable - some issues |
| 4 | Good - minor issues |
| 5 | Perfect - exact semantic match |

### Chain-of-Thought Impact

> "Implementing Chain of Thought prompting improved alignment with human evaluators from approximately 89% to 92%."

A 3% improvement sounds small, but at scale it's significant.

### Model Comparison Results

| Model | Score (out of 5.0) |
|-------|-------------------|
| Claude | 4.02 |
| GPT-4-8k | 4.53 |
| **GPT-4-32k** | **4.55** |

> "There was remarkable similarity in scores between the 8K and 32K context length versions of GPT-4."

**Lesson**: More context doesn't always help.

### Overall Alignment

> "The overall LLM Judge Evaluation system achieved over 90% alignment with human evaluation."

This became their benchmark for production deployment.

### Application to idealo

| Segment Practice | idealo Equivalent |
|------------------|-------------------|
| Synthetic prompt generation | Generate test queries from product catalog |
| AST semantic equivalence | Product recommendation semantic equivalence |
| 1-5 discrete scoring | Binary or 1-5 for relevance judging |
| Model comparison framework | Compare Claude vs GPT-4 for idealo's tasks |

---

## 4. Instacart LACE — Customer Support Chatbot Evaluation

### Company Context

- **Product**: Grocery delivery platform
- **Feature**: AI-powered customer support chatbot
- **Challenge**: Evaluate chatbot quality across multiple dimensions

### The Quote That Matters

> "A chatbot can only be as good as our ability to measure whether it's actually helping customers in real conversations."

This should be idealo's mantra for any GenAI feature.

### Five-Dimension Evaluation Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LACE EVALUATION DIMENSIONS                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. QUERY UNDERSTANDING                                             │
│     └── Did the chatbot understand what the user wanted?            │
│                                                                     │
│  2. ANSWER CORRECTNESS                                              │
│     ├── Contextual Relevancy                                        │
│     ├── Factual Correctness                                         │
│     ├── Consistency                                                 │
│     └── Usefulness                                                  │
│                                                                     │
│  3. CHAT EFFICIENCY                                                 │
│     ├── Conciseness                                                 │
│     └── Task completion speed                                       │
│                                                                     │
│  4. CLIENT SATISFACTION                                             │
│     └── Would user be happy with this response?                     │
│                                                                     │
│  5. COMPLIANCE                                                      │
│     ├── Professionalism / Tone                                      │
│     ├── Safety (no harmful content)                                 │
│     └── Policy adherence                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Binary Scoring Decision

> "Binary evaluations provide greater consistency, simplicity, and alignment with human judgment. While a 1-10 scale might seem more precise, binary evaluations require less extensive prompt engineering while maintaining robust performance."

**Why Binary Works**:
- Less ambiguity ("Is this good?" → Yes/No)
- Higher inter-rater agreement
- Easier to calibrate

### Three Evaluation Methods

**1. Direct Prompting**
- Single-pass evaluation
- Fast but less nuanced
- Good for simple criteria

**2. Agentic via Reflection**
```
Step 1: Initial evaluation
Step 2: Self-critique — "What did I miss?"
Step 3: Revised evaluation
```

**3. Agentic via Debate**
```
Customer Agent: Criticizes harshly
Support Agent: Defends the chatbot
Judge Agent: Makes final decision
```

> "Using an agentic, debate-style approach, we achieved near-perfect accuracy for these criteria."

### Criteria Complexity Tiers

**Tier 1: Simple Criteria** (e.g., professionalism)
- Universal standards
- Near-perfect automated accuracy
- Example: "Is the tone professional?"

**Tier 2: Context-Dependent Criteria** (e.g., contextual relevancy)
- Require domain knowledge
- ~90% accuracy with embedded knowledge
- Example: "Does the chatbot understand Instacart's refund policy?"

**Tier 3: Subjective Criteria** (e.g., conciseness)
- Vary by human preference
- Most challenging to automate
- Example: "Is the response appropriately brief?"

**Their Wisdom on Tier 3**:

> "Instead of investing significant effort in refining ambiguous evaluation criteria (a low-ROI path), we focus on directly improving the chatbot's behavior through prompt refinement and model fine-tuning."

**Translation**: If you can't measure it reliably, just make the chatbot better and skip that metric.

### Prompt Formatting Insight

> "We paid special attention to how we structured our prompts and adopted industry best practices — preparing them in Markdown format with clearly organized sections to delineate content and convey instructions."

Markdown-formatted prompts outperform plain text.

### Decoupling Evaluation from Formatting

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TWO-STEP EVALUATION                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Generate free-form evaluation (no JSON constraint)         │
│                                                                     │
│  "This response is good because it addresses the user's            │
│   refund question directly. However, it could be more concise..."   │
│                                                                     │
│  Step 2: Parse into structured format                               │
│                                                                     │
│  {                                                                  │
│    "relevancy": true,                                               │
│    "conciseness": false,                                            │
│    "reasoning": "..."                                               │
│  }                                                                  │
│                                                                     │
│  WHY: Structured output constraints can hurt reasoning quality      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Real Example of LACE Finding Issues

From their blog:

```
[USER]: I'm attempting to contact the shopper and I'm not getting a response.
[ASSISTANT]: To chat with your shopper... [gives instructions]

[USER]: Representative
[ASSISTANT]: Our specialists are available... Would you like to see if I can help?

[USER]: Continue where we left off
[ASSISTANT]: Our specialists are available... Would you like to see if I can help?
```

**LACE Findings**:
1. **Failure to maintain context** (Answer Correctness dimension)
2. **Inefficient responses** (Chat Efficiency dimension)

**Action**: Updated assistant to better handle conversational continuity.

### Application to idealo

| Instacart Practice | idealo Equivalent |
|--------------------|-------------------|
| 5-dimension framework | Product relevance, price accuracy, completeness, UX, compliance |
| Binary scoring | TRUE/FALSE for "Is this the best price?" |
| Debate-style evaluation | Critical evaluator vs. supportive evaluator |
| Context-dependent criteria | idealo-specific rules (price comparison logic) |

---

## 5. trivago — AI Adoption at Company Scale

### Company Context

- **Product**: Hotel price comparison
- **Scale**: ~700 employees
- **Challenge**: Not evaluation per se, but **AI adoption across the organization**

### The Vision

> "trivago's vision is to empower its almost 700 talents to have the impact of 7,000 — with AI as a key enabler."

### Adoption Metrics

| Metric | 2023 | 2025 | Change |
|--------|------|------|--------|
| AI tool usage | 55% | 90% | +35pp |
| Days saved/person/year | 8 | 16 | 2x |
| Positive sentiment | ~60% | 70% | +10pp |

### AI Ambassadors Program

Cross-functional group that:
- Meets bi-weekly
- Shares knowledge
- Benchmarks solutions
- Ensures strategic adoption

> "Not just technical experts — but acting as connectors, educators, and advocates for a smarter, more collaborative trivago."

### Key Use Cases

| Application | Impact |
|-------------|--------|
| **IT Support Chatbot** | 35% automatic resolution rate |
| **Competitive Intelligence** | 2x competitor coverage |
| **Illustration AI Agent** | Instant on-brand image generation |

### Governance Framework

**Three-Path Procurement**:
- **Green**: Low risk, fast approval
- **Yellow**: Medium risk, security review
- **Red**: High risk, full audit

**trv-AI Radar**: Live dashboard categorizing 42 approved AI tools by function and adoption status.

### The Mindset Shift

> "We're shifting from a '+AI' mindset to 'AI+'"

**+AI**: Add AI to existing processes  
**AI+**: Reimagine work with AI at the center

### Application to idealo

| trivago Practice | idealo Equivalent |
|------------------|-------------------|
| AI Ambassadors | AI Booster Team (already exists!) |
| Days saved metric | Quantify AI Booster Team's impact |
| Tool governance | Standardize evaluation tools and processes |
| Internal chatbot | IT/internal support automation |

---

## 6. Synthesis: Common Patterns

### Pattern 1: LLM-as-Judge is Universal

Every company uses LLM-as-Judge:
- GitHub: GPT-4 evaluates chat answers
- DoorDash: GPT-4o evaluates search relevance
- Segment: GPT-4 evaluates AST semantic equivalence
- Instacart: Multi-agent debate with LLMs

**Takeaway**: Master LLM-as-Judge — it's the core technique.

### Pattern 2: Human Calibration is Non-Negotiable

Every company maintains human oversight:
- GitHub: "Routinely audit the outputs of this LLM"
- DoorDash: "External raters audit outputs"
- Segment: "90% alignment with human evaluation" benchmark
- Instacart: "Iterative validation process where human evaluators rated conversations"

**Takeaway**: Build feedback loops, not one-time evaluations.

### Pattern 3: Binary Beats Continuous

Both Instacart and Segment discovered this:
- Segment: "LLMs struggle with continuous scores"
- Instacart: "Binary evaluations provide greater consistency"

**Takeaway**: Use 1-5 scale max, or TRUE/FALSE when possible.

### Pattern 4: Domain-Specific Criteria Matter

Generic evaluation doesn't work:
- DoorDash: Dish-to-store vs. cuisine-to-store require different prompts
- Instacart: Instacart-specific policies need embedded knowledge
- Segment: AST equivalence requires specialized logic

**Takeaway**: Build idealo-specific evaluation criteria.

### Pattern 5: Efficiency Metrics Are Critical

Everyone tracks cost/latency alongside quality:
- GitHub: "Token usage is one of our main model performance measures"
- DoorDash: "98% reduction in turnaround time"
- trivago: "16 days saved per person per year"

**Takeaway**: Quality without efficiency is useless.

### Pattern 6: The Golden Dataset

Every company creates and maintains a golden dataset:
- GitHub: 100 containerized repos with passing tests
- DoorDash: Expert-labeled relevance judgments
- Segment: Human-created AST ↔ prompt pairs
- Instacart: Human-rated conversations

**Takeaway**: Invest in golden data early.

---

## 7. Applying These Lessons to idealo

### idealo's GenAI Use Cases (Hypothesized)

Based on the job description:

1. **Search enhancement**: Natural language product search
2. **Product comparison**: AI-generated comparison summaries
3. **Price alerts**: Intelligent notification system
4. **Customer support**: Chatbot for product questions
5. **Internal tools**: Analytics, reporting automation

### Recommended Evaluation Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IDEALO EVALUATION FRAMEWORK                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SEARCH & DISCOVERY (inspired by DoorDash)                          │
│  ├── Whole-Page Relevance for product grids                         │
│  ├── Price accuracy (is the "best price" actually best?)            │
│  └── Query understanding (extracted intent correct?)                │
│                                                                     │
│  PRODUCT CONTENT (inspired by Instacart)                            │
│  ├── Factual correctness (specs match source data?)                 │
│  ├── Completeness (all key attributes mentioned?)                   │
│  └── Faithfulness (no hallucinated features?)                       │
│                                                                     │
│  CHATBOT (inspired by Instacart)                                    │
│  ├── Query understanding                                            │
│  ├── Answer correctness                                             │
│  ├── Efficiency                                                     │
│  └── Compliance                                                     │
│                                                                     │
│  CROSS-CUTTING                                                      │
│  ├── Latency (target: <1s for search, <2s for chat)                 │
│  ├── Cost per query (budget: <€0.02)                                │
│  └── User satisfaction (thumbs up rate: >80%)                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Roadmap

**Phase 1: Foundation (Weeks 1-4)**
- Create golden dataset (500+ queries with expert labels)
- Build basic LLM-as-Judge pipeline
- Establish baseline metrics

**Phase 2: Automation (Weeks 5-8)**
- Implement multi-criteria evaluation (like LACE)
- Add human audit sampling
- Deploy continuous monitoring

**Phase 3: Optimization (Weeks 9-12)**
- Fine-tune judge model on idealo data
- Implement WPR for search pages
- Build Pareto analysis for model selection

### Interview Talking Points

When discussing case studies:

1. **GitHub**: "I studied how GitHub evaluates Copilot — they run 4,000+ offline tests and use a proxy server that lets them switch models without changing product code. At idealo, we could build similar infrastructure for rapid model iteration."

2. **DoorDash**: "DoorDash's WPR metric is directly applicable to idealo. Search results aren't a simple list — WPR weights results by visual prominence. Their fine-tuned GPT-4o outperformed crowd annotators, showing the value of domain-specific training."

3. **Segment**: "Segment's insight about LLMs struggling with continuous scores changed my thinking. I'd use binary or 1-5 discrete scales, not 0-100. Their 90% alignment with human evaluation is a good benchmark to target."

4. **Instacart**: "Instacart's LACE framework is comprehensive — five dimensions covering query understanding through compliance. I'd adapt this for idealo's product-focused use cases, with emphasis on price accuracy and factual correctness."

5. **trivago**: "trivago is the closest comparable company — both are price comparison platforms. Their AI Ambassadors program achieved 90% AI tool adoption. The AI Booster Team at idealo could follow a similar playbook for company-wide GenAI enablement."

---

## Summary: Key Quotes to Remember

| Company | Quote |
|---------|-------|
| **GitHub** | "Just because a model is newer doesn't mean it will perform better for your use case." |
| **GitHub** | "Higher latency might actually lead to a higher acceptance rate because users might see fewer suggestions total." |
| **DoorDash** | "AutoEval has reduced relevance judgment turnaround time by 98%." |
| **DoorDash** | "Fine-tuned GPT-4o outperformed external raters in overall accuracy." |
| **Segment** | "LLMs struggle with continuous scores." |
| **Segment** | "The system achieved over 90% alignment with human evaluation." |
| **Instacart** | "A chatbot can only be as good as our ability to measure whether it's actually helping." |
| **Instacart** | "Binary evaluations provide greater consistency, simplicity, and alignment with human judgment." |
| **trivago** | "We're shifting from a '+AI' mindset to 'AI+'." |

---

*Next: [05_INTERVIEW_QUESTIONS.md](05_INTERVIEW_QUESTIONS.md)*
