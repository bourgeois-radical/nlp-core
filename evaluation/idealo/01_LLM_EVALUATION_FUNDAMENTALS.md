# LLM Evaluation Fundamentals

> **Purpose**: Understand how to measure LLM quality systematically  
> **Key Insight**: "A chatbot can only be as good as our ability to measure whether it's actually helping" — Instacart

## Table of Contents
1. [The Evaluation Challenge](#1-the-evaluation-challenge)
2. [Reference-Based Metrics](#2-reference-based-metrics)
3. [Reference-Free Evaluation](#3-reference-free-evaluation)
4. [LLM-as-Judge Framework](#4-llm-as-judge-framework)
5. [Task-Specific Metrics](#5-task-specific-metrics)
6. [Human-in-the-Loop Systems](#6-human-in-the-loop-systems)
7. [Practical Implementation](#7-practical-implementation)
8. [Common Pitfalls](#8-common-pitfalls)

---

## 1. The Evaluation Challenge

### Why LLM Evaluation is Hard

Traditional ML evaluation is straightforward: compare predictions to ground truth labels. LLM evaluation faces unique challenges:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WHY LLM EVALUATION IS HARD                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. MULTIPLE VALID OUTPUTS                                          │
│     "What's 2+2?" → "4" OR "Four" OR "The answer is 4"              │
│                                                                     │
│  2. SEMANTIC vs LEXICAL SIMILARITY                                  │
│     "The cat sat on the mat" ≈ "A feline rested on the rug"         │
│     (Same meaning, zero word overlap)                               │
│                                                                     │
│  3. SUBJECTIVE QUALITY                                              │
│     What makes a "good" summary? Concise? Comprehensive? Engaging?  │
│                                                                     │
│  4. STOCHASTIC OUTPUTS                                              │
│     Same prompt → different outputs (temperature > 0)               │
│                                                                     │
│  5. CONTEXT DEPENDENCE                                              │
│     Quality depends on use case, user, domain                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The Segment Insight

> "The fundamental challenge was how to evaluate a generative AI system when there can be an unbounded set of 'right answers.'"

Example: "Customers who purchased at least 1 time" can also be expressed as:
- "Customers who purchased more than 0 times"
- "Customers who purchased 1 or more times"
- "Customers with purchase_count >= 1"

All semantically equivalent, but lexically different.

### Evaluation Taxonomy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION METHODS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  REFERENCE-BASED (need ground truth)                                │
│  ├── BLEU, ROUGE, METEOR (n-gram overlap)                           │
│  ├── BERTScore (semantic similarity)                                │
│  └── Exact Match (structured outputs)                               │
│                                                                     │
│  REFERENCE-FREE (no ground truth needed)                            │
│  ├── LLM-as-Judge (GPT-4 rates outputs)                             │
│  ├── Pairwise Comparison (A vs B)                                   │
│  └── Human Evaluation (gold standard)                               │
│                                                                     │
│  TASK-SPECIFIC                                                      │
│  ├── Factuality (claim verification)                                │
│  ├── Relevance (query-response alignment)                           │
│  ├── Safety (toxicity, bias detection)                              │
│  └── Efficiency (latency, cost, tokens)                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Reference-Based Metrics

### BLEU (Bilingual Evaluation Understudy)

Originally designed for machine translation, BLEU measures n-gram overlap between candidate and reference.

**Formula:**

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where:
- $p_n$ = precision of n-grams
- $w_n$ = weight (typically $\frac{1}{N}$)
- BP = brevity penalty

**Brevity Penalty:**

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1-r/c} & \text{if } c \leq r \end{cases}$$

Where $c$ = candidate length, $r$ = reference length.

**Modified Precision (clips to max reference count):**

$$p_n = \frac{\sum_{\text{n-gram} \in C} \min(\text{Count}_{C}(\text{n-gram}), \text{MaxRef}(\text{n-gram}))}{\sum_{\text{n-gram} \in C} \text{Count}_{C}(\text{n-gram})}$$

**Example:**
```
Reference: "The cat sat on the mat"
Candidate: "The cat sat on the mat"
→ BLEU-4 = 1.0 (perfect match)

Candidate: "A cat was sitting on a mat"
→ BLEU-4 ≈ 0.3 (semantically similar, low n-gram overlap)
```

**When to use BLEU:**
- ✅ Translation quality
- ✅ Code generation (syntax matters)
- ❌ Open-ended generation
- ❌ Summarization (paraphrasing is good)

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Focuses on recall rather than precision — did we capture the important content?

**ROUGE-N (n-gram recall):**

$$\text{ROUGE-N} = \frac{\sum_{s \in \text{Ref}} \sum_{\text{gram}_n \in s} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{s \in \text{Ref}} \sum_{\text{gram}_n \in s} \text{Count}(\text{gram}_n)}$$

**ROUGE-L (Longest Common Subsequence):**

$$\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot R_{lcs} \cdot P_{lcs}}{R_{lcs} + \beta^2 \cdot P_{lcs}}$$

Where:
- $R_{lcs} = \frac{\text{LCS}(X, Y)}{m}$ (recall)
- $P_{lcs} = \frac{\text{LCS}(X, Y)}{n}$ (precision)
- $\beta$ controls recall/precision balance

**Variants:**
| Metric | Focus | Best For |
|--------|-------|----------|
| ROUGE-1 | Unigrams | Content coverage |
| ROUGE-2 | Bigrams | Phrase matching |
| ROUGE-L | Subsequence | Sentence structure |
| ROUGE-S | Skip-bigrams | Flexible ordering |

**When to use ROUGE:**
- ✅ Summarization
- ✅ Document generation
- ❌ Translation
- ❌ Creative writing

### BERTScore

Uses BERT embeddings to compute semantic similarity, not just lexical overlap.

**Algorithm:**

1. Encode reference and candidate with BERT
2. Compute pairwise cosine similarity between all tokens
3. Greedily match tokens to maximize similarity
4. Aggregate into precision, recall, F1

**Formula:**

$$P_{\text{BERT}} = \frac{1}{|\hat{x}|} \sum_{\hat{x}_i \in \hat{x}} \max_{x_j \in x} \cos(\mathbf{h}_{\hat{x}_i}, \mathbf{h}_{x_j})$$

$$R_{\text{BERT}} = \frac{1}{|x|} \sum_{x_j \in x} \max_{\hat{x}_i \in \hat{x}} \cos(\mathbf{h}_{x_j}, \mathbf{h}_{\hat{x}_i})$$

$$F_{\text{BERT}} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$

**Advantages over BLEU/ROUGE:**
- Captures semantic similarity
- Handles paraphrasing
- More aligned with human judgment

**Python Example:**
```python
from bert_score import score

references = ["The cat sat on the mat"]
candidates = ["A feline rested on the rug"]

P, R, F1 = score(candidates, references, lang="en")
print(f"BERTScore F1: {F1.mean():.4f}")
# Output: BERTScore F1: 0.8547 (semantically similar)
```

### Comparison Table

| Metric | Type | Handles Paraphrase | Computational Cost |
|--------|------|-------------------|-------------------|
| BLEU | Precision-based | ❌ | Low |
| ROUGE | Recall-based | ❌ | Low |
| BERTScore | Semantic | ✅ | Medium |
| Human | Gold standard | ✅ | High |

---

## 3. Reference-Free Evaluation

When you don't have ground truth, you need alternative approaches.

### LLM-as-Judge (Core Concept)

Use a powerful LLM (e.g., GPT-4) to evaluate outputs from another LLM.

**Basic Pattern:**
```python
def llm_as_judge(output, criteria):
    prompt = f"""
    Rate the following output on a scale of 1-5 for {criteria}.
    
    Output: {output}
    
    Provide:
    - Score (1-5)
    - Brief reasoning
    """
    return call_gpt4(prompt)
```

**Segment's Discovery:**
> "LLMs struggle with continuous scores. When asked to provide scores from 0 to 100, models tend to output only discrete values like 0 and 100."

**Solution: Use discrete scales (1-5)**

| Score | Meaning |
|-------|---------|
| 1 | Very bad - completely wrong |
| 2 | Bad - major issues |
| 3 | Acceptable - some issues |
| 4 | Good - minor issues |
| 5 | Excellent - no issues |

### Chain-of-Thought for Evaluation

Adding reasoning improves accuracy significantly.

**Without CoT:**
```
Score: 4
```

**With CoT:**
```
Let me evaluate step by step:
1. Factual accuracy: The response correctly states X and Y. ✓
2. Completeness: Missing important detail about Z. ✗
3. Clarity: Well-structured and easy to understand. ✓
4. Relevance: Directly addresses the user's question. ✓

Based on this analysis, the response is good but incomplete.
Score: 4
```

**Impact (from Segment):**
> "Implementing Chain of Thought prompting improved alignment with human evaluators from approximately 89% to 92%."

### Pairwise Comparison

Instead of absolute scores, compare two outputs directly.

**Prompt:**
```
Given the query: "{query}"

Which response is better?

Response A: {response_a}
Response B: {response_b}

Choose: A, B, or Tie
Explain your reasoning.
```

**Advantages:**
- More consistent than absolute scoring
- Natural for model comparison
- Handles subjective criteria better

**Bradley-Terry Model for Ranking:**

Given pairwise comparisons, estimate global rankings:

$$P(i \succ j) = \frac{e^{s_i}}{e^{s_i} + e^{s_j}} = \sigma(s_i - s_j)$$

**Implementation:**
```python
from scipy.optimize import minimize

def bradley_terry_mle(comparisons, n_items):
    """
    comparisons: list of (winner, loser) tuples
    Returns: estimated skill scores for each item
    """
    def neg_log_likelihood(scores):
        nll = 0
        for winner, loser in comparisons:
            nll -= np.log(1 / (1 + np.exp(scores[loser] - scores[winner])))
        return nll
    
    result = minimize(neg_log_likelihood, np.zeros(n_items))
    return result.x
```

### Agentic Evaluation Methods

From Instacart's LACE framework:

**Method 1: Direct Prompting**
- Single-pass evaluation
- Fast but less nuanced
- Good for simple criteria

**Method 2: Reflection**
1. Initial evaluation
2. Self-critique: "What did I miss?"
3. Revised evaluation

```python
def evaluate_with_reflection(output, criteria):
    # Initial evaluation
    initial = llm_evaluate(output, criteria)
    
    # Reflection
    reflection_prompt = f"""
    You gave this evaluation: {initial}
    
    Critically review your assessment:
    - Did you miss any important aspects?
    - Were you too harsh or too lenient?
    - What biases might have affected your judgment?
    
    Provide a revised evaluation if needed.
    """
    
    revised = call_llm(reflection_prompt)
    return revised
```

**Method 3: Multi-Agent Debate**
- Customer Agent: Critiques harshly
- Support Agent: Defends the response
- Judge Agent: Makes final decision

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT EVALUATION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐         ┌─────────────┐                            │
│  │  CUSTOMER   │         │   SUPPORT   │                            │
│  │   AGENT     │         │    AGENT    │                            │
│  │ (Critical)  │         │ (Defensive) │                            │
│  └──────┬──────┘         └──────┬──────┘                            │
│         │                       │                                   │
│         │   Both assess the     │                                   │
│         │   chatbot response    │                                   │
│         │                       │                                   │
│         └───────────┬───────────┘                                   │
│                     │                                               │
│                     ▼                                               │
│              ┌─────────────┐                                        │
│              │    JUDGE    │                                        │
│              │    AGENT    │                                        │
│              │ (Impartial) │                                        │
│              └──────┬──────┘                                        │
│                     │                                               │
│                     ▼                                               │
│              Final Evaluation                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. LLM-as-Judge Framework

### Building a Robust Judge System

**Key Components:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM-AS-JUDGE PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. PROMPT ENGINEERING                                              │
│     ├── Clear criteria definitions                                  │
│     ├── Structured output format                                    │
│     ├── Chain-of-thought reasoning                                  │
│     └── Few-shot examples                                           │
│                                                                     │
│  2. SCORING DESIGN                                                  │
│     ├── Discrete scales (1-5) not continuous                        │
│     ├── Binary for simple criteria (True/False)                     │
│     └── Multiple criteria aggregated                                │
│                                                                     │
│  3. VARIANCE REDUCTION                                              │
│     ├── Multiple runs (3-11)                                        │
│     ├── Temperature = 0 for reproducibility                         │
│     └── Majority voting                                             │
│                                                                     │
│  4. HUMAN CALIBRATION                                               │
│     ├── Golden dataset with human labels                            │
│     ├── Regular auditing of LLM judgments                           │
│     └── Iterative prompt refinement                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Your LLM-as-Judge Implementation (Interview Story)

```python
def robust_llm_judge(output, criteria, n_runs=5):
    """
    LLM-as-Judge with statistical validation.
    
    Key innovations:
    1. Multiple runs to reduce variance
    2. Majority voting for final score
    3. Distribution analysis for confidence
    4. Human review trigger for edge cases
    """
    scores = []
    
    for _ in range(n_runs):
        score = single_llm_evaluation(output, criteria)
        scores.append(score)
    
    # Check distribution consistency
    score_std = np.std(scores)
    
    if score_std > THRESHOLD:
        # High variance = uncertain → flag for human review
        flag_for_human_review(output, scores)
        return None, "NEEDS_HUMAN_REVIEW"
    
    # Majority vote
    final_score = statistics.mode(scores)
    
    # Statistical validation
    # Binomial test: is the majority statistically significant?
    majority_count = scores.count(final_score)
    p_value = binomtest(majority_count, n_runs, 1/5).pvalue  # 5 possible scores
    
    confidence = "HIGH" if p_value < 0.05 else "MEDIUM"
    
    return final_score, confidence
```

**Key Innovation: Distribution Validation**

Beyond simple majority voting, analyze the distribution:

```python
def validate_score_distribution(scores):
    """
    Check if scores are consistent across runs.
    
    Cases:
    1. [4, 4, 4, 4, 4] → Perfect agreement, high confidence
    2. [3, 4, 4, 4, 5] → Minor disagreement, acceptable
    3. [1, 2, 4, 5, 5] → Bimodal, flag for review
    """
    
    # Calculate entropy
    from collections import Counter
    counts = Counter(scores)
    probs = [c / len(scores) for c in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    
    # Low entropy = high agreement
    if entropy < 0.5:
        return "HIGH_CONFIDENCE"
    elif entropy < 1.5:
        return "MEDIUM_CONFIDENCE"
    else:
        return "LOW_CONFIDENCE_NEEDS_REVIEW"
```

### Evaluation Criteria Design (from Instacart's LACE)

**Five Dimensions:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION DIMENSIONS                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. QUERY UNDERSTANDING                                             │
│     └── Did the system understand what the user wanted?             │
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

**Binary Scoring is Better:**

From Instacart:
> "Binary evaluations provide greater consistency, simplicity, and alignment with human judgment. While a 1-10 scale might seem more precise, binary evaluations require less extensive prompt engineering while maintaining robust performance."

### Prompt Design Best Practices

**Structure:**
```markdown
# Task Description
You are evaluating a chatbot response for [CRITERIA].

# Rating Scale
- TRUE: The response meets the criteria
- FALSE: The response does not meet the criteria

# Evaluation Guidelines
[Specific rules for this criteria]

# Context
Query: {user_query}
Conversation History: {history}
Chatbot Response: {response}

# Instructions
1. Analyze the response step by step
2. Consider edge cases
3. Provide your rating with reasoning

# Output Format
Rating: [TRUE/FALSE]
Reasoning: [Your explanation]
```

**Instacart's Insight on Prompt Formatting:**
> "We paid special attention to how we structured our prompts and adopted industry best practices — preparing them in Markdown format with clearly organized sections."

---

## 5. Task-Specific Metrics

### Factuality / Hallucination Detection

**Definition**: Does the output contain claims not supported by the source?

**Approach 1: Claim Extraction + Verification**
```python
def detect_hallucinations(response, source_docs):
    # Step 1: Extract claims from response
    claims = extract_claims(response)
    # e.g., ["The product weighs 5kg", "It has a 2-year warranty"]
    
    # Step 2: Verify each claim against source
    hallucinations = []
    for claim in claims:
        if not is_supported_by_source(claim, source_docs):
            hallucinations.append(claim)
    
    # Step 3: Calculate rate
    hallucination_rate = len(hallucinations) / len(claims)
    
    return {
        "hallucination_rate": hallucination_rate,
        "hallucinated_claims": hallucinations
    }
```

**Approach 2: NLI-Based (Natural Language Inference)**

Use an NLI model to check if source ENTAILS response:

```python
from transformers import pipeline

nli = pipeline("text-classification", model="facebook/bart-large-mnli")

def check_entailment(premise, hypothesis):
    result = nli(f"{premise} [SEP] {hypothesis}")
    # Returns: entailment, neutral, or contradiction
    return result
```

**Approach 3: LLM-as-Judge for Factuality**

```
Given the source document:
{source}

Evaluate this response:
{response}

Does the response contain any facts that are:
1. Not mentioned in the source document
2. Contradicted by the source document
3. Extrapolated beyond what the source supports

For each such fact, explain why it's problematic.
```

### Relevance Metrics

**Embedding Similarity:**

$$\text{Relevance}(q, r) = \cos(\mathbf{e}_q, \mathbf{e}_r) = \frac{\mathbf{e}_q \cdot \mathbf{e}_r}{\|\mathbf{e}_q\| \|\mathbf{e}_r\|}$$

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_relevance(query, response):
    q_embedding = model.encode(query)
    r_embedding = model.encode(response)
    
    similarity = np.dot(q_embedding, r_embedding) / (
        np.linalg.norm(q_embedding) * np.linalg.norm(r_embedding)
    )
    
    return similarity
```

### Safety Metrics

**Categories:**
- Toxicity (hate speech, violence)
- Bias (demographic, political)
- PII Leakage (names, addresses)
- Harmful Instructions (weapons, self-harm)

**GitHub's Approach:**
> "Regardless of the model used, GitHub Copilot tests both prompts and responses for relevance, such as non-code related questions, and toxic language such as hate speech, sexual content, violence, and evidence of self-harm."

### Cost and Efficiency Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Latency P50 | Median response time | < 1s |
| Latency P99 | 99th percentile | < 3s |
| Tokens/Request | Input + Output tokens | Minimize |
| Cost/Request | Tokens × Price/Token | Minimize |
| Success Rate | Successful / Total | > 99% |

**Cost Calculation:**

$$\text{Cost} = (\text{Input Tokens} \times P_{\text{input}}) + (\text{Output Tokens} \times P_{\text{output}})$$

Example (GPT-4):
```
Input: 1000 tokens × $0.03/1K = $0.03
Output: 500 tokens × $0.06/1K = $0.03
Total: $0.06 per request
```

---

## 6. Human-in-the-Loop Systems

### The Calibration Loop

From Instacart:
```
┌──────────────────────────────────────────────────────────────────┐
│                HUMAN-IN-THE-LOOP FEEDBACK LOOP                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Internal experts generate golden data                        │
│                    ↓                                             │
│  2. Model is fine-tuned and evaluated                            │
│                    ↓                                             │
│  3. External raters audit outputs                                │
│                    ↓                                             │
│  4. Experts analyze flagged outputs                              │
│                    ↓                                             │
│  5. Refine prompts or labels                                     │
│                    ↓                                             │
│  Loop continues...                                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Alignment Measurement

**Inter-Annotator Agreement:**

Cohen's Kappa:
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where:
- $p_o$ = observed agreement
- $p_e$ = expected agreement by chance

```python
from sklearn.metrics import cohen_kappa_score

human_labels = [1, 2, 3, 4, 5, 3, 4, 2, 1, 5]
llm_labels   = [1, 2, 3, 4, 4, 3, 4, 2, 2, 5]

kappa = cohen_kappa_score(human_labels, llm_labels)
print(f"Cohen's Kappa: {kappa:.3f}")
```

**Interpretation:**
| Kappa | Interpretation |
|-------|----------------|
| < 0.20 | Poor |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| 0.61-0.80 | Substantial |
| 0.81-1.00 | Almost perfect |

**Target (from Segment):**
> "The overall LLM Judge Evaluation system achieved over 90% alignment with human evaluation for ASTs."

### Criteria Complexity Tiers

From Instacart's experience:

**Tier 1: Simple Criteria**
- Universal standards (tone, politeness)
- Near-perfect accuracy with debate approach
- Example: "Is the response professional?"

**Tier 2: Context-Dependent Criteria**
- Require domain knowledge
- ~90% accuracy with embedded knowledge
- Example: "Does the response correctly explain Instacart's policies?"

**Tier 3: Subjective Criteria**
- Vary by human preference
- Most challenging to automate
- Example: "Is the response appropriately concise?"

**Strategy for Tier 3:**
> "Instead of investing significant effort in refining ambiguous evaluation criteria (a low-ROI path), we focus on directly improving the chatbot's behavior through prompt refinement and model fine-tuning."

---

## 7. Practical Implementation

### GitHub's Evaluation Infrastructure

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GITHUB COPILOT EVALUATION                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SCALE: > 4,000 offline tests                                       │
│                                                                     │
│  INFRASTRUCTURE:                                                    │
│  ├── Custom platform built with GitHub Actions                      │
│  ├── Apache Kafka for data streaming                                │
│  ├── Microsoft Azure for compute                                    │
│  └── Dashboards for result exploration                              │
│                                                                     │
│  KEY INSIGHT:                                                       │
│  "We can test a new model without changing the product code.        │
│   We have a proxy server that can switch API endpoints easily."     │
│                                                                     │
│  TEST TYPES:                                                        │
│  1. Code completion (unit test pass rate)                           │
│  2. Chat quality (1000+ technical questions)                        │
│  3. Safety (toxicity, prompt injection)                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### DoorDash's AutoEval System

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DOORDASH AUTOEVAL PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. QUERY SAMPLING                                                  │
│     └── Stratified by intent, frequency, geography, time           │
│                                                                     │
│  2. PROMPT CONSTRUCTION                                             │
│     └── Structured context (store, menu, metadata)                  │
│                                                                     │
│  3. LLM INFERENCE                                                   │
│     └── Base or fine-tuned model                                    │
│                                                                     │
│  4. WPR AGGREGATION                                                 │
│     └── Whole-Page Relevance metric                                 │
│                                                                     │
│  5. AUDITING & MONITORING                                           │
│     └── Regular human review of samples                             │
│                                                                     │
│  RESULTS:                                                           │
│  - 98% reduction in turnaround time                                 │
│  - 9x increase in capacity                                          │
│  - Fine-tuned GPT-4o outperformed external raters                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Complete Evaluation Pipeline Code

```python
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class EvalResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"

@dataclass
class EvaluationConfig:
    criteria: List[str]
    n_runs: int = 5
    threshold: float = 0.5
    model: str = "gpt-4"
    temperature: float = 0.0

@dataclass
class EvaluationOutput:
    query: str
    response: str
    scores: Dict[str, float]
    result: EvalResult
    confidence: float
    reasoning: str

class LLMEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def evaluate(self, query: str, response: str, context: Optional[str] = None) -> EvaluationOutput:
        """
        Full evaluation pipeline.
        """
        all_scores = {}
        all_reasoning = []
        
        for criterion in self.config.criteria:
            # Multiple runs for variance reduction
            criterion_scores = []
            for _ in range(self.config.n_runs):
                score, reasoning = self._single_evaluation(
                    query, response, criterion, context
                )
                criterion_scores.append(score)
            
            # Majority vote
            final_score = self._aggregate_scores(criterion_scores)
            all_scores[criterion] = final_score
            
            # Check confidence
            confidence = self._calculate_confidence(criterion_scores)
            if confidence < self.config.threshold:
                return EvaluationOutput(
                    query=query,
                    response=response,
                    scores=all_scores,
                    result=EvalResult.NEEDS_REVIEW,
                    confidence=confidence,
                    reasoning="Low confidence - flagged for human review"
                )
        
        # Aggregate across criteria
        overall_pass = all(s >= 0.6 for s in all_scores.values())
        
        return EvaluationOutput(
            query=query,
            response=response,
            scores=all_scores,
            result=EvalResult.PASS if overall_pass else EvalResult.FAIL,
            confidence=min(all_scores.values()),
            reasoning="; ".join(all_reasoning)
        )
    
    def _single_evaluation(self, query, response, criterion, context):
        prompt = self._build_prompt(query, response, criterion, context)
        # Call LLM
        result = call_llm(prompt, temperature=self.config.temperature)
        return self._parse_result(result)
    
    def _aggregate_scores(self, scores: List[float]) -> float:
        """Majority voting for binary, mean for continuous."""
        if all(s in [0.0, 1.0] for s in scores):
            # Binary - majority vote
            return 1.0 if sum(scores) > len(scores) / 2 else 0.0
        else:
            # Continuous - mean
            return sum(scores) / len(scores)
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Higher agreement = higher confidence."""
        if len(set(scores)) == 1:
            return 1.0  # Perfect agreement
        std = np.std(scores)
        return max(0, 1 - std)
    
    def _build_prompt(self, query, response, criterion, context):
        return f"""
# Evaluation Task
Evaluate the following response for: {criterion}

# Context
{f"Background: {context}" if context else ""}

# Conversation
User Query: {query}
System Response: {response}

# Instructions
1. Analyze the response carefully
2. Consider the specific criterion: {criterion}
3. Provide a score from 0.0 to 1.0
4. Explain your reasoning

# Output Format (JSON)
{{
    "score": <float 0.0-1.0>,
    "reasoning": "<explanation>"
}}
"""
```

---

## 8. Common Pitfalls

### Pitfall 1: Over-relying on Automated Metrics

BLEU/ROUGE can be misleading:
```
Reference: "The cat sat on the mat"
Bad output: "mat the on sat cat The"  # High n-gram overlap, nonsense
Good output: "A feline rested on the rug"  # Low overlap, meaningful
```

**Solution**: Combine automated metrics with LLM-as-Judge and human review.

### Pitfall 2: Using Continuous Scores

From Segment:
> "LLMs struggle with continuous scores. When asked to provide scores from 0 to 100, models tend to output only discrete values like 0 and 100."

**Solution**: Use 1-5 scale or binary (True/False).

### Pitfall 3: Not Accounting for LLM Variance

Same prompt → different outputs.

**Solution**: Multiple runs + aggregation.

### Pitfall 4: Circular Evaluation

Using GPT-4 to evaluate GPT-4 outputs may share blind spots.

**Solution**: 
- Use different model families for generation vs evaluation
- Regular human calibration
- Diverse evaluation criteria

### Pitfall 5: Neglecting Edge Cases

Head queries are easy; tail queries reveal problems.

From DoorDash:
> "Annotated datasets overrepresent high-frequency, or head, queries, while underrepresenting tail queries, where relevance problems often hide."

**Solution**: Stratified sampling across query types.

### Pitfall 6: Ignoring Cost Trade-offs

From GitHub:
> "Higher latency might actually lead to a higher acceptance rate because users might see fewer suggestions total."

**Solution**: Consider inverse relationships between metrics.

---

## Summary: Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM EVALUATION CHECKLIST                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ☐ Define clear evaluation criteria                                 │
│  ☐ Use multiple evaluation methods (automated + human)              │
│  ☐ Implement LLM-as-Judge with discrete scales                      │
│  ☐ Add Chain-of-Thought for better reasoning                        │
│  ☐ Use multiple runs for variance reduction                         │
│  ☐ Calibrate against human judgments regularly                      │
│  ☐ Track cost alongside quality metrics                             │
│  ☐ Sample across query distributions (head + tail)                  │
│  ☐ Build feedback loops for continuous improvement                  │
│  ☐ Document everything for reproducibility                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Next: [02_EXPERIMENTATION_CAUSAL_INFERENCE.md](02_EXPERIMENTATION_CAUSAL_INFERENCE.md)*
