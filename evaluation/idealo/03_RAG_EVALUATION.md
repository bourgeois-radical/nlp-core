# RAG Evaluation — Measuring Retrieval-Augmented Generation

> **Purpose**: Evaluate RAG systems end-to-end, from retrieval quality to generation faithfulness  
> **Key Insight**: "RAG is only as good as its weakest link — poor retrieval guarantees poor generation"

## Table of Contents
1. [What is RAG and Why Does It Matter?](#1-what-is-rag-and-why-does-it-matter)
2. [The RAG Pipeline](#2-the-rag-pipeline)
3. [Retrieval Evaluation](#3-retrieval-evaluation)
4. [Generation Evaluation](#4-generation-evaluation)
5. [End-to-End Evaluation](#5-end-to-end-evaluation)
6. [Cost-Quality Trade-offs](#6-cost-quality-trade-offs)
7. [Production Monitoring](#7-production-monitoring)
8. [Case Study: Harvey's Enterprise-Grade RAG](#8-case-study-harveys-enterprise-grade-rag)

---

## 1. What is RAG and Why Does It Matter?

### The Problem RAG Solves

Large Language Models have a fundamental limitation: their knowledge is frozen at training time. GPT-4 doesn't know:
- What happened after its training cutoff
- Your company's internal documents
- Today's prices on idealo.de

**Retrieval-Augmented Generation (RAG)** solves this by retrieving relevant documents at inference time and injecting them into the prompt.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG vs PURE LLM                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PURE LLM:                                                          │
│  User: "What's the best price for iPhone 15 on idealo?"             │
│  LLM: "I don't have access to real-time pricing data..."           │
│       (Fails - knowledge cutoff)                                    │
│                                                                     │
│  RAG:                                                               │
│  User: "What's the best price for iPhone 15 on idealo?"             │
│  [RETRIEVAL] → Fetch current prices from idealo database            │
│  [AUGMENTATION] → Inject prices into prompt                         │
│  [GENERATION] → LLM generates response with current data            │
│  LLM: "The best price for iPhone 15 on idealo is €749 at Amazon..." │
│       (Succeeds - grounded in retrieved data)                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why RAG Matters for idealo

idealo's AI Booster Team is building GenAI features that need access to:
- **Product catalog**: 600M+ product offers
- **Price data**: Real-time prices from 50K merchants
- **User data**: Browsing history, preferences
- **Policies**: Return policies, shipping information

RAG is how these features will access this data reliably.

---

## 2. The RAG Pipeline

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  USER    │───▶│   QUERY      │───▶│  RETRIEVER   │               │
│  │  QUERY   │    │  PROCESSOR   │    │              │               │
│  └──────────┘    └──────────────┘    └───────┬──────┘               │
│                                              │                      │
│                                              ▼                      │
│                                    ┌──────────────────┐             │
│                                    │  VECTOR DATABASE │             │
│                                    │  (Pinecone/Qdrant)│            │
│                                    └────────┬─────────┘             │
│                                              │                      │
│                                              ▼                      │
│                                    ┌──────────────────┐             │
│                                    │   RE-RANKER      │             │
│                                    │   (optional)     │             │
│                                    └────────┬─────────┘             │
│                                              │                      │
│                                              ▼                      │
│                                    ┌──────────────────┐             │
│  ┌──────────┐                      │   GENERATOR      │             │
│  │ RESPONSE │◀─────────────────────│   (LLM)          │             │
│  └──────────┘                      └──────────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Function | Example |
|-----------|----------|---------|
| **Query Processor** | Transform user query | Expand synonyms, extract entities |
| **Retriever** | Find relevant documents | kNN search on embeddings |
| **Re-ranker** | Refine top results | Cross-encoder scoring |
| **Generator** | Produce final answer | GPT-4, Claude |

### The Evaluation Challenge

Each component can fail independently:
- **Retrieval fails**: Wrong documents retrieved → LLM can't answer correctly
- **Retrieval succeeds, generation fails**: Right documents, but LLM hallucinates or ignores them
- **Both succeed technically, but answer is bad**: Correct but unhelpful

You need metrics for **each stage** and **end-to-end**.

---

## 3. Retrieval Evaluation

### Core Metrics

#### Recall@K

**Question**: Of all relevant documents, how many did we retrieve in the top K?

$$\text{Recall@K} = \frac{|\text{Relevant} \cap \text{Retrieved@K}|}{|\text{Relevant}|}$$

**Example**:
```
Query: "Best price for Sony WH-1000XM5"
Relevant documents: [Doc A, Doc B, Doc C] (3 total)
Retrieved top-5: [Doc A, Doc X, Doc B, Doc Y, Doc Z]

Recall@5 = 2/3 = 0.67
```

**Interpretation**: We found 2 out of 3 relevant documents in the top 5.

#### Precision@K

**Question**: Of the documents we retrieved, how many are relevant?

$$\text{Precision@K} = \frac{|\text{Relevant} \cap \text{Retrieved@K}|}{K}$$

**Example** (same as above):
```
Precision@5 = 2/5 = 0.40
```

#### Mean Reciprocal Rank (MRR)

**Question**: How high is the first relevant document?

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Where $\text{rank}_i$ is the position of the first relevant document for query $i$.

**Example**:
```
Query 1: First relevant at position 1 → 1/1 = 1.0
Query 2: First relevant at position 3 → 1/3 = 0.33
Query 3: First relevant at position 2 → 1/2 = 0.5

MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61
```

#### Normalized Discounted Cumulative Gain (NDCG)

The most sophisticated metric. Accounts for:
- Position (higher is better)
- Graded relevance (not just binary)

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

Where IDCG is the DCG of the ideal (perfect) ranking.

**Implementation**:

```python
import numpy as np

def dcg_at_k(relevances: list, k: int) -> float:
    """
    Discounted Cumulative Gain at K.
    
    relevances: list of relevance scores (e.g., [3, 2, 0, 1, 0])
    """
    relevances = np.asarray(relevances)[:k]
    n = len(relevances)
    
    if n == 0:
        return 0.0
    
    # Discounts: log2(2), log2(3), log2(4), ...
    discounts = np.log2(np.arange(2, n + 2))
    
    # DCG formula
    return np.sum((2 ** relevances - 1) / discounts)

def ndcg_at_k(relevances: list, k: int) -> float:
    """
    Normalized DCG at K.
    """
    dcg = dcg_at_k(relevances, k)
    
    # Ideal DCG: sort relevances descending
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

# Example
relevances = [3, 2, 0, 1, 0]  # Position 1 has score 3, position 2 has score 2, etc.
print(f"NDCG@5: {ndcg_at_k(relevances, 5):.4f}")
# Output: NDCG@5: 0.9608 (good ranking)
```

### DoorDash's Whole-Page Relevance (WPR)

From DoorDash's AutoEval case study:

> "Unlike NDCG, which evaluates a vertical list, WPR measures multiple content blocks arranged spatially on the screen... each content type is weighted by its visual prominence and expected user impact."

This is critical for idealo: search results aren't just a list — they have product cards, price comparisons, merchant info, all with different visual weights.

**Concept**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WHOLE-PAGE RELEVANCE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  HERO PRODUCT                                           │ w=3.0  │
│  │  iPhone 15 - €749 at Amazon                             │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                     │
│  ┌──────────────────────┐  ┌──────────────────────┐                 │
│  │  Alternative 1       │  │  Alternative 2       │  w=1.5          │
│  │  €759 at MediaMarkt  │  │  €769 at Saturn      │                 │
│  └──────────────────────┘  └──────────────────────┘                 │
│                                                                     │
│  ┌──────────────────────┐  ┌──────────────────────┐                 │
│  │  Related Product 1   │  │  Related Product 2   │  w=0.5          │
│  │  iPhone 15 Case      │  │  Screen Protector    │                 │
│  └──────────────────────┘  └──────────────────────┘                 │
│                                                                     │
│  WPR = Σ (relevance_i × weight_i) / Σ weight_i                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Embedding-Based Retrieval Quality

From Airbnb and Vinted:

**Embedding Similarity Distribution**:
```python
def analyze_retrieval_quality(queries, retrieved_docs, embedding_model):
    """
    Analyze the similarity distribution between queries and retrieved docs.
    """
    similarities = []
    
    for query, docs in zip(queries, retrieved_docs):
        query_emb = embedding_model.encode(query)
        
        for doc in docs:
            doc_emb = embedding_model.encode(doc)
            sim = cosine_similarity(query_emb, doc_emb)
            similarities.append(sim)
    
    return {
        'mean_similarity': np.mean(similarities),
        'p50_similarity': np.percentile(similarities, 50),
        'p10_similarity': np.percentile(similarities, 10),  # Worst 10%
        'p90_similarity': np.percentile(similarities, 90)   # Best 10%
    }
```

**Key Insight from Vinted**:
> "Dense retrieval improved recall significantly but sometimes retrieved semantically similar but contextually wrong results."

Example: Query "blue dress" might retrieve "blue shirt" (semantically similar) instead of the actual blue dress in the catalog.

---

## 4. Generation Evaluation

Once you have the right documents, does the LLM use them correctly?

### Faithfulness / Groundedness

**Question**: Is the generated answer supported by the retrieved documents?

**Approach 1: NLI-Based**

Use Natural Language Inference to check if source ENTAILS response.

```python
from transformers import pipeline

def check_faithfulness(source: str, response: str) -> dict:
    """
    Check if response is entailed by source using NLI model.
    """
    nli = pipeline("text-classification", 
                   model="facebook/bart-large-mnli")
    
    result = nli(f"{source} </s></s> {response}")
    
    return {
        'label': result[0]['label'],  # entailment, neutral, contradiction
        'score': result[0]['score']
    }
```

**Approach 2: Claim Extraction + Verification**

```python
def extract_and_verify_claims(response: str, sources: list) -> dict:
    """
    Extract claims from response and verify each against sources.
    """
    # Use LLM to extract claims
    extraction_prompt = f"""
    Extract all factual claims from this text:
    {response}
    
    Output each claim on a new line.
    """
    claims = call_llm(extraction_prompt).split('\n')
    
    # Verify each claim
    verified_claims = []
    hallucinated_claims = []
    
    for claim in claims:
        is_supported = verify_claim_against_sources(claim, sources)
        if is_supported:
            verified_claims.append(claim)
        else:
            hallucinated_claims.append(claim)
    
    return {
        'total_claims': len(claims),
        'verified_claims': len(verified_claims),
        'hallucinated_claims': len(hallucinated_claims),
        'faithfulness_score': len(verified_claims) / len(claims) if claims else 1.0,
        'hallucinations': hallucinated_claims
    }
```

### Answer Relevance

**Question**: Does the answer address the user's query?

**Embedding-Based Relevance**:

$$\text{Relevance} = \cos(\mathbf{e}_q, \mathbf{e}_a)$$

Where $\mathbf{e}_q$ is the query embedding and $\mathbf{e}_a$ is the answer embedding.

**LLM-as-Judge for Relevance**:

```python
def judge_relevance(query: str, answer: str) -> dict:
    prompt = f"""
    Evaluate how well this answer addresses the user's query.
    
    Query: {query}
    Answer: {answer}
    
    Score from 1-5:
    1 = Completely irrelevant
    2 = Mentions the topic but doesn't answer
    3 = Partially answers
    4 = Mostly answers with minor gaps
    5 = Fully addresses the query
    
    Score: 
    Reasoning:
    """
    return call_llm(prompt)
```

### Answer Completeness

From Instacart's LACE framework:

**Dimensions of Completeness**:

| Dimension | Question | Example Failure |
|-----------|----------|-----------------|
| **Factual** | Are all required facts included? | Missing the price in a price query |
| **Actionable** | Can user take action based on answer? | Says "check the website" instead of giving direct link |
| **Contextual** | Does it consider conversation history? | Ignores user's previous preferences |

### Conciseness vs Completeness Trade-off

Instacart's insight:

> "What feels concise to one person might seem overly brief to another. We also observed that LLMs often apply stricter standards for brevity than humans do."

**Metric: Information Density**

$$\text{Density} = \frac{\text{Unique Information Units}}{\text{Word Count}}$$

Higher density = more information per word = better.

---

## 5. End-to-End Evaluation

### The RAGAS Framework

RAGAS (Retrieval-Augmented Generation Assessment) provides a comprehensive evaluation:

| Metric | What It Measures |
|--------|------------------|
| **Context Precision** | Are retrieved docs relevant? |
| **Context Recall** | Did we retrieve all needed docs? |
| **Faithfulness** | Is answer grounded in context? |
| **Answer Relevancy** | Does answer address the query? |
| **Answer Correctness** | Is the answer factually correct? |

**Implementation**:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

def evaluate_rag_system(queries, answers, contexts, ground_truths):
    """
    Comprehensive RAG evaluation using RAGAS.
    """
    from datasets import Dataset
    
    data = {
        'question': queries,
        'answer': answers,
        'contexts': contexts,  # List of lists
        'ground_truth': ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )
    
    return results
```

### Human Evaluation Integration

From Harvey (legal AI):

> "Scaling AI Evaluation Through Expertise... we use domain experts to create golden datasets and validate model outputs."

**Three-Tier Evaluation**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION TIERS                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TIER 1: AUTOMATED (100% of traffic)                                │
│  ├── Embedding similarity                                           │
│  ├── Retrieval metrics (NDCG, MRR)                                  │
│  ├── NLI-based faithfulness                                         │
│  └── Latency & cost tracking                                        │
│                                                                     │
│  TIER 2: LLM-AS-JUDGE (10% of traffic, sampled)                     │
│  ├── Multi-criteria evaluation                                      │
│  ├── Pairwise comparisons                                           │
│  └── Hallucination detection                                        │
│                                                                     │
│  TIER 3: HUMAN REVIEW (1% of traffic, edge cases)                   │
│  ├── Domain expert validation                                       │
│  ├── Golden dataset creation                                        │
│  └── Calibration of automated metrics                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Task Success Rate

The ultimate metric: did the user accomplish their goal?

**For idealo**:
- **Search task**: Did user find a product to click on?
- **Comparison task**: Did user compare prices across merchants?
- **Purchase task**: Did user click through to make a purchase?

```python
def calculate_task_success_rate(sessions: list) -> dict:
    """
    Calculate task success based on session outcomes.
    """
    success_count = 0
    total_sessions = len(sessions)
    
    for session in sessions:
        # Define success criteria
        if session.get('merchant_click'):
            success_count += 1
    
    return {
        'task_success_rate': success_count / total_sessions,
        'total_sessions': total_sessions,
        'successful_sessions': success_count
    }
```

---

## 6. Cost-Quality Trade-offs

### The RAG Cost Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG COST BREAKDOWN                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. EMBEDDING GENERATION                                            │
│     └── $0.0001 per 1K tokens (OpenAI ada-002)                      │
│                                                                     │
│  2. VECTOR DATABASE                                                 │
│     └── ~$0.05 per 1M vectors/month (Pinecone)                      │
│                                                                     │
│  3. RE-RANKING (optional)                                           │
│     └── ~$0.001 per query (Cohere rerank)                           │
│                                                                     │
│  4. LLM GENERATION (dominant cost)                                  │
│     ├── GPT-4: $0.03 input / $0.06 output per 1K tokens             │
│     ├── Claude 3.5: $0.003 input / $0.015 output per 1K             │
│     └── GPT-3.5: $0.0005 input / $0.0015 output per 1K              │
│                                                                     │
│  Total per query (typical RAG):                                     │
│     GPT-4: ~$0.05 - $0.10                                           │
│     Claude 3.5: ~$0.01 - $0.02                                      │
│     GPT-3.5: ~$0.002 - $0.005                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Quality vs Cost Pareto Analysis

From GitHub:

> "We measure token usage. This is one of our main model performance measures. Typically, the fewer tokens a model takes to achieve a result, the more efficient it is."

**Methodology**:

```python
def pareto_analysis(model_results: list) -> list:
    """
    Find Pareto-optimal models (best quality for given cost).
    
    model_results: list of dicts with 'model', 'quality', 'cost_per_query'
    """
    pareto_optimal = []
    
    # Sort by cost ascending
    sorted_results = sorted(model_results, key=lambda x: x['cost_per_query'])
    
    best_quality_so_far = -float('inf')
    
    for result in sorted_results:
        if result['quality'] > best_quality_so_far:
            pareto_optimal.append(result)
            best_quality_so_far = result['quality']
    
    return pareto_optimal
```

**Example Output**:

| Model | Quality (1-5) | Cost/Query | Pareto-Optimal? |
|-------|---------------|------------|-----------------|
| GPT-3.5 | 3.2 | $0.003 | ✓ |
| Claude 3 Haiku | 3.8 | $0.008 | ✓ |
| GPT-4 | 4.5 | $0.060 | ✓ |
| Claude 3.5 | 4.4 | $0.015 | ✓ (best value) |
| GPT-4-turbo | 4.3 | $0.040 | ✗ (dominated) |

### Context Window Optimization

More context = better answers but higher cost.

**Strategy: Adaptive Context**

```python
def adaptive_context_retrieval(query: str, max_tokens: int = 4000) -> list:
    """
    Retrieve documents adaptively based on query complexity.
    """
    # Estimate query complexity
    complexity = estimate_query_complexity(query)
    
    if complexity == 'simple':
        # Single-fact queries need fewer documents
        num_docs = 2
        tokens_per_doc = 500
    elif complexity == 'moderate':
        num_docs = 4
        tokens_per_doc = 750
    else:
        # Complex queries need more context
        num_docs = 6
        tokens_per_doc = 1000
    
    # Retrieve and truncate
    docs = retrieve_documents(query, k=num_docs)
    truncated_docs = [truncate(doc, tokens_per_doc) for doc in docs]
    
    return truncated_docs
```

---

## 7. Production Monitoring

### Key Metrics to Track

From DoorDash's AutoEval:

> "WPR supports full-stack search evaluation across all stages: Retrieval, Ranking, Post-processing, and User experience composition."

**Dashboard Metrics**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG MONITORING DASHBOARD                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RETRIEVAL HEALTH                                                   │
│  ├── Recall@10: 0.85 (target: >0.80) ✓                              │
│  ├── MRR: 0.72 (target: >0.65) ✓                                    │
│  ├── Empty result rate: 2.3% (target: <5%) ✓                        │
│  └── Avg retrieval latency: 45ms (target: <100ms) ✓                 │
│                                                                     │
│  GENERATION HEALTH                                                  │
│  ├── Faithfulness score: 0.91 (target: >0.90) ✓                     │
│  ├── Hallucination rate: 3.2% (target: <5%) ✓                       │
│  ├── Answer relevance: 4.1/5 (target: >4.0) ✓                       │
│  └── Avg generation latency: 850ms (target: <1000ms) ✓              │
│                                                                     │
│  BUSINESS METRICS                                                   │
│  ├── Task success rate: 78% (target: >75%) ✓                        │
│  ├── User satisfaction (thumbs up): 82% (target: >80%) ✓            │
│  └── Cost per query: $0.018 (budget: <$0.025) ✓                     │
│                                                                     │
│  ALERTS                                                             │
│  └── No active alerts                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Drift Detection for RAG

**Query Distribution Drift**:
- Are users asking different types of questions?
- New products/categories not in knowledge base?

**Document Freshness**:
- Are embeddings stale?
- When was the knowledge base last updated?

```python
def monitor_rag_drift(current_queries: list, baseline_queries: list):
    """
    Detect drift in query distribution.
    """
    from scipy.stats import ks_2samp
    
    # Embed queries
    current_embs = embed_queries(current_queries)
    baseline_embs = embed_queries(baseline_queries)
    
    # Compare distributions (simplified: first principal component)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    
    current_pc = pca.fit_transform(current_embs).flatten()
    baseline_pc = pca.transform(baseline_embs).flatten()
    
    # KS test for distribution difference
    ks_stat, p_value = ks_2samp(current_pc, baseline_pc)
    
    return {
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'drift_detected': p_value < 0.01
    }
```

### Continuous Evaluation Pipeline

From Instacart:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS EVALUATION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Daily:                                                             │
│  ├── Sample 1000 random queries from production                     │
│  ├── Run through evaluation pipeline                                │
│  ├── Compare to baseline metrics                                    │
│  └── Alert if degradation detected                                  │
│                                                                     │
│  Weekly:                                                            │
│  ├── Full evaluation on golden dataset                              │
│  ├── Human review of flagged cases                                  │
│  └── Update baseline if system improved                             │
│                                                                     │
│  Monthly:                                                           │
│  ├── Re-evaluate golden dataset with human raters                   │
│  ├── Calibrate LLM-as-Judge against humans                          │
│  └── Update evaluation criteria if needed                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Case Study: Harvey's Enterprise-Grade RAG

### Context

Harvey is a legal AI company that built an enterprise RAG system for law firms. Their blog post "Enterprise-Grade RAG Systems" provides valuable insights.

### Key Lessons

**1. Domain-Specific Chunking**

Generic chunking (e.g., 512 tokens) doesn't work for legal documents:
- Contracts have clauses that must stay together
- Case law has citations that reference each other

**idealo Application**: Product descriptions have structured fields (price, specs, reviews) that should be chunked appropriately.

**2. Hybrid Retrieval**

Pure semantic search misses exact matches:
- "Section 4.2.1" needs exact match, not semantic similarity
- Product SKUs, model numbers need keyword matching

```python
def hybrid_retrieval(query: str, k: int = 5) -> list:
    """
    Combine semantic and keyword retrieval.
    """
    # Semantic retrieval
    semantic_results = semantic_search(query, k=k*2)
    
    # Keyword retrieval (BM25)
    keyword_results = keyword_search(query, k=k*2)
    
    # Reciprocal Rank Fusion
    combined = reciprocal_rank_fusion(semantic_results, keyword_results)
    
    return combined[:k]
```

**3. Evaluation at Scale**

From Harvey:
> "We use domain experts to create golden datasets and validate model outputs."

Their process:
1. Lawyers create test questions with known answers
2. System generates answers
3. Lawyers grade answers on accuracy, completeness, citation quality
4. Iterate on retrieval and generation

### Applying Harvey's Lessons to idealo

| Harvey Challenge | idealo Equivalent | Solution |
|------------------|-------------------|----------|
| Legal clauses must stay together | Product specs are structured | Structure-aware chunking |
| Section references need exact match | Product SKUs need exact match | Hybrid retrieval |
| Domain expertise for evaluation | Product/pricing expertise | Partner with product team |
| Citation accuracy | Source attribution | Track which merchant data generated answer |

---

## Summary: Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG EVALUATION CHECKLIST                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RETRIEVAL:                                                         │
│  ☐ Measure Recall@K, MRR, NDCG                                      │
│  ☐ Track empty result rate                                          │
│  ☐ Monitor retrieval latency                                        │
│  ☐ Consider WPR for complex layouts                                 │
│                                                                     │
│  GENERATION:                                                        │
│  ☐ Evaluate faithfulness (grounded in sources)                      │
│  ☐ Measure hallucination rate                                       │
│  ☐ Check answer relevance to query                                  │
│  ☐ Assess completeness and conciseness                              │
│                                                                     │
│  END-TO-END:                                                        │
│  ☐ Task success rate                                                │
│  ☐ User satisfaction (thumbs up/down)                               │
│  ☐ A/B test against baseline                                        │
│                                                                     │
│  COST:                                                              │
│  ☐ Track cost per query                                             │
│  ☐ Pareto analysis for model selection                              │
│  ☐ Optimize context window usage                                    │
│                                                                     │
│  MONITORING:                                                        │
│  ☐ Daily automated evaluation                                       │
│  ☐ Drift detection for queries and documents                        │
│  ☐ Regular human calibration                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Next: [04_CASE_STUDIES.md](04_CASE_STUDIES.md)*
