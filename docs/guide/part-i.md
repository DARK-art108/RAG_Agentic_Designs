# Part I — RAG & Agentic AI: Advanced Interview Preparation Guide

**Themes:** Latency · Chunking · Memory · Context Compaction · Agent Harness · Advanced RAG · Agentic AI

!!! abstract "How to read Part I"
    Treat this part as your **technical phrasebook**: each section gives you definitions, trade-offs, and numbers you can say aloud in interviews. You do **not** need to memorize every bullet — you need to know **where bottlenecks hide** (embed → ANN → rerank → LLM), **when** to use hybrid retrieval vs agents, and **which metrics** (RAGAS, latency percentiles) prove improvement.

!!! tip "What interviewers listen for"
    When you answer latency questions, always separate **measured wall-clock** from **user-perceived** latency (streaming / TTFT). When you answer chunking questions, connect **chunk size** to **embedding geometry** and **what the model actually reads**.

---

## 1. RAG latency — from 15s down to <1s

This is the most common senior-level RAG question. Interviewers want to see that you can systematically diagnose the bottleneck rather than blindly throwing hardware at it.

!!! note "Plain-language framing"
    Think of the pipeline as a chain of queues: every stage adds delay **and** shapes what the model believes is true. Fixing latency without touching retrieval quality is useless; fixing retrieval without measuring LLM token cost is incomplete.

### Q: Your RAG pipeline takes 15 seconds to return an answer. Walk me through how you diagnose and fix it.

#### Step 1 — Instrument & identify the bottleneck

Before optimising anything, measure each stage individually. A typical RAG pipeline has these stages:

| Stage | Typical cost | Culprit signs |
|-------|----------------|---------------|
| Query embedding | 20–100 ms | Slow model, no batching |
| Vector DB search (ANN) | 50–500 ms | No HNSW index, large ef_search |
| Document fetch / rerank | 200 ms–3 s | Cross-encoder on 50+ chunks |
| LLM generation | 2–12 s | Large context, slow model, no streaming |
| Network / serialisation | 50–300 ms | Non-local DB, JSON over-fetching |

#### Step 2 — Optimisation playbook

**Embedding cache** — Cache embeddings for frequent queries using Redis or an in-memory LRU. Semantic de-duplication via cosine similarity (threshold ~0.97) lets you skip re-embedding near-identical queries.

```python
redis_client.setex(f"emb:{hash(query)}", 3600, embed(query).tobytes())
```

**Async retrieval + reranking pipeline** — Run embedding, vector search, and BM25 sparse search concurrently with `asyncio.gather`. Apply cross-encoder reranking only on the top-k candidates, not the full corpus.

```python
results = await asyncio.gather(
    vector_store.search(q_emb, k=20),
    bm25_index.search(query, k=20)
)
reranked = cross_encoder.rank(query, merge(results))[:5]
```

**HNSW index tuning** — Tune `ef_construction=200`, `M=16` at index-build time. At query time use `ef_search=50–100`. Use Product Quantisation (PQ) to compress vectors by 8–32× with <5% recall loss.

```python
# Qdrant example
qdrant.create_collection('docs',
    vectors_config=VectorParams(
        size=1536, distance=Distance.COSINE,
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200)
    ))
```

**Streaming LLM output** — Start streaming tokens as soon as the LLM begins generating. User-perceived latency drops to time-to-first-token (TTFT), typically 300–800 ms, even if total generation takes 5 s.

```python
async for chunk in llm.stream(prompt):
    yield chunk  # SSE / WebSocket
```

**Context window reduction** — Send fewer, more relevant chunks to the LLM. A 2000-token context is 3–4× faster to generate over than 8000 tokens. Use a reranker to ruthlessly prune to the top-3 chunks.

```python
# Instead of top-10 chunks (~8k tokens)
# Send top-3 reranked chunks (~2k tokens)
```

**Speculative decoding / smaller LLM** — Use a small draft model (e.g. Llama-3.2-1B) to propose tokens, verified by the large model. Alternatively, route simple queries to a smaller/faster model (GPT-4o-mini, Haiku).

```python
model = route(query)  # "gpt-4o-mini" or "gpt-4o"
response = llm.chat(model=model, messages=msgs)
```

**Key insight:** The single highest-impact fix is almost always: **reduce context sent to the LLM** **and** **enable streaming**. Together they cut perceived latency by 60–80%.

#### Step 3 — Architecture: the fast RAG stack

```
Query → [Embedding cache check] → Async(Vector ANN + BM25) → Reciprocal Rank Fusion →
Cross-encoder rerank top-5 → Context compression (LLMLingua / selective summarisation) → Streaming LLM
```

Quote numbers in the interview: *"We cut P95 latency from 15 s to 1.2 s by streaming + reducing context from 8 k to 2 k tokens + async retrieval."*

---

## 2. Chunking strategies

### Q: What chunking strategy would you use for a 500-page legal document vs a codebase? Why does chunk size matter?

Chunking is the single biggest lever on RAG quality that most engineers underestimate. The goal is to create chunks that are semantically complete, embedding-friendly, and LLM-context-efficient.

| Strategy | Best for | Chunk size | Overlap | Gotcha |
|----------|-----------|------------|---------|--------|
| Fixed-size (chars) | Baseline, logs | 512–1024 chars | 10–20% | Breaks mid-sentence |
| Recursive splitter | General prose | 512 tokens | 64 tokens | Still ignores semantics |
| Semantic splitter | Legal/medical docs | Variable | None needed | Slow; needs embedder |
| Sentence window | Q&A, FAQs | 1 sentence | ±3 sentences | Large index size |
| Document hierarchy | PDFs with sections | Section-aware | None | Requires PDF parsing |
| Code-aware splitter | Codebases | Function/class level | None | Lang-specific parsers |
| Agentic chunking | Complex research docs | Dynamic | LLM decides | Expensive offline |

### The parent–child chunking pattern

Store large **parent** chunks (1024 tokens) for context, but embed small **child** chunks (128 tokens) for precise retrieval. When a child chunk matches, return its parent to the LLM. This solves the precision–recall tension inherent in chunking.

```python
# LlamaIndex ParentChildNodeParser equivalent
small_chunks = split(doc, size=128)
large_chunks = split(doc, size=1024)
for small in small_chunks:
    small.metadata['parent_id'] = find_parent(small, large_chunks)
# At query time:
results = vector_store.search(query_emb, index='small_chunks')
context = [fetch_parent(r.parent_id) for r in results]
```

### Chunk size vs retrieval quality trade-off

- **Small chunks (64–128 tokens):** high precision, low context — great for fact lookup.  
- **Large chunks (512–1024 tokens):** lower precision, richer context — great for reasoning tasks.  

The optimal size depends on your query distribution; run ablation experiments with RAGAS metrics (faithfulness, context relevancy).

**For interviews:** propose A/B testing chunk sizes using RAGAS offline evaluation before shipping.

---

## 3. Advanced RAG techniques

### Q: What are HyDE, FLARE, and Self-RAG? When would you use each?

**HyDE — Hypothetical Document Embedding**

- **When to use:** Queries are short/ambiguous and don't match document vocabulary.  
- **How it works:** Generate a fake “ideal answer” with the LLM, embed it, then search. The hypothesis embedding is closer to real documents than the raw query embedding.

**FLARE — Forward-Looking Active Retrieval**

- **When to use:** Long-form generation where facts may change mid-response.  
- **How it works:** The LLM generates token-by-token; when confidence drops below a threshold, it pauses, retrieves fresh context, and continues (“self-healing” generation).

**Self-RAG — Self-Reflective RAG**

- **When to use:** High-stakes domains (medical, legal) requiring factual grounding.  
- **How it works:** Fine-tune the LLM to emit special tokens: `[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`. The model decides whether to retrieve, judges relevance, and flags unsupported claims.

**RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval**

- **When to use:** Long documents requiring multi-hop reasoning across sections.  
- **How it works:** Cluster and recursively summarise chunks, building a tree. Query can match at any level — leaf for detail, root for overview.

### Q: Explain hybrid search and Reciprocal Rank Fusion (RRF)

Hybrid search combines **dense** vector search (semantic) with **sparse** BM25/TF-IDF (keyword) search. Neither alone is sufficient: dense misses exact keyword matches; sparse misses paraphrase/synonym matches.

```python
def reciprocal_rank_fusion(rankings: list[list[str]], k=60) -> list[str]:
    scores = defaultdict(float)
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list, 1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

`k=60` is the standard RRF constant that prevents high-rank docs from dominating. RRF outperforms simple score normalisation because it's **rank-based**, not score-based — scale differences between BM25 and cosine similarity don't matter.

---

## 4. Agentic AI architecture

### Q: What is the difference between a RAG pipeline and an Agentic RAG system?

| Dimension | Naive RAG | Agentic RAG |
|-----------|-----------|-------------|
| Decision-making | Static pipeline | LLM decides next action |
| Retrieval | Single-shot | Iterative, multi-hop |
| Tools | None | Search, code exec, APIs |
| Error handling | None | Self-correction loops |
| Latency | 1–3 s | 5–30 s (more powerful) |
| Use cases | FAQ, search | Research, coding, analysis |

### Q: Describe the ReAct pattern. What are its failure modes?

**ReAct** (Reasoning + Acting) interleaves chain-of-thought reasoning with tool actions in a loop: **Thought → Action → Observation → Thought → … → Final Answer.**

Example trace:

- **Thought:** I need to find the Q3 revenue for Acme Corp.  
- **Action:** `search('Acme Corp Q3 2024 revenue')`  
- **Observation:** Acme Corp reported $2.1B revenue in Q3 2024 …  
- **Thought:** Now I can answer the user.  
- **Final Answer:** Acme Corp's Q3 2024 revenue was $2.1B.

**ReAct failure modes**

- **Hallucinated tool calls** — agent invents tool names or arguments that don't exist.  
- **Infinite loops** — agent keeps searching without converging; add a **max-steps** budget.  
- **Context overflow** — long ReAct traces fill the context window; use **context compaction**.  
- **Action grounding** — destructive actions (DELETE, SEND EMAIL) without confirmation.  
- **Reward hacking** — shortcuts that satisfy the task spec but not user intent.  

**In interviews:** always mention **guardrails** — `max_steps`, tool-call validation schema (Pydantic), and **human-in-the-loop** checkpoints for high-risk actions.

---

## 5. Agent harness design

### Q: Design an agent harness that supports multiple LLM backends, tool registries, and safe execution sandboxing

An **agent harness** is the infrastructure layer that wraps an LLM agent loop and provides: tool registration, execution, sandboxing, observability, retry logic, and state persistence.

**Core components**

| Component | Role |
|-----------|------|
| **Orchestrator** | Runs the Thought→Action→Observation loop; abstracts LLM backend (OpenAI, Anthropic, local). |
| **Tool registry** | Typed tool definitions (JSON Schema / Pydantic); validates inputs **before** execution; supports hot-reload. |
| **Execution sandbox** | E2B or Docker for code execution; network/filesystem isolation; CPU/memory limits. |
| **Memory store** | Short-term: conversation buffer; long-term: vector DB; episodic: structured event log. |
| **Checkpointing** | Persist agent state after each action; resume from checkpoint on failure or timeout. |
| **Observability** | OpenTelemetry traces per step; log input/output tokens, tool latency, cost per run. |
| **Safety layer** | PII redaction, prompt injection detection, action confirmation for destructive ops. |

**Harness pseudo-code**

```python
class AgentHarness:
    def run(self, task: str, max_steps=15):
        state = self.memory.load() or AgentState(task=task)
        for step in range(max_steps):
            action = self.llm.think(state)  # Thought + Action
            if action.type == 'final_answer':
                return action.value
            validated = self.tool_registry.validate(action)  # Pydantic
            with self.sandbox.timeout(30):
                obs = self.tool_registry.execute(validated)
            state.add(action, obs)
            self.memory.checkpoint(state)
            self.tracer.log(step, action, obs)
        raise MaxStepsExceeded(state)
```

### Q: How do you handle prompt injection in an agentic system?

- **Separate data from instructions** — use structured tool output schemas; never concatenate raw web content into the system prompt.  
- **Input sanitisation** — strip `[INST]` tokens and jailbreak patterns from tool observations.  
- **Privilege separation** — agent has a read-only context window; only the harness calls tools.  
- **Output validation** — before executing a tool call, validate against the registered schema and reject anomalous arguments.  
- **Canary tokens** — invisible watermarks in sensitive documents; alert if the agent tries to exfiltrate them.

---

## 6. Memory systems in LLM agents

### Q: Explain the four types of memory in agentic AI. How do you decide which to use?

| Type | Analogy | Storage | Retrieval | TTL | Best for |
|------|---------|---------|-----------|-----|----------|
| In-context (short-term) | Working memory | Token window | Direct | Session | Active conversation state |
| External (long-term) | Hard disk | Vector DB / SQL | Semantic / ID | Persistent | Facts, user prefs, history |
| Episodic | Diary | Structured log | Temporal query | Persistent | Past agent runs, decisions |
| Procedural (skills) | Muscle memory | Prompt library / fine-tune | Retrieval / learned | Persistent | Tool usage, domain SOPs |

### Q: How do you implement memory compression so a long-running agent doesn't overflow the context window?

Long-running agents accumulate conversation history that eventually exceeds the context window. Three strategies:

1. **Sliding window** — keep only the last N turns. Simple but loses early context; good for conversational agents.  
2. **Summarisation compression** — when history > 80% of context limit, run a cheap LLM to summarise older turns into a ~200-token digest; prepend digest to new context.  
3. **Memory retrieval (MemGPT pattern)** — move old messages to external memory (vector DB); at each step retrieve top-k relevant memories and inject them; full recall, higher latency.

```python
# Summarisation-based compression
def compress_history(messages, token_limit=6000):
    if count_tokens(messages) < token_limit * 0.8:
        return messages
    old = messages[:-6]  # keep last 3 turns verbatim
    summary = llm.summarise(old, max_tokens=200)
    return [{'role': 'system', 'content': f'Past context: {summary}'}] + messages[-6:]
```

---

## 7. Context compaction

### Q: What is context compaction and how does it differ from simple summarisation? Implement a production-grade solution.

**Context compaction** reduces tokens in the active context window while preserving information needed for the **current task**. Unlike plain summarisation, compaction is **task-aware** — it keeps task-relevant detail and drops irrelevant noise.

**Compaction techniques**

| Technique | Notes |
|-----------|--------|
| **LLMLingua / LLMLingua-2** | Token-level compression via small LM perplexity; drops low-information tokens; ~3–5× compression, often <5% accuracy drop. |
| **Selective context** | Remove sentences with low self-information (high n-gram overlap); fast, no LLM call. |
| **Hierarchical summarisation** | Summarise blocks, then summarise summaries; preserves local coherence. |
| **Key-value cache reuse (prompt caching)** | Stable system prompt + document prefix cached at provider; only new turns sent. |
| **Structured extraction** | Replace raw conversation with JSON/dataclass facts; deterministic for well-defined domains. |

**When to trigger compaction**

```python
COMPACTION_THRESHOLD = 0.75  # trigger at 75% of context limit

def should_compact(messages, model_limit=128_000):
    used = sum(count_tokens(m['content']) for m in messages)
    return used / model_limit > COMPACTION_THRESHOLD

def compact(messages, task_description):
    # Keep system prompt + last 2 turns verbatim
    pinned = [messages[0]] + messages[-4:]
    to_compress = messages[1:-4]
    digest = llm.call(
        f'Compress this conversation. Preserve details relevant to: {task_description}. '
        f'Output as dense bullet points. Max 300 tokens.\n\n{format(to_compress)}'
    )
    return [messages[0],
            {'role': 'system', 'content': f'[COMPACTED HISTORY]\n{digest}'},
            *messages[-4:]]
```

**Rule:** Never compact the **system prompt** or the **last 2 user turns** — they anchor current task and identity.

---

## 8. Multi-agent systems & evaluation

### Q: Design a multi-agent system for automated code review. What are the coordination patterns?

Multi-agent systems decompose complex tasks across specialised agents. Three coordination patterns:

1. **Orchestrator–worker (hierarchical)** — planner breaks task into subtasks; workers report back; orchestrator synthesises. **Best for:** research, code review, report generation.  
2. **Peer-to-peer (debate / society of mind)** — independent outputs; judge/aggregator resolves conflicts; can improve factual accuracy ~15–30% on some reasoning tasks. **Best for:** adversarial verification, factual Q&A, essay grading.  
3. **Pipeline (assembly line)** — Agent A → Agent B in fixed sequence. **Best for:** extract → transform → generate → review.

### Q: How do you evaluate a RAG pipeline? What metrics matter most?

| Metric | What it measures | Tool | Target |
|--------|------------------|------|--------|
| Faithfulness | Answer grounded in context? No hallucinations? | RAGAS | > 0.9 |
| Answer relevancy | Does answer address the question? | RAGAS | > 0.85 |
| Context precision | Fraction of retrieved chunks relevant | RAGAS | > 0.8 |
| Context recall | Fraction of relevant info retrieved | RAGAS | > 0.75 |
| MRR / NDCG | Ranking quality | Custom | MRR > 0.7 |
| Latency P95 | End-to-end response time | Prometheus | < 2 s |
| Cost per query | LLM + embedding + DB | Custom | < $0.01 |

**RAGAS** is the standard framework. Mention building an **offline golden dataset** (100–500 Q&A pairs) and **regression tests** on every pipeline change.

---

## 9. Rapid-fire interview questions

**Q1: What is the curse of dimensionality in vector search?**  
→ As dimensions grow, distances saturate; nearest-neighbour becomes less meaningful. Mitigation: PCA/UMAP reduction, PQ compression, or **matryoshka** embeddings.

**Q2: Explain HNSW vs IVF-Flat vs FAISS.**  
→ **IVF-Flat:** partition corpus into Voronoi cells, probe top-N cells — fast but fixed at index time. **HNSW:** hierarchical graph, dynamic, excellent recall, high memory. **FAISS IVF-PQ:** partitioning + compression — high throughput at billions of vectors.

**Q3: What is lost-in-the-middle?**  
→ LLMs under-attend to content in the **middle** of long contexts. Fix: place most relevant chunks at **start and end** of the packed context.

**Q4: What is a router in RAG?**  
→ Classifier routing queries to vector DB, SQL, API, or direct LLM answer (LLM tool-call or lightweight classifier like BERT).

**Q5: How do you handle multi-modal RAG?**  
→ Embed images with CLIP; retrieve across modalities; reason with GPT-4o / Claude 3 over mixed text+image context.

**Q6: What is speculative RAG?**  
→ Small draft LLM proposes answer; verifier checks against retrieved docs; accept/reject/edit; reduces verifier compute ~3×.

**Q7: How do you prevent RAG hallucinations?**  
→ (1) Faithfulness guardrail: NLI vs context. (2) Citation forcing with source IDs. (3) Self-consistency / voting across samples.

**Q8: What is knowledge-graph RAG?**  
→ Extract entities/relations to Neo4j; traverse for multi-hop; linearise subgraph for LLM — strong for relational questions.

---

## 10. Quick reference cheat sheet

| Topic | Key numbers / facts |
|-------|---------------------|
| Chunk size (general) | 512 tokens; overlap 10% |
| Parent–child chunks | Child: 128 t / Parent: 1024 t |
| HNSW params | M=16, ef_construct=200, ef_search=50–100 |
| RRF constant | k=60 |
| Compaction trigger | 75–80% of context limit |
| RAGAS faithfulness target | > 0.90 |
| Latency budget (P95) | < 2 s with streaming |
| ReAct max steps | 10–20 (add budget_forced_halt) |
| Embedding models | text-embedding-3-small (1536d), BGE-M3, E5-Large |
| Vector DBs | Qdrant (self-host), Pinecone (managed), Weaviate, pgvector |
| Reranker models | Cohere Rerank, cross-encoder/ms-marco, bge-reranker-v2 |
| Eval frameworks | RAGAS, TruLens, DeepEval, PromptFlow |
| Agent frameworks | LangGraph, CrewAI, AutoGen, LlamaIndex Workflows |

*Generated for Advanced ML / LLM Engineer Interview Prep · April 2026*

