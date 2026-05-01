## Chapter 2 — System design scenarios

!!! tip "Focus for this chapter"
    Practice **back-of-envelope budgets**: milliseconds per stage, cost per query, and what you **remove from the hot path** (cross-encoder? sync LLM?). Tie every knob to an SLA or dollar figure.

### Scenario 4 — Design RAG for 10M daily users *(Big Tech / FAANG — Senior)*

**Situation**

Imagine you own **in-product help or knowledge search** for a consumer super-app: tens of millions of **daily active users**, tens of millions of **documents** (help articles, policies, community posts, structured FAQs). Users expect answers **as fast as a web search**—product leadership has advertised a **~100 ms P99 end-to-end SLA** for the *full* experience (not just vector search).

Documents change often (campaigns, regional rules, feature launches), so leadership wants **near–real-time freshness**—old stale answers are a support-ticket driver. At the same time, finance caps incremental infra + LLM spend at roughly **$50K/month** for this surface.

This scenario forces you to **negotiate reality**: pure “retrieve 50 chunks + GPT-4 full context” cannot meet **100 ms P99**. You must explain **what ships synchronously**, what is **cached**, what is **async/streamed**, how **ingestion** scales, and how you **prove** you hit latency and cost targets.

**Constraints:** 10M DAU, **100 ms P99** latency SLA, **50M documents**, near-real-time updates, **$50K/month** budget.

**Interviewer:** Where do you start?

**You:** **Latency first** — 100 ms P99 is extremely tight for full RAG; rules out sync cross-encoder on hot path; demands aggressive caching and slim generation.

!!! note "Wall-clock SLA vs TTFT (tie-in)"
    A consumer SLA saying “100 ms” rarely means “full answer rendered.” In practice you often **negotiate phases**: **TTFT** after minimal retrieval + mini-model stub or streamed prefix, while **complete** answer or heavy retrieval finishes later — or you redefine SLA as **cached path only**. Say explicitly whether the SLA is **first byte**, **first model token**, or **full completion**; mixing these invalidates the architecture discussion (same idea as Part I §1).

**Ingestion pipeline (real-time)**

```
Document Update → Kafka (doc_updates) → Flink/Spark Streaming
  → chunking (semantic, 512t — trades embedding specificity vs context size per chunk) → embedding (batched, GPU) → Qdrant upsert (sharded by namespace)
Lag target: < 30 s doc-to-searchable
```

**Serving path — 100 ms budget (example breakdown)**

| Step | Budget | Optimisation |
|------|--------|--------------|
| Query embedding | 5 ms | Cache; local ONNX |
| Cache lookup (Redis) | 3 ms | Semantic cache |
| Vector ANN (Qdrant) | 15 ms | HNSW ef_search=40, sharded |
| Merge + filter | 5 ms | In-process |
| LLM (streaming TTFT) | 60 ms | GPT-4o-mini, 1024 ctx, speculative decode |
| Network | 12 ms | gRPC, SSE |
| **Total P50** | ~70 ms | |
| **P99** | 100 ms | Circuit break → cached fallback |

**Cost — stay under $50K/month**

- Semantic cache ~40% hit → fewer LLM calls.  
- Route simple queries to mini vs complex to flagship.  
- `text-embedding-3-small` vs large where quality delta small.  
- Self-hosted Qdrant on K8s spot vs managed where viable.  
- Batch non-urgent re-embeddings overnight on spot.

---

### Scenario 5 — Multi-tenant RAG with data isolation *(SaaS / B2B — Hard)*

**Situation**

You sell a **B2B SaaS assistant** that indexes each customer’s private documents—contracts, HR policies, runbooks, ticket exports—and answers employee questions. You have **hundreds of enterprises** on shared infrastructure. Most tenants are fine with standard SaaS controls, but a **Fortune 500** prospect’s security review asks explicitly: *“Demonstrate that our embeddings and metadata cannot appear in another tenant’s retrieval results—even under bug conditions.”*

That requirement goes beyond polite promises: it implies thinking about **failure modes** (bad filter, wrong API key, broken namespace), **compliance narratives** (SOC 2, sometimes HIPAA), and **commercial packaging** (shared vs isolated indexes affect **margin**). Sales wants “full isolation,” engineering wants **cost efficiency**, and you must articulate **tiered isolation models** and when each is appropriate.

**Interviewer:** Isolation options and trade-offs?

**You:** Three models on cost/security curve:

| Model | Isolation | Cost | Compliance | Best for |
|-------|-----------|------|------------|----------|
| Namespace + filter | Logical | $ | Low | SMB, <1000 tenants |
| Collection per tenant | Logical + physical index | $$ | Medium | Mid-market |
| Dedicated cluster | Full infra | $$$ | SOC2/HIPAA | Regulated enterprise |

**Hybrid (tiered product)**

- SMB: shared cluster, `tenant_id` filter.  
- Enterprise: collection isolation.  
- Regulated: dedicated Qdrant in tenant VPC via Terraform.

```python
def get_vector_client(tenant_id: str) -> QdrantClient:
    tenant = db.get_tenant(tenant_id)
    if tenant.tier == 'enterprise_isolated':
        return QdrantClient(url=tenant.dedicated_cluster_url,
                            api_key=vault.get(f'qdrant/{tenant_id}'))
    elif tenant.tier == 'business':
        return shared_client  # collection=f'tenant_{tenant_id}'
    else:
        return shared_client  # collection='shared', filter=tenant_id
```

---

