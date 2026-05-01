## Chapter 1 — Production incident scenarios

!!! tip "Focus for this chapter"
    These scenarios reward **data-plane discipline**: versioning, metadata filters, dual-write migrations, and **numeric grounding**. Explain **why** the failure happened (missing invariant), not only which tool you would install next.

### Scenario 1 — RAG returns stale legal clauses *(LegalTech / Fintech — Hard)*

**Situation**

You are a senior engineer at a **LegalTech** company. The product is an internal-facing assistant that answers questions about **contracts, policies, and regulations** (GDPR, DPAs, vendor terms). Lawyers and customer-success teams paste questions in natural language; the app retrieves chunks from a **vector database** and an LLM drafts an answer with citations.

The retrieval corpus was built months earlier by batch-ingesting PDFs and wiki pages. When regulations or templates change, **someone occasionally uploads a new PDF**, but there was **no formal “retirement” process** for older material—new chunks were added next to old ones. Embeddings for both versions can look similar (same topic, overlapping wording), so the model sometimes pulls **the outdated clause** because it still matches the query semantically.

A large enterprise customer **audits** an answer and discovers the bot cited a **GDPR article that was superseded eight months ago**. For them this is not a small UX glitch—it touches **compliance, audit trails, and liability**. The issue is escalated to your **CEO**, and you are asked *today* what broke and how you stop it from happening again.

**Interviewer:** What went wrong architecturally, and what's your immediate triage plan?

**You:** Three failures: **no document versioning** in the vector store, **no freshness metadata** on chunks, **no retrieval filter** excluding deprecated content. **Immediate triage:** add `valid_until` and `version` metadata at ingest; **hard filter** on every query — only chunks where validity holds. Interim: manual audit and **tombstone** outdated chunks.

**Root cause analysis**

- Ingestion had no document lifecycle — updates created new chunks without retiring old ones.  
- Metadata lacked `version`, `source_date`, `superseded_by`.  
- No retrieval-time freshness enforcement.  
- No automated re-ingestion on source updates (webhook/polling gap).

**Fix — document lifecycle architecture**

```python
# Chunk metadata schema (enforced at ingest)
class ChunkMetadata(BaseModel):
    doc_id: str
    version: int
    effective_date: date
    superseded_date: Optional[date] = None  # set when doc is updated
    jurisdiction: str
    doc_type: str  # 'regulation', 'contract', 'policy'

# Query filter — ALWAYS apply this
filter = Filter(
    must=[
        FieldCondition(key='superseded_date', match=MatchValue(value=None)),
        FieldCondition(key='effective_date', range=DateRange(lte=date.today())),
    ]
)
results = qdrant.search(collection, query_emb, query_filter=filter)
```

**Preventing recurrence**

```python
# GitHub webhook → ingest pipeline trigger
@app.post('/webhook/doc-updated')
async def handle_update(payload: dict):
    doc_id = payload['doc_id']
    # Retire all old chunks
    qdrant.set_payload(collection, {'superseded_date': str(date.today())},
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key='doc_id',
                match=MatchValue(value=doc_id))])
        ))
    # Ingest new version
    await ingest_document(payload['new_content'], doc_id, version=payload['version'])
```

**Follow-up — How do you test this doesn't regress?**  
Golden suite (~50 questions) where correct answers **changed across versions**; CI runs RAGAS faithfulness on every deploy; daily cron compares **source doc hashes** to ingested chunk hashes — mismatch → Slack alert and deploy block.

**Key insight:** Legal/medical RAG must treat documents as **versioned entities**, not static blobs.

---

### Scenario 2 — Embedding model migration at scale *(E-commerce / FAANG — Hard)*

**Situation**

You support **search and recommendations** for a large e-commerce catalog: titles, descriptions, attributes, and reviews are embedded into vectors and stored in **Pinecone** (or equivalent). The live index holds about **50 million vectors**, built with OpenAI’s legacy **`text-embedding-ada-002`** model.

Product and ML leadership ran offline benchmarks and found that **`text-embedding-3-large`** (or another newer embedding) improves **recall@k** and semantic relevance by a meaningful margin—roughly **~20%** on internal golden queries. The VP wants the better model **in production**, but the business cannot afford **hours of broken search**, **empty results**, or **wrong rankings** during a migration.

So you face a classic constraint: **all IDs must keep resolving**, **traffic must stay up**, new products must remain embeddable **during** the migration, and you need a **safe rollback** if metrics regress after cutover. The interviewer is testing whether you understand **blue–green indexes**, **dual-write**, **validation gates**, and **feature-flagged cutover**—not “re-embed everything in place overnight.”

**Interviewer:** How do you migrate 50M vectors with zero downtime?

**You:** **Blue–green embedding deployment.** Keep `index-v1` live; build `index-v2` in parallel; background re-embed (~6–8 h at batch 2048, 3 workers). Reads stay on v1 until v2 validated; **atomic pointer swap** via feature flag; v1 remains rollback target ~24 h.

**Migration architecture**

```python
# Phase 1: Background re-embedding (no traffic impact)
async def migrate_batch(product_ids: list[str]):
    texts = await db.fetch_texts(product_ids)
    embeddings = await embed_v2(texts, model='text-embedding-3-large')
    await pinecone_v2.upsert(zip(product_ids, embeddings))

# Phase 2: Validation before cutover
def validate_migration(sample_size=10_000):
    test_queries = load_golden_queries()  # 500 known good query-product pairs
    recall_v1 = evaluate_recall(pinecone_v1, test_queries)
    recall_v2 = evaluate_recall(pinecone_v2, test_queries)
    assert recall_v2 >= recall_v1 * 0.98  # allow max 2% regression

# Phase 3: Atomic cutover via feature flag
FEATURE_FLAGS['embedding_index'] = 'v2'  # LaunchDarkly / Flagsmith
```

**Follow-up — New products during migration?**  
**Dual-write** to v1 and v2 from product-creation handler (e.g. Kafka-triggered). Remove dual-write after stable cutover.

**Note:** Do not delete v1 until v2 runs ~1 week in prod with no recall regression in A/B metrics.

---

### Scenario 3 — RAG hallucinating numbers in finance reports *(Fintech / Banking — Hard)*

**Situation**

A **bank or asset-management** team deployed internal RAG over **10-Q / 10-K PDFs**, earnings slides, and MD&A sections so analysts can ask questions in chat (“What was revenue last quarter?”, “How did net interest margin change?”). Retrieval returns chunks that *look* relevant—the right company, the right quarter, tables nearby—so stakeholders trust the answers.

An analyst notices that the model stated revenue as **$2.3B** when the filing actually says **$2.1B**. The scary part is that the answer **sounds authoritative**: correct units, plausible rounding, no obvious “I don’t know.” After investigation you confirm that **the correct figure exists somewhere in the corpus**, yet the model still emitted the wrong number—this is not simply “missing context”; it is **miscomposition** of numeric evidence.

Downstream, wrong numbers could inform **internal models, executive summaries, or client-facing drafts**, so risk and compliance are involved. The question is **why** numeric hallucinations persist even when documents contain the truth, and **what architecture** (chunking, structured extraction, tools, verification) prevents them.

**Interviewer:** Why happens even when figures exist in documents?

**You:** (1) **Numeric split across chunks** — table header vs values separated. (2) **Interpolation** across quarters in one context. (3) **PDF extraction corruption** — `$2,310` mangled or de-contextualised.

**Solution architecture — numerical precision**

- **Table-aware chunking** — `pdfplumber` → structured JSON `{quarter, metric, value, unit}`; embed JSON; LLM sees clean structure.  
- **SQL sidecar** — ingest metrics to Postgres; tool `get_metric(company, quarter, metric)` for exact numbers.  
- **Faithfulness guardrail** — post-gen NLI / string verify: every number in answer appears verbatim in retrieved chunks.  
- **Citation forcing** — every number must cite `[Source: chunk_id, exact_quote]`.

```python
# Numeric faithfulness guardrail
import re

def verify_numeric_claims(response: str, retrieved_chunks: list[str]) -> bool:
    numbers = re.findall(r'\$[\d,\.]+[BMK]?|[\d,\.]+%', response)
    combined_context = ' '.join(retrieved_chunks)
    for num in numbers:
        if num not in combined_context:
            return False  # hallucinated number detected
    return True

# If verification fails, retry with stricter prompt
if not verify_numeric_claims(response, chunks):
    response = llm.call(prompt + '\nYou MUST only use numbers found verbatim in the provided context.')
```

---

