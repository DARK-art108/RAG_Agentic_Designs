# Guide overview & table of contents

Welcome to the structured interview guide. Pages are split so the **left sidebar** lists every major section, and each page has an **on-page table of contents** on the right (desktop). Use **search** for terms like `HNSW`, `RRF`, `SPAR`, or `injection`.

!!! tip "How to study this material"
    **First pass:** skim Part I for vocabulary and numbers you can quote in interviews.  
    **Second pass:** work through Part II scenarios aloud — cover Situation → Principle → Action → Resilience (**[SPAR](part-ii/chapter-05-advanced.md#interview-answer-framework-spar)**).  
    **Staff prep:** use Part III for gateway/HLD/LLD arcs and capstones.

---

## Full map

### [:octicons-book-24: Part I — Advanced preparation](part-i.md)

Deep dives you will reuse in almost every RAG or agents conversation:

!!! note "Two phrases to reuse everywhere"
    **Latency:** Always distinguish **wall-clock** (full response / SLA timers) from **user-perceived** latency — with streaming, lead with **TTFT** (time-to-first-token) vs **time-to-last-token** / total decode.  
    **Chunking:** Always tie **chunk granularity** to **embedding neighborhoods** (which passages become vector neighbors) **and** to **prompt reality** (concatenated tokens the LM reads after pack/rerank — see lost-in-the-middle).

| Section | Topics |
|---------|--------|
| §1–2 | Latency instrumentation, caching, streaming, chunking & parent–child |
| §3–4 | HyDE, FLARE, Self-RAG, hybrid + RRF, agentic vs naive RAG, ReAct |
| §5–7 | Harness design, memory tiers, compaction vs summarisation |
| §8–10 | Multi-agent patterns, RAGAS metrics, rapid-fire Q&A, cheat sheet |

---

### [:octicons-package-dependencies-24: Part II — Real-world scenarios](part-ii/index.md)

Fifteen production-style cases, grouped by theme:

| Chapter | Focus | Page |
|---------|--------|------|
| **1** | Production incidents (stale docs, embeddings, finance numerics) | [:octicons-link-external-24: Open](part-ii/chapter-01-production.md) |
| **2** | System design (scale, multi-tenant isolation) | [:octicons-link-external-24: Open](part-ii/chapter-02-system-design.md) |
| **3** | Agentic failures (blast radius, loops, conflicting sources) | [:octicons-link-external-24: Open](part-ii/chapter-03-agentic.md) |
| **4** | Memory & context (support bots, cost explosion) | [:octicons-link-external-24: Open](part-ii/chapter-04-memory.md) |
| **5** | Advanced cases + **SPAR** answer framework | [:octicons-link-external-24: Open](part-ii/chapter-05-advanced.md) |

Start here if you prefer **story-driven** learning: [:octicons-arrow-right-24: Part II introduction](part-ii/index.md)

---

### [:octicons-git-branch-24: Part III — System design, guardrails, HLD/LLD](part-iii.md)

Staff-level extensions: compliance-aware RAG, RAG gateway shape, guardrail layers, red teaming in CI, multi-agent HLD, tool contracts, deterministic replay, and capstones (SRE agent, contract copilot, literature agent).

---

!!! note "Source manuscript"
    The canonical Markdown manuscript also lives at the repo root as `Scenarios.md`. The **MkDocs site** uses the split files under `docs/guide/` for navigation and faster loading.
