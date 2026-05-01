---
hide:
  - toc
---

<div class="doc-hero" markdown="1">

# RAG & Agentic AI interview playbook

**Senior / staff ML & LLM engineering** — latency, chunking, retrieval, agents, harnesses, production incidents, and system design in one place.

Compiled by **Ritesh Yadav**.

[:octicons-arrow-right-24: Open the full interview guide](scenarios.md){ .md-button .md-button--primary }

</div>

## What’s inside

<div class="grid cards" markdown>

- :material-speedometer: **Part I — Deep prep**

    ---

    RAG latency budgets, chunking, HyDE / FLARE / Self-RAG, hybrid search & RRF, ReAct, harness design, memory, context compaction, multi-agent patterns, RAGAS-style evaluation.

- :material-domain: **Part II — Real scenarios**

    ---

    Fifteen production-grade cases — stale legal RAG, embedding migrations, numeric hallucinations, multi-tenant isolation, agent safety, memory at scale, injection defence, and more.

- :material-sitemap: **Part III — HLD / LLD**

    ---

    RAG gateways, compliance chat design, guardrail layers, red teaming in CI, agent workflows, HITL escalation, tool contracts, and replay for incidents.

</div>

!!! tip "How to use this site"
    Use the **search** bar for topics like “HNSW”, “RRF”, or “prompt injection”. On long pages, the **table of contents** tracks your scroll.

---

## Deploy on Vercel

This project builds a **static site** with MkDocs Material. Connect the repo to Vercel and use the included `vercel.json` (build runs `mkdocs build`; output is `site/`).
