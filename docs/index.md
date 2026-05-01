---
hide:
  - toc
---

<div class="doc-hero" markdown="1">

# RAG & Agentic AI interview playbook

**Senior / staff ML & LLM engineering** — latency, chunking, retrieval, agents, harnesses, production incidents, and system design in one place.

Compiled by **Ritesh Yadav**.

[:octicons-arrow-right-24: Table of contents & full map](guide/index.md){ .md-button .md-button--primary }
[:octicons-book-24: Start Part I — Preparation](guide/part-i.md){ .md-button }

</div>

## What’s inside

<div class="grid cards" markdown>

- :material-speedometer: **Part I — Deep prep**

    ---

    RAG latency budgets, chunking, HyDE / FLARE / Self-RAG, hybrid search & RRF, ReAct, harness design, memory, context compaction, multi-agent patterns, RAGAS-style evaluation.

    [:octicons-arrow-right-24: Open Part I](guide/part-i.md)

- :material-domain: **Part II — Real scenarios**

    ---

    Fifteen production-grade cases across five chapters — incidents, system design, agent safety, memory, advanced topics + **SPAR** framework.

    [:octicons-arrow-right-24: Part II overview](guide/part-ii/index.md)

- :material-sitemap: **Part III — HLD / LLD**

    ---

    RAG gateways, compliance chat design, guardrail layers, red teaming in CI, agent workflows, HITL escalation, tool contracts, and replay for incidents.

    [:octicons-arrow-right-24: Open Part III](guide/part-iii.md)

</div>

!!! tip "How to use this site"
    Open **[Table of contents](guide/index.md)** for the full map. The **left sidebar** lists every chapter; each article shows an **on-page TOC** on wide screens. **Search** works across all sections (“HNSW”, “RRF”, “SPAR”, “injection”, …).

---

## Deploy on Vercel

This project builds a **static site** with MkDocs Material. Connect the repo to Vercel and use the included `vercel.json` (build runs `mkdocs build`; output is `site/`).
