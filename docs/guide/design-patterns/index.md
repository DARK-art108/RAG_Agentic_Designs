# Reference architectures & patterns

These pages pair **high-level diagrams** with **beginner-friendly explainers** that tie back to concepts from [Part I](../part-i.md): hybrid retrieval, agentic orchestration, latency (wall-clock vs streaming / TTFT), and chunking vs “what the model actually reads.”

| Page | What it illustrates |
|------|---------------------|
| [Advanced multi-agent RAG (millions of documents)](advanced-multi-agent-rag.md) | Query intelligence → agentic router → fusion retrieval → context engineering → specialist swarm → Self-RAG / CRAG → indexing strategies |
| [Pharma clinical intelligence platform](pharma-clinical-platform.md) | Multi-modal healthcare data, GraphRAG + clinical orchestrator, HIPAA / 21 CFR Part 11 posture, multi-region LLMOps |
| [Banking LLMOps RAG](banking-llmops-rag.md) | Document ingestion, pgvector + replicas + semantic cache, LangGraph orchestrator, cross-region latency SLOs |

!!! tip "How to read any diagram here"
    **Latency:** Diagram SLOs like “P99 &lt; 300–400 ms” usually mean **wall-clock** for a bounded path (gateway → retrieval → generate first byte or full response). With **streaming**, users often perceive **TTFT** separately from total decode time — say both in interviews.  
    **Quality vs speed:** Fusion retrieval, reranking, and critic loops improve grounding but add **parallel wall-clock** work unless you aggressively pipeline or trim stages.

---

## Quick cross-links

- [Part I — Hybrid + RRF, HyDE, Self-RAG](../part-i.md)  
- [Part III — Gateway / guardrails / HLD](../part-iii.md)  
- [Part II — Scenario drills](../part-ii/index.md)
