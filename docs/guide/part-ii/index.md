# Part II — Real-world scenario interviews

**Themes:** RAG · Agentic AI · Memory · Production systems  

**Scope:** 15 scenarios drawn from production incidents, Big Tech system design interviews, and startup constraints — each written so you can practice **dialogue** (interviewer prompt → your structured answer → follow-ups).

!!! abstract "What makes a strong answer"
    Interviewers are not checking trivia — they want **structured reasoning**: name what broke (**situation**), cite the missing invariant (**principle**), propose concrete systems (**action**: metadata schemas, filters, harness gates), and close with **resilience** (evals, CI gates, monitoring). The **[SPAR framework](chapter-05-advanced.md#interview-answer-framework-spar)** at the end of Chapter 5 is the recap checklist.

    Cross-cutting reminders from Part I: **latency** stories should separate **wall-clock** vs **TTFT/streaming**; **retrieval** failures often trace to **chunk granularity** and **embedding neighborhoods**, not only “bad embeddings.”

---

## Why scenarios are grouped this way

| Chapter | What you are proving | Typical interviewer hook |
|---------|----------------------|---------------------------|
| **[:octicons-flame-24: Production incidents](chapter-01-production.md)** | Data lifecycle, migrations, numeric grounding | “CEO escalation — wrong clause cited” |
| **[:octicons-graph-24: System design](chapter-02-system-design.md)** | Latency budgets, tenancy, cost | “10M DAU — 100ms P99” |
| **[:octicons-robot-24: Agentic AI](chapter-03-agentic.md)** | Safety, progress models, conflicts | “Agent emailed 10k customers” |
| **[:octicons-database-24: Memory & context](chapter-04-memory.md)** | Session design, token economics | “Bot forgets after 20 turns” |
| **[:octicons-light-bulb-24: Advanced & SPAR](chapter-05-advanced.md)** | Scale pathology, personal memory, security, eval maturity | “Faithfulness dropped after 10× corpus growth” |

---

## Chapter links

1. [:octicons-arrow-right-24: Chapter 1 — Production incident scenarios](chapter-01-production.md) — scenarios 1–3  
2. [:octicons-arrow-right-24: Chapter 2 — System design scenarios](chapter-02-system-design.md) — scenarios 4–5  
3. [:octicons-arrow-right-24: Chapter 3 — Agentic AI scenarios](chapter-03-agentic.md) — scenarios 6–8  
4. [:octicons-arrow-right-24: Chapter 4 — Memory & context](chapter-04-memory.md) — scenarios 9–10  
5. [:octicons-arrow-right-24: Chapter 5 — Advanced & tricky](chapter-05-advanced.md) — scenarios 11–15 + **SPAR**
