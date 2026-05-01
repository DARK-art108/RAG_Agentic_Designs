# Part III — Extended: GenAI System Design, Guardrails & Agentic AI HLD/LLD

This part adds **new** complex questions with **scenario framing**, **ideal reasoning**, and **architecture hooks** you can use in staff-level interviews. It **does not** replace Parts I–II; it deepens system design, guardrails, and agent HLD/LLD.

!!! abstract "Staff-level angle"
    Parts I–II build **mechanisms** (indexes, harnesses, filters). Part III builds **products**: gateways with quota and audit, compliance-aware retrieval, policy engines that sit **above** the LLM, and multi-agent graphs with explicit failure domains. Read III.A–III.D as modular blocks you can reorder in an interview whiteboard.

!!! success "Carry these phrases"
    **“Immutable retrieval snapshot”**, **“policy service as single enforcement point”**, **“replay from events, not from screenshots of CoT”**, and **“degraded mode when vector path is unhealthy”** signal maturity beyond a single happy-path diagram.

---

## III.A — GenAI & RAG system design (complex Q&A)

### Q1 — Scenario: “Compliance officer chat” across jurisdictions

**Situation**

A **global bank** (or insurer) runs one internal **GenAI assistant** for tens of thousands of employees: HR policies, trading conduct rules, privacy notices, and breach-playbooks live in different **regions**. Regulatory reality is harsh: **EU** employees must see answers grounded **only** in EU-approved policy packs; **US** staff see US variants—and **mixing** jurisdictions in one answer is not a wording issue, it can be a **regulatory finding**.

Legal expects **audit**: for any sensitive answer, they want to show **exact clause IDs**, **document versions**, and **hashes** that were in context at generation time—not “the model said so.” Users hop between topics quickly and sometimes **travel** or **switch entities**, so a naive chat session could accidentally pull US chunks for an EU question unless **session policy** is explicit.

**Scenario (requirements):** One assistant; corpus spans **EU** and **US** authoritative docs; answers must **never blend** jurisdictions; auditors must prove **which doc version** grounded **each normative sentence**.

**What interviewers probe:** tenancy, retrieval filters, grounding, logging.

**Strong answer structure**

1. **Data plane:** Ingest with `jurisdiction`, `effective_date`, `doc_hash`, `clause_id`; embeddings per chunk with same metadata.  
2. **Control plane:** Router detects jurisdiction from user profile + explicit session selector; **mandatory filter** on retrieve.  
3. **Generation:** Prompt forces **per-sentence citations** mapping to `clause_id`; post-check that every normative claim has citation.  
4. **Observability:** Store `retrieval_snapshot_id` (chunk IDs + versions + hashes) immutably for 7 years; correlate to model invocation trace.

**Failure mode:** User switches jurisdiction mid-thread — maintain **session-scoped policy lock** unless user confirms switch.

---

### Q2 — Scenario: Real-time vs batch knowledge for same product

**Situation**

You ship **shopping assistant** Q&A on a large marketplace: “Does this laptop ship to my ZIP?” “Is this SKU in stock?” “What’s the warranty?” Some facts live in **marketing copy and PDF manuals**—they change occasionally and embed well in a **vector index**. Other facts—**price, promotions, inventory, delivery ETA**—change **minute by minute** from **OLTP / catalog services**. If the model reads yesterday’s embedded snippet for “price,” you create **angry customers** and **pricing-compliance** risk.

Product wants **one conversational surface**, not two bots. Engineering wants **clear ownership**: docs vs transactional APIs. Reliability teams worry about **latency** when every answer hits multiple backends.

**Scenario (requirements):** Same assistant must combine **stable** narrative knowledge with **live** inventory and price—without trusting stale chunks for **numbers**.

**Design:** **Hybrid retrieval** — (a) vector index for docs; (b) **structured tools** (`get_sku_price`, `get_inventory`) called **after** intent classification; **never** trust stale chunks for numbers. **Orchestration:** Planner emits DAG — parallel doc retrieval + API fetch → merge in canonical JSON → LLM explains.

**Trade-off:** Higher latency; mitigate with aggressive caching on catalog APIs with short TTL and cache invalidation on webhook.

---

### Q3 — Design a “RAG gateway” as an enterprise API product

**Situation**

Your company sells **API access** to “enterprise search + answer,” similar to packaging Pinecone + OpenAI behind **your** controls. Buyers are banks and hospitals: they demand **SSO**, **per-tenant isolation**, **SOC2 evidence**, **immutable audit**, predictable **SLAs**, and **cost caps**. Internal teams also want **feature flags** for experimental rerankers without shipping ten microservices to each customer.

So you are not sketching one chatbot—you are designing the **shared edge** where auth, quotas, routing, safety, and observability **must** live once, while retrieval strategies remain pluggable.

**Scenario (requirements):** Multi-tenant **RAG-as-a-service**: stable external API, strong governance, graceful degradation when vector DB or LLM vendor flakes.

**HLD:** Edge AuthN/Z → **quota/rate** → **query planner** (rewrite, HyDE optional) → **router** (SQL vs vector vs KG) → **retrieval** (hybrid + RRF) → **guardrail** (PII, injection scan on packs) → **rerank** → **packer** (lost-in-the-middle aware ordering) → **LLM** → **post-validator** (citations, numeric grounding) → **audit log**.

**LLD highlights:** Idempotent `request_id`; **streaming** with trace IDs per token phase; **circuit breakers** per downstream (embedder, DB, LLM); **degraded mode** — keyword-only search + template answer when vector path unhealthy.

---

### Q4 — Cold start + long-tail queries at scale

**Situation**

You launch a **new marketplace** (or expand into a new category). Search logs are **thin**: most SKUs have almost **no clicks**, reviews are sparse, and query distributions are **heavy-tailed**—“obscure adapter for 2016 appliance model” appears once a month but still needs a correct match. Pure **dense retrieval** trained only on organic behavior **collapses**: embeddings for rare items sit in generic regions of space and lose to popular neighbors.

Growth teams push **fast catalogue ingestion** from sellers with noisy titles; ML wants **quality without waiting six months** for click data. Ops worries about **cold-start sellers** who never surface.

**Scenario (requirements):** Sparse behavioral signals; embeddings weak on **long-tail** SKUs; still need **reasonable retrieval** on day one and a path to **learn** as traffic grows.

**Mitigations:** **Synthetic Q generation** from PDP attributes for training retrieval; **BM25-heavy hybrid** early; **human-in-loop** labels for top failure buckets weekly; **two-tower** or **late interaction** rerankers once traffic exists; **exploration** slot in top-k for new listings.

---

## III.B — Guardrails, safety & responsible AI (complex Q&A)

### Q5 — Layered guardrail architecture (input → tool → output → org)

**Situation**

You operate a **customer-support agent** wired into Zendesk-style tickets, order history, and internal CRM. It can **summarize** threads, **draft** replies, but also invoke tools: **issue refunds**, **change subscription tiers**, **grant credits**, **open fraud cases**. Each capability is useful—and each is a **weapon** if the model is jailbroken, prompted indirectly via ticket attachments, or simply **wrong** about eligibility.

Security expects **defense in depth**, not “we wrote a safe system prompt.” Compliance asks **who approved** high-impact actions and whether policy checks are **deterministic** rather than LLM-judged alone.

**Scenario (requirements):** Agent reads sensitive tickets and can **mutate** customer state; design guardrails from **user input** through **tools** to **org controls**.

**Layers**

| Layer | Responsibility | Example mechanisms |
|-------|----------------|---------------------|
| **Input** | Toxicity, PII exfil attempts, jailbreak | Classifiers; allowlists for outbound domains |
| **Retrieval** | Injection in docs | Sanitise ingest; delimiters; never elevate retrieved text to system |
| **Tool** | Schema validation; capability matrix | Pydantic; OAuth scopes per tool; MFA for refunds >$X |
| **Output** | Policy, hallucination, brand | NLI grounding; regex + semantic judge; red-team prompts |
| **Org** | Audit, escalations | SIEM alerts on anomaly tool volume |

**Scenario twist:** Model proposes refund above policy — **business rules engine** evaluates **before** harness executes tool; LLM cannot bypass.

---

### Q6 — Red teaming pipeline in CI

**Situation**

Your team ships **weekly**: new prompts, retrieval tweaks, occasional **model version** bumps. Product velocity is high; safety reviews cannot be a **manual gate** every time. Yet one bad merge can re-open **prompt injection**, **PII leakage**, or **toxic** outputs—especially after changes to **tool schemas** or **context templates**.

Leadership asks for **proof** that automation catches regressions **before** prod, and **canaries** that roll back when live metrics drift—not heroics from the on-call red-teamer.

**Scenario (requirements):** Frequent prompt/model changes; must block **safety** and **privacy** regressions in CI/CD with measurable thresholds.

**Design:** Curated **attack suite** (injection strings, role escapes, data exfil templates); **LLM-as-judge** + **deterministic** checks; **block merge** if attack success rate > baseline + epsilon; **canary** in prod with automatic rollback on safety KPI drift.

---

### Q7 — PII handling across RAG and fine-tuning

**Situation**

Support transcripts—full of names, addresses, order IDs, sometimes **government IDs**—were ingested into the **vector DB** so the assistant could retrieve similar past tickets. Now ML engineers propose **fine-tuning** on that same corpus to improve tone and resolution quality. Privacy officers panic: **indexes and training exports** multiply copies of sensitive data; **re-identification** risk rises; regions disagree on **lawful basis** and retention.

You must explain how **RAG’s need for readable text** interacts with **training’s need for diverse examples**—and why “dump Redis to S3” is not a policy.

**Scenario (requirements):** Same sensitive transcripts feed **retrieval** and a proposal for **fine-tuning**; need architecture that minimizes raw PII proliferation.

**Answer:** **Detect & mask** at ingest (NER + vault tokens); **store only surrogates** in indexes used for training exports; maintain **reversible mapping** in restricted KMS for narrow ops roles; **never** fine-tune on raw PAN/SSN — use **synthetic** replacement spans verified by privacy review.

---

## III.C — Agentic AI — high-level design (HLD)

### Q8 — HLD: Multi-agent research assistant with external APIs

**Situation**

Users ask open-ended **research tasks**: “Summarize clinical evidence on drug X,” “Build a comparison table from these URLs,” “Draft code that calls our billing API.” One monolithic prompt+browse loop **burns context**, hides failure modes, and makes debugging a **nightmare**. Product wants **specialized behaviors**—planning, retrieval, coding, verification—while platform wants **one orchestrated workflow** with retries, billing, and tracing.

External APIs (search, GitHub, proprietary REST) add **rate limits**, **partial failures**, and **PII** exposure risks. You need an HLD where agents cooperate **without** an infinitely recursive ping-pong.

**Scenario (dynamic behavior):** A **Critic** agent flags an unsupported factual claim; the system must loop **back** to retrieval with a sharper query—but **not forever**.

**Actors:** User → **API Gateway** → **Session orchestrator** → subgraph of agents (**Planner**, **Retriever**, **Coder**, **Critic**).

**State:** Durable workflow store (e.g. event log per `run_id`); checkpoints after each tool batch.

**Cross-cutting:** Feature flags for model/tools; cost budgets per user tier; **dead-letter** queue for failed tool spans.

**Closing loop:** Critic flags unsupported claim → Retriever gets narrowed query — **bounded** by max revisits and diminishing returns detector.

---

### Q9 — HLD: Human-in-the-loop (HITL) escalation graph

**Situation**

Half your agent runs are mundane lookups; the other half touch **money**, **legal commitments**, or **angry VIP customers**. Fully autonomous completion is unacceptable—yet forcing humans on every step destroys ROI. You need **selective escalation**: the agent runs fast until a **predicate** fires (risk, uncertainty, abuse), then **pauses** with enough context that a human can decide in **minutes**, not hours.

Regulators and execs also ask: **what did the human approve**, and can we **replay** it for audits? Exporting raw chain-of-thought is often **wrong** legally and practically.

**Scenario (requirements):** Define **when** to stop, **what packet** the human sees, and **how** execution resumes without silent drift.

**Triggers:** Irreversible tool; confidence < τ; policy engine uncertainty; user frustration signal from CX models.

**Flow:** Agent suspends → writes **structured escalation packet** (goal, attempts, evidence IDs) → human UI → human edits **plan** or **answers** → resume from checkpoint with **immutable** human decision logged.

**Design principle:** Humans edit **plans/state**, not raw hidden chain-of-thought blobs — clearer liability and training signal.

---

### Q10 — Fleet of autonomous vs supervised agents

**Situation**

Your org doesn’t have “one agent”—it has **dozens**: onboarding bots, DevOps helpers, sales outreach drafts, internal doc copilots. Some are **read-only**; others propose **writes**; still others run **batch jobs** overnight. Security reviews grind to a halt if every team reinvents OAuth scopes and confirmations differently. Auditors want evidence that **policy is centralized**, not scattered across prompts.

Interview framing: how do you **classify** agents, **roll out** autonomy gradually, and prove **who was allowed** to do **what**?

**Scenario (requirements):** Mixed fleet of agents with different risk profiles; need consistent governance without blocking innovators entirely.

**Taxonomy:** **Read-only** agents (default), **write-capable** with confirmation, **batch** agents with progressive rollout.

**HLD:** Central **policy service** answers “may agent X invoke tool Y under context Z?” — single enforcement point for SOC2 evidence.

---

## III.D — Agentic AI — low-level design (LLD)

### Q11 — LLD: Tool registry contract

**Situation**

Interviewers shift from boxes-and-arrows to **contracts**: “What exactly does the harness require before `execute()`?” Teams break because tool payloads are **ad hoc strings**, retries **double-charge** customers, or sandbox timeouts leave **zombie** processes. You need a **registry** that is machine-verifiable and aligns with **IAM** scopes.

**Scenario (requirements):** Precisely specify tool metadata and the ordered **harness pipeline** so implementations stay consistent across services.

Each tool exposes:

- `name`, `description`, **JSON Schema** for args, **risk_class** (`read`, `mutate`, `irreversible`), **timeout_ms**, **idempotency_key** support, **rate_limit**, **required_auth_scope`.

**Harness execution steps:** validate schema → policy check → **budget check** (tokens + USD + wall clock) → execute in sandbox → normalise observation → append to trace → checkpoint.

---

### Q12 — LLD: Deterministic replay and debugging

**Situation**

Production incident: an agent issued **wrong refunds** or **leaked** partial ticket contents. Postmortem asks: **can we replay** the exact reasoning path? Raw logs show streaming tokens—not helpful. Vendor APIs (**search**, **weather**, **payments**) are **non-deterministic** across days. Without disciplined **event capture**, you cannot answer whether the bug was **model**, **tool**, **prompt**, or **bad data**.

**Scenario (requirements):** Incident investigation needs **reproducible** agent behavior **enough** for engineering—not necessarily bit-identical creativity.

**LLD:** Persist ordered **events**: `{model_snapshot_id, temperature, messages_hash, tool_inputs_hash, tool_outputs_redacted, rng_seed_if_any}`. Replays use **frozen** model version where possible; for non-deterministic APIs, **mock** with recorded stubs in staging.

---

### Q13 — LLD: Context assembly for long ReAct traces

**Situation**

After fifteen **Thought → Action → Observation** cycles, your context holds megabytes of **logs**, **HTML dumps**, and **JSON**. Costs spike and models **lose** the original user goal—but if you blindly summarize, you drop the **one error line** that explains the bug. Teams ask for **reference-based memory**: keep pointers to blobs, not blobs inline.

**Scenario (requirements):** Long ReAct runs must stay within token budgets while preserving **debuggability** and **goal fidelity**.

**Techniques:** Tool-output **summarisation** with **reference IDs** to full blobs in object storage; **scratchpad compression** task-aware; pin **user goals** and **active constraints**; **retrieve** prior substeps from episodic store instead of inline full history.

---

### Q14 — Multi-agent LLD: Message bus vs shared blackboard

**Situation**

You split work across agents (**researcher**, **writer**, **reviewer**). Now they must share intermediate artifacts—snippets, structured facts, critique notes. Two camps emerge: engineers love a **shared blackboard** (fast, simple dict in Redis); platform wants a **message bus** (Kafka/NATS) for ordering, retries, and cross-service boundaries. Tenancy blurs when **Team A’s** agent accidentally reads **Team B’s** scratchpad.

**Scenario (requirements):** Compare coordination mechanisms for **multi-agent state** with production-grade **audit** and **isolation** constraints.

| Pattern | Pros | Cons |
|---------|------|------|
| **Message bus** | Loose coupling; clear audit per hop | Higher latency; envelope versioning |
| **Shared blackboard** | Fast iteration | Contention; harder tenancy boundaries |

**Production hybrid:** Blackboard for **ephemeral working set**; bus for **cross-team agent** boundaries with signed envelopes.

---

## III.E — Integrated capstone scenarios (staff-level)

### Capstone A — “Autonomous SRE agent” with blast-radius controls

**Situation**

On-call is drowning: paging storms, flaky deploys, noisy alerts. Leadership proposes an **SRE copilot** that reads **metrics**, **logs**, **traces**, and internal **runbooks**, then suggests—or executes—**mitigations** (scale up, rollback feature flag, restart pods). Every ops engineer hears “autonomous” and thinks **production fire**: one bad restart at cluster scope could exceed **every** incident you prevented.

You must articulate **progressive trust**: what the agent may **read**, what it may **propose**, what requires **human + peer**, and how **blast radius** stays bounded.

**Goal:** Diagnose incidents using logs + metrics + runbooks.

**Non-negotiables:** Read-only prod by default; **mutating** runbooks require **two-person rule**; **canary** remediation on single host; **automatic rollback** hooks; full trace exported to incident ticket.

**Talking points:** Progressive autonomy ladder; **simulation** environment mirroring prod schemas with masked data.

---

### Capstone B — Contract negotiation copilot

**Situation**

Legal teams paste a **counterparty’s redlined PDF** into chat and ask: “Are we exposed on liability caps?” Your copilot must reconcile **your clause library**, **their edits**, and **firm policy** (“never unlimited liability,” jurisdiction quirks). Wrong advice isn’t a bad UX moment—it’s **malpractice risk**. Lawyers insist on **human sign-off** before anything leaves the building, but still want **draft speed**.

**Scenario (requirements):** Combine retrieval over **approved clauses**, structural **diff** against counterparty text, and **policy rules**—without autonomous outbound email.

**Architecture sketch:** RAG over clause library + **diff** tool against counterparty PDF + **policy graph** (“never accept unlimited liability”).  

**Guardrails:** jurisdiction-specific clause packs; **human approval** before sending redlines; **version** counterparty doc snapshots.

---

### Capstone C — Scientific literature agent with provenance

**Situation**

Biotech or clinical researchers ask: “What’s the evidence for intervention Y in population Z?” Answers draw from **PubMed**, internal lab notes, and regulatory submissions—sources **conflict**, methodologies differ, and stakes are **patient safety**. A fluent summary that **hides disagreement** is worse than useless.

Interview expectation: show **epistemic humility**—retrieve broadly, **actively seek contradiction**, and surface **evidence strength**, not fake consensus.

**Scenario (requirements):** Iterative retrieval plus an explicit **adversarial** pass; outputs must encode **provenance** and **uncertainty**.

**Mechanisms:** HyDE + iterative retrieval + **opposition** agent that searches **contradicting** trials.  

**Output:** Claim chart with **GRADE-style** certainty labels when sources conflict.

---

## III.F — Quick checklist for Agentic HLD/LLD interviews

**Situation**

You’re at the whiteboard endgame: time is short and the interviewer wants **coverage**, not poetry. They’ll ping-pong across **state ownership**, **retries**, **security**, **cost**, and **eval**. The checklist below is a **mental sweep**—use it to avoid forgetting something embarrassing (“no idempotency on payments”) after spending ten minutes on retrieval math.

- **Boundaries:** Who owns state — orchestrator vs agents vs external workflow engine?  
- **Failure domains:** Partial retries; idempotent tools; compensation sagas for multi-step writes.  
- **Security:** Injection surfaces (web, email, docs); secret scopes; sandbox egress policies.  
- **Cost:** Per-step token budgets; model cascade; cache embeddings and retrieval packs.  
- **Eval:** Offline trajectories + online outcome metrics + human review on escalations.

---

*End of merged guide — Parts I–II preserve PDF content; Part III extends with additional system design, guardrails, and agentic HLD/LLD material.*
