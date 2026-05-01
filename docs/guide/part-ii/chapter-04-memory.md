## Chapter 4 — Memory & context scenarios

!!! tip "Focus for this chapter"
    Separate **what must stay verbatim** (recent turns, explicit user constraints) from **what can be summarized** (old chit-chat). Structured session state is often the missing piece in “forgetful bot” incidents.

### Scenario 9 — Support bot forgets context after ~20 turns *(SaaS / CX — Medium)*

**Situation**

You run a **customer-support chatbot** embedded in a SaaS product. Traffic is meaningful—on the order of **thousands of conversations per day**. Sessions are not “one-turn FAQ”: users explain billing disputes, migration problems, or flaky integrations across **many messages**; the average conversation length is **~35 turns**.

Engineering chose a **fixed token budget** (~16K tokens) for the model context and implemented the simplest policy: keep only the **most recent messages** and drop the rest. After roughly **20 turns**, early details disappear—**order IDs**, **prior troubleshooting steps**, **promises the bot made**, or **emotional context** (“I already tried that twice”). Customers experience this as **amnesia**: the bot repeats questions or contradicts itself; CSAT drops and human handoffs increase.

The business constraint is you cannot infinitely grow context per ticket; you need a **memory architecture** that preserves **stable facts** and **task state** while still summarizing long chatter.

**Interviewer:** Memory architecture?

**You:** Naive buffer drops oldest turns. Use **three tiers:** (1) **structured session state** (always in context), (2) **rolling summary** of mid history, (3) **last N verbatim** turns.

```python
class SupportSessionMemory:
    session_state: dict = {
        'issue_category': 'billing',
        'account_id': 'USR-12345',
        'issue_summary': 'Double charged for Pro plan in October',
        'steps_taken': ['verified identity', 'pulled invoice #INV-889'],
        'sentiment': 'frustrated',
        'resolution_status': 'in_progress'
    }
    history_summary: str = ''
    recent_turns: list = []

    def build_context(self) -> str:
        return (
            f'SESSION STATE: {json.dumps(self.session_state)}\n'
            f'HISTORY SUMMARY: {self.history_summary}\n'
            f'RECENT TURNS: {format_turns(self.recent_turns)}'
        )

    def update(self, turn):
        self.recent_turns.append(turn)
        self.session_state = llm.extract_state(turn, self.session_state)
        if len(self.recent_turns) % 5 == 0:
            self.history_summary = llm.summarise(self.recent_turns[:-6])
            self.recent_turns = self.recent_turns[-6:]
```

**Insight:** `session_state` preserves facts **outside** lossy summarisation.

---

### Scenario 10 — Context cost explosion on long documents *(Enterprise AI — Hard)*

**Situation**

Legal or procurement teams use your assistant to query **very large contracts**—often **hundreds of pages**, dense definitions, schedules, and amendments. The retrieval pipeline is conservative: for each question it pulls **many overlapping chunks** (say **~50 chunks**) to avoid missing a niche clause. Combined prompts routinely reach on the order of **~30K tokens** before the model even answers.

Finance reports **~$180K/month** in LLM spend attributable to this workload alone (flagship model pricing × volume × long prompts). Leadership asks for roughly **~80% cost reduction** **without** turning answers into useless summaries—lawyers will reject the product if citations drift or nuance is lost.

So you must explain **how to shrink retrieved evidence** (reranking, compression), **route cheaper models** where safe, **cache** repeated clause lookups, and measure **quality** while dollars fall.

**Interviewer:** Strategy?

**You:** Four levers: fewer tokens/query, cheaper model units, cache repeats, skip LLM for trivial lookups.

**Lever 1 — Aggressive compression** — Cross-encoder → top-5 chunks; LLMLingua-2 ~3× on those → ~30K → ~2K tokens on context.

**Lever 2 — Model routing**

```python
def route_to_model(query: str, context_tokens: int) -> str:
    complexity = classifier.score(query)
    if complexity < 0.3:
        return 'gpt-4o-mini'
    elif context_tokens < 4000:
        return 'gpt-4o-mini'
    return 'gpt-4o'
```

**Lever 3 — Semantic cache** — Key `(query_embedding, hash(top_chunk_ids))`; TTL 24 h; often 35–45% hit on repetitive contract Q&A.

**Expected savings (illustrative)**

| Optimisation | Effect |
|--------------|--------|
| Rerank 50→5 | ~90% fewer context tokens |
| LLMLingua 3× | Further reduction on remainder |
| 70% to mini | Much cheaper per token on bulk |
| 40% cache hit | Fewer calls |

Combined: example trajectory ~**$180K → ~$18K/month** (order-of-magnitude; tune to your traffic).

---

