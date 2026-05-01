## Chapter 3 — Agentic AI real-world scenarios

!!! tip "Focus for this chapter"
    Emphasize **blast radius**: reversible vs irreversible tools, confirmations, rate limits, dry-run, and escalation when the agent loses **progress signal**. Security-minded answers beat clever prompting alone.

### Scenario 6 — Agent sent 10,000 emails by mistake *(Enterprise SaaS — Hard)*

**Situation**

Your company shipped an **AI assistant for customer-success teams** that can **draft outreach emails** based on CRM context (renewals, outages, onboarding milestones). In staging, the tool wrote drafts into a queue for humans to approve. In production, an integration mistake—or an overly permissive flag—meant the **`send_email`** action was wired to the **live transactional provider**, not the sandbox.

Over roughly **two hours**, the agent (or a scheduled batch job invoking it) executed **~10,000 real sends**: customers received partially personalized messages that may be **wrong, duplicated, or inappropriate**. Support volume spikes; legal and PR get involved; leadership asks for **root cause**, **immediate containment**, and **guardrails** so “AI speed” never again translates into **irreversible blast radius**.

This scenario is about **classifying tool risk**, **human approvals**, **rate limits**, **dry-run defaults**, and **operational kill switches**—not about improving prompt wording alone.

**Interviewer:** How do you prevent this class of bug?

**You:** **Irreversible actions without confirmation.** Need: (1) **action classification** — reversible vs irreversible tools; (2) **confirmation gates** — thresholds for approval; (3) **dry_run** in staging and first prod behaviour.

**Safety architecture**

```python
@tool(category='communication', reversible=False,
      requires_confirmation=True, rate_limit=50)
async def send_email(to: str, subject: str, body: str):
    ...

class AgentHarness:
    async def execute_tool(self, action: ToolCall):
        tool = self.registry.get(action.name)
        count = self.counter.get(action.name)
        if count > tool.rate_limit:
            raise RateLimitExceeded(f'{action.name}: {count} calls, limit {tool.rate_limit}')
        if not tool.reversible and not self.dry_run:
            confirmed = await self.confirm(action)
            if not confirmed:
                raise ActionRejected(action)
        if self.dry_run:
            self.audit_log.append({'action': action, 'status': 'DRY_RUN_SKIPPED'})
            return {'status': 'dry_run', 'would_have_called': action}
        return await tool.execute(**action.args)
```

**Additional safeguards:** blast-radius caps (e.g. max 10 emails/run); recall APIs within ~30 s; anomaly detection on tool volume; **canary** bulk ops on 1% before full send.

**Interview tip:** Naming **human-in-the-loop for irreversible actions** is often the decisive answer.

---

### Scenario 7 — Agent loops endlessly on coding task *(AI Dev Tools — Medium)*

**Situation**

You are building an **AI coding agent** inside an IDE or CI bot. A user triggers **“fix all failing tests”** on a large repo. There are **47 failing tests** spanning multiple modules—not one trivial typo. The agent follows a generic **ReAct-style loop**: read files, patch, run tests, read logs, repeat.

Because there is **no explicit plan or progress metric**, the agent keeps **re-editing the same hot files**, toggling fixes that break other tests, or chasing misleading stack traces. Observability shows **~200 tool calls** over a long session; latency and cost balloon, and eventually the **context window fills** with noisy logs. From the user’s perspective the agent looks **stuck** even though it appears “busy.”

The interviewer wants to hear how you introduce **hierarchical task decomposition**, **per-subgoal budgets**, **detectors for lack of progress**, and **escalation**—the same structures human engineering leads use for incidents.

**Interviewer:** Root cause and architectural fix?

**You:** No **progress model**; no **subgoal decomposition**. Fix: **plan** (group failures), **progress monitor**, **per-subgoal step budget**.

```python
class CodingAgentHarness:
    def run(self, task: str):
        failures = self.run_tests()
        groups = self.llm.group_failures(failures)
        for group in groups:
            self.fix_group(group, budget=15)

    def fix_group(self, group: FailureGroup, budget: int):
        for step in range(budget):
            action = self.llm.next_action(group, step)
            self.execute(action)
            remaining = self.run_tests(group.test_ids)
            if len(remaining) >= len(group.failures) and step > 3:
                self.escalate(group, f'Stuck at {len(remaining)} failures after {step} steps')
                return
            group.failures = remaining
            if not remaining:
                break
```

**Insight:** Structured progress monitoring separates toy agents from production coding assistants.

---

### Scenario 8 — Research agent returns contradictory facts *(AI Research / Media — Medium)*

**Situation**

You operate a **research copilot** used by analysts or journalists. It combines **live web search**, **news APIs**, and **internal PDF / wiki RAG**. Different sources legitimately disagree: preprints vs later retractions, rumor-stage reporting vs official filings, or two vendors claiming incompatible market-share stats.

The failure mode here is subtle: the model **smoothly narrates** a single story and picks **one side arbitrarily**, or worse, **blends** incompatible claims so the answer reads coherent but is **internally inconsistent**. Users trust the voice of authority and may not notice until someone fact-checks externally—creating **reputational risk**.

You need a design that **surfaces disagreement** instead of hiding it: detecting contradiction, preserving provenance, and presenting **confidence and source quality** transparently.

**Interviewer:** Design for source conflict?

**You:** **Conflict detection** + **transparent presentation**, not naive merge. Pipeline: cluster facts by claim → **NLI** across clusters → if contradiction, use **debate-style** prompt vs synthesis.

```python
from enum import Enum

class Relation(Enum):
    ENTAILS = 'entails'
    CONTRADICTS = 'contradicts'
    NEUTRAL = 'neutral'

def detect_conflicts(claims: list[Claim]) -> list[ConflictPair]:
    conflicts = []
    for i, a in enumerate(claims):
        for b in claims[i+1:]:
            rel = nli_model.classify(a.text, b.text)
            if rel == Relation.CONTRADICTS:
                conflicts.append(ConflictPair(a, b))
    return conflicts

if conflicts:
    prompt = CONFLICT_PROMPT.format(...)
else:
    prompt = SYNTHESIS_PROMPT.format(facts=retrieved_facts)
```

**Source authority ranking:** peer-reviewed > government > major news > blog; surface authority differential when conflicts exist.

---

