# AgentCrew Architecture

AgentCrew is a LangGraph multi-agent research & writing pipeline with **real
parallel fan-out**: a Planner decomposes a topic, a dynamic set of
Researcher branches investigate each subtopic *concurrently*, their findings
are merged, a Writer/Critic loop refines a report within a bounded number of
revisions, and a Fact-Checker verifies the final draft is actually grounded
in the research. Every piece of this runs fully offline by default.

## Graph shape

```mermaid
flowchart TD
    START([START]) --> Planner["Planner<br/>decomposes topic into N subtopics"]
    Planner -- "Send(\"researcher\", ...) x N<br/>(dynamic fan-out)" --> R1["Researcher<br/>subtopic 1"]
    Planner -- "Send(\"researcher\", ...)" --> R2["Researcher<br/>subtopic 2"]
    Planner -- "Send(\"researcher\", ...)" --> R3["Researcher<br/>subtopic N"]
    R1 --> Agg["aggregate_research<br/>(fan-in)"]
    R2 --> Agg
    R3 --> Agg
    Agg --> Writer["Writer<br/>drafts / revises report"]
    Writer --> Critic["Critic<br/>reviews structure"]
    Critic -- "not approved AND\nrevision_count < max_revisions" --> Writer
    Critic -- "approved OR\nrevision budget exhausted" --> FactChecker["Fact-Checker<br/>scores groundedness"]
    FactChecker --> END([END])

    classDef parallel fill:#1f6feb,color:#fff,stroke:#1f6feb;
    class R1,R2,R3 parallel;
```

Each box is a LangGraph node (a plain Python function `state -> partial
update`); the arrows out of Critic are a single conditional edge
(`route_after_critic`), and the arrows out of Planner are a single
conditional edge (`_route_to_researchers`) whose path function returns a
*list* of `Send` objects rather than a fixed set of named edges.

## Why `Send`-based dynamic fan-out instead of hardcoded parallel branches

A naive "parallel" multi-agent demo hardcodes three branches in the graph
definition -- `researcher_1`, `researcher_2`, `researcher_3` -- each wired to
a fixed edge. That only works for exactly the number of subtopics the author
anticipated, and it is not actually driven by what the Planner decided; it's
a fixed topology wearing a parallel costume.

AgentCrew instead uses LangGraph's `Send` API:

```python
def _route_to_researchers(state: CrewState) -> List[Send]:
    return [
        Send("researcher", {"subtopic": subtopic, "topic": state["topic"]})
        for subtopic in state["subtopics"]
    ]
```

`_route_to_researchers` is registered as the path function of a single
conditional edge out of `planner`. At runtime, LangGraph calls it with the
current state, gets back a *list* of `Send` objects -- one per subtopic --
and schedules exactly that many invocations of the single `researcher` node,
all in the same super-step, truly concurrently. If the Planner decomposed
the topic into 3 subtopics, 3 branches run; if a different Planner (or a
smarter real-LLM backend) decomposed it into 5, 5 branches would run, with
*zero changes to `build_graph`*. The degree of parallelism is data, not
topology -- which is exactly the property a "real" parallel multi-agent
system needs and a sequential-pretending-to-be-parallel demo doesn't have.

Each `Send` also carries its own isolated input (`{"subtopic": ..., "topic":
...}`), so every researcher branch sees only its own subtopic via
`state["subtopic"]`, never the others' -- the branches are genuinely
independent, not just separate turns of one shared loop.

## Why `operator.add` reducers are required for concurrent writes

When N researcher branches run in the same super-step, each one returns its
own partial state update, e.g. `{"subtopic_findings": [finding]}`. LangGraph
has to merge N concurrent writes to `subtopic_findings` into one consistent
value before the next node (`aggregate_research`) runs. Without a reducer,
LangGraph's default behavior for a state key written by more than one branch
in the same step is to raise `InvalidUpdateError` -- "at key
'subtopic_findings': can receive only one value per step" -- because it has
no way to know whether two concurrent writes should overwrite each other or
combine.

`CrewState` declares:

```python
subtopic_findings: Annotated[List[SubtopicFinding], operator.add]
trace: Annotated[List[str], operator.add]
usage_log: Annotated[List[dict], operator.add]
```

`Annotated[..., operator.add]` tells LangGraph exactly how to merge
concurrent writes to that key: concatenate the lists with `+`. Each
researcher branch returns a *one-element* list for `subtopic_findings`
(never the whole accumulated list), and the reducer is what turns N
single-element lists arriving concurrently into one N-element list by the
time `aggregate_research` runs. The same trick is used for `trace` and
`usage_log` so every branch's log lines and usage records are preserved
rather than clobbering each other -- this is the mechanism, not a
workaround, for safely merging concurrent agent output in LangGraph.

## The bounded-revision design

The Writer/Critic loop is a cycle in the graph (`critic -> writer -> critic
-> ...`), and any cycle in a state machine needs an explicit, structural
guarantee that it terminates -- "the critic will eventually approve" is not
a guarantee, it's a hope, and a hope is not something you want to depend on
in production. `route_after_critic` encodes the real guarantee:

```python
def route_after_critic(state: CrewState) -> Literal["revise", "fact_check"]:
    approved = state.get("approved", False)
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)
    if not approved and revision_count < max_revisions:
        return "revise"
    return "fact_check"
```

`revision_count` is incremented by the critic node on *every* pass,
regardless of verdict, so it is a monotonically increasing counter that is
completely independent of whatever the critic's judgment happens to be. The
loop takes the `"revise"` edge only while both conditions hold; once
`revision_count` reaches `max_revisions`, the function always returns
`"fact_check"`, even if the draft is still not approved. This means the loop
provably terminates within `max_revisions + 1` critic calls no matter how a
(fake or real) critic behaves -- including a pathological critic that never
approves, which is exactly what `tests/test_graph.py` exercises directly
(`test_max_revisions_cap_terminates_into_fact_check`).

## Why fact-checking is a separate step from the critic

The critic and the fact-checker look at the same draft but check for
*different failure modes*, and conflating them would hide bugs:

- **The critic** evaluates *structure and completeness* -- does the report
  have a clear summary and conclusion, is it reasonably organized, does it
  look like a finished piece of writing. A critic can approve a
  beautifully-structured report that confidently states something no source
  ever said.
- **The fact-checker** evaluates *groundedness* -- for each claim-bearing
  sentence in the draft, is there supporting content in `research_notes`
  (the real output of the researcher branches), independent of how well the
  sentence is written. `make_fact_checker` computes a word-overlap score per
  sentence and returns a `FactCheckResult` with a `groundedness_score` and
  the specific `flagged_sentences` that aren't supported.

A single combined "critic" that both judges prose quality and verifies
grounding conflates "this reads well" with "this is true," and a model
that's good at one is not automatically good at the other. Keeping them as
separate nodes means each one has one job, the trace clearly attributes
which failure mode (if any) was caught where, and a future version could
swap in a stronger fact-checking model (e.g. a retrieval-grounded verifier)
without touching the critic at all.

## Cost / resilience design

- **Resilience**: real network calls (`OpenAILLM.generate`,
  `AnthropicLLM.generate`) are wrapped with
  `@retry_with_backoff(max_attempts=3)` from `agentcrew/resilience.py`,
  which retries only on the exception types it's told to retry on, with
  exponential backoff and jitter, and re-raises once attempts are exhausted.
  `sleep_fn` is injectable specifically so this can be unit-tested without
  real sleeps (see `tests/test_resilience.py`).
- **Determinism / zero-cost by default**: `FakeLLM` and the offline
  `WebSearchTool` mean the entire graph -- fan-out, aggregation, the
  revision loop, fact-checking -- can be built, tested, and demoed with no
  API keys, no network access, and no per-run dollar cost, while still
  exercising the real LangGraph execution engine (not a mock of it).
- **Cost observability**: every LLM call anywhere in the graph appends a
  usage record via `record_usage(role, input_tokens, output_tokens, model)`
  into `CrewState["usage_log"]` (merged via the same `operator.add` pattern
  used for concurrent researcher writes). `aggregate_usage` then summarizes
  total calls/tokens/estimated cost, broken down by role, so switching from
  `FakeLLM` to a real backend in production gets cost visibility "for free"
  with no code changes to the agents themselves.
- **Resumability**: `run_crew` accepts a `checkpointer` and `thread_id`.
  LangGraph's checkpointer (`MemorySaver` in-process, `SqliteSaver` on disk
  for the CLI) persists state after every super-step, keyed by
  `thread_id`. If a run fails partway through (a transient API error that
  exhausted its retries, a process crash), calling `run_crew` again with the
  same `thread_id` resumes from the last completed super-step instead of
  re-running the whole pipeline from the Planner -- see
  `tests/test_graph.py::TestCheckpointerResume` for a test that forces a
  mid-run failure and proves the Planner is not re-invoked on resume.
