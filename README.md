# AgentCrew

[![CI](https://github.com/Josna-R/agentcrew-multiagent/actions/workflows/ci.yml/badge.svg)](https://github.com/Josna-R/agentcrew-multiagent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1f6feb)
![Offline by default](https://img.shields.io/badge/offline-by%20default-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**AgentCrew** is a LangGraph multi-agent research & writing crew with **real
parallel fan-out** -- not a sequential pipeline dressed up as "parallel."
A Planner decomposes a topic into subtopics, a dynamic set of Researcher
agents investigate each subtopic *concurrently* via LangGraph's `Send` API,
their findings are merged, a Writer/Critic loop refines a report within a
bounded number of revisions, and a Fact-Checker verifies the final draft is
actually grounded in the research -- not just well-written.

It runs **fully offline by default** (a deterministic `FakeLLM` plus an
offline knowledge-base-backed search tool), so you can clone it and run it
with zero API keys and zero network access. Optional OpenAI/Anthropic
backends are available as a drop-in swap for production use.

Built by [Josna Deepa Rayana](https://github.com/Josna-R) as a portfolio
project demonstrating production-minded multi-agent system design: real
concurrency, bounded loops, separated failure modes, cost observability, and
resumable runs.

## Why this exists

A lot of "multi-agent" demos run agents one after another and call it
parallel. AgentCrew is built around the parts of LangGraph that make
concurrency and reliability *real*:

- **Dynamic parallel fan-out** via `Send` -- the number of concurrent
  researcher branches comes from how many subtopics the Planner produced,
  not a hardcoded graph shape.
- **`operator.add` reducers** -- the only reason N concurrent branches can
  all write into shared state without LangGraph raising a conflicting-write
  error.
- **A provably-bounded revision loop** -- the Writer/Critic cycle always
  terminates within `max_revisions + 1` passes, even against a critic that
  never approves.
- **Fact-checking as its own step** -- the critic checks structure, the
  fact-checker checks that claims are grounded in the research. Different
  failure modes, different nodes.
- **Resumable runs** -- checkpointed state means a run that fails partway
  through can pick back up without starting over.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design
rationale and a diagram of the graph.

## Get the code

```bash
git clone https://github.com/Josna-R/agentcrew-multiagent.git
cd agentcrew-multiagent
```

## Quickstart

```bash
# 1. Clone and set up a virtual environment
git clone https://github.com/Josna-R/agentcrew-multiagent.git
cd agentcrew-multiagent
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the test suite (65 tests, fully offline, no API keys needed)
pytest tests/ -v

# 4. Run the demo: full trace of every agent's output, offline
python demo.py
python demo.py "The future of quantum computing"   # or any topic you like

# 5. Or use the CLI
python -m agentcrew.cli "Renewable energy trends"
python -m agentcrew.cli "Renewable energy trends" --max-revisions 3 --output json
```

### Resuming a run

The CLI persists checkpointed state to a local SQLite file
(`.agentcrew_checkpoints.sqlite`) keyed by `--thread-id`, so an interrupted
or failed run can pick back up instead of starting over:

```bash
python -m agentcrew.cli "Space exploration" --thread-id my-run-1
# ...if that run fails partway through (e.g. a transient API error with a
# real LLM backend after retries are exhausted)...
python -m agentcrew.cli "Space exploration" --resume --thread-id my-run-1
```

### Docker

```bash
docker build -t agentcrew .
docker run --rm agentcrew "The impact of AI on scientific research"
```

The Docker image's entrypoint runs `demo.py`, which is fully offline, so no
environment variables or network access are required.

## Using a real LLM backend (optional)

AgentCrew defaults to `FakeLLM`, which is deterministic and free. To use a
real backend instead:

```python
from agentcrew.graph import run_crew
from agentcrew.llm import OpenAILLM  # or AnthropicLLM

result = run_crew(
    topic="The future of renewable energy",
    llm=OpenAILLM(model="gpt-4o-mini", api_key="sk-..."),
)
```

Both `OpenAILLM` and `AnthropicLLM` wrap their network call with
`@retry_with_backoff(max_attempts=3)` (exponential backoff with jitter), and
their SDKs (`openai`, `anthropic`) are imported lazily -- they're only
required if you actually construct one of these classes.

## Project structure

```
agentcrew-multiagent/
├── agentcrew/
│   ├── tools.py        # WebSearchTool (offline KB) + CalculatorTool (safe ast eval)
│   ├── state.py        # CrewState TypedDict, SubtopicFinding, FactCheckResult
│   ├── usage.py         # token estimation, cost tracking, usage aggregation
│   ├── resilience.py    # retry_with_backoff: exponential backoff + jitter
│   ├── llm.py            # LLM ABC, FakeLLM, OpenAILLM, AnthropicLLM
│   ├── agents.py          # node functions: planner, researcher, writer, critic, fact_checker
│   ├── graph.py            # Send-based fan-out wiring, build_graph, run_crew
│   └── cli.py                # argparse CLI with --resume / --thread-id
├── demo.py            # runnable end-to-end trace, fully offline
├── tests/              # pytest suite (65 tests)
├── docs/
│   └── ARCHITECTURE.md  # design rationale + mermaid diagram
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml
```

## Tech stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** -- the graph
  execution engine: `StateGraph`, the `Send` API for dynamic fan-out,
  `MemorySaver`/`SqliteSaver` checkpointers for resumable runs.
- **Python's `ast` module** -- for `CalculatorTool`'s safe arithmetic
  evaluator (parses to an AST, only evaluates numeric/operator nodes --
  `__import__(...)`, `open(...)`, and friends are structurally impossible to
  reach).
- **pytest** -- the full test suite, including a real end-to-end run of the
  compiled graph (not mocked), with a fake `sleep_fn` so the resilience
  tests never actually sleep.
- **Zero required external services** -- `FakeLLM` and the offline
  `WebSearchTool` knowledge base mean the whole system is self-contained.

## Testing

```bash
pytest tests/ -v
```

The suite covers:

- `tools.py` -- `WebSearchTool` keyword-overlap ranking, and `CalculatorTool`
  safe evaluation *plus* explicit rejection of unsafe expressions like
  `__import__('os')` and `open('/etc/passwd').read()`.
- `usage.py` -- token estimation and usage/cost aggregation.
- `resilience.py` -- retry count, exponential backoff growth, the delay cap,
  and that excluded exception types are never retried -- all using a fake,
  non-sleeping `sleep_fn`.
- `agents.py` -- each node function in isolation (planner subtopic count,
  researcher's one-element return shape, aggregation, routing, fact-check
  flagging).
- `graph.py` -- the **real, compiled** LangGraph graph end-to-end: the
  parallel fan-out produces exactly one finding per subtopic, the
  writer/critic loop fires a revision and then approves, the
  `max_revisions` cap terminates the loop into the fact-checker instead of
  looping forever, and checkpointer save/resume works (including resuming
  after a simulated mid-run failure without re-running the planner).
- `cli.py` -- Markdown/JSON output rendering, argument parsing, and
  `--resume`/`--thread-id`.

## Roadmap

- [ ] Streaming output (token-by-token) from real LLM backends via
      LangGraph's streaming API.
- [ ] A pluggable, real `WebSearchTool` backend (e.g. an actual search API)
      behind the same `Tool` interface, so switching from offline demo mode
      to live research is a one-line change.
- [ ] A lightweight web UI (FastAPI + a small frontend) to kick off runs and
      watch the trace live instead of reading a CLI/demo transcript.
- [ ] Human-in-the-loop approval as an interrupt point before the
      fact-checker, using LangGraph's `interrupt`/`Command` support.
- [ ] Structured citations in the final report (mapping sentences back to
      specific `source_id`s) instead of per-subtopic source lists.

## License

MIT
