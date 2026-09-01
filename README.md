# AgentCrew — Multi-Agent Research & Report System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Tests](https://img.shields.io/badge/tests-3%2F3%20passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-lightgrey) ![Offline](https://img.shields.io/badge/runs-offline%2C%20no%20API%20key-success)

A LangGraph-orchestrated crew of three agents — **Researcher → Writer →
Critic** — that collaborate to produce a structured report on any topic,
with the Critic able to send work back to the Writer for revision (bounded
by a revision cap so the graph always terminates). Runs fully offline out
of the box via a deterministic `FakeLLM`; point it at OpenAI or Anthropic
for production-quality output with a one-line env var change.

**TL;DR:** run `python demo.py "your topic"` and watch three agents hand
work off to each other in real time, with a printed trace of who did what
and a critic that can send the draft back for revision. See
[Get the code](#get-the-code) below.

## What it does

Give it a topic. The **Researcher** gathers key findings, the **Writer**
turns them into a structured report (Overview / Key Findings /
Recommendation), and the **Critic** either approves it or sends it back to
the Writer with specific feedback — up to a configurable revision cap, so
the graph is guaranteed to finish. Every step is logged to a human-readable
trace so you can see exactly what each agent contributed.

## Why this project

Agentic, multi-step AI workflows (not single-shot prompting) are the
fastest-growing category of GenAI roles right now. This project demonstrates
stateful multi-agent orchestration with LangGraph — the same pattern used
in production systems for automated research, report generation, and
compliance workflows (e.g. the author's CrewAI/LangGraph work automating
market-risk reporting).

## Architecture

```
        ┌─────────────┐
        │  Researcher  │  gathers key findings on the topic
        └──────┬───────┘
               ▼
        ┌─────────────┐
   ┌───▶│    Writer    │  drafts a structured report (Overview / Findings / Recommendation)
   │    └──────┬───────┘
   │           ▼
   │    ┌─────────────┐
   │    │    Critic    │  APPROVEs or REVISEs with specific feedback
   │    └──────┬───────┘
   │           │
   └── revise ─┴─ done ──▶ END
       (capped at `max_revisions`, default 2, to guarantee termination)
```

State (`CrewState`) — topic, research notes, draft, critique, approval flag,
revision count, and a human-readable trace — flows through every node as a
single typed dict, LangGraph's standard pattern for stateful graphs.

## Get the code

```bash
git clone https://github.com/Josna-R/agentcrew-multiagent.git
cd agentcrew-multiagent
```

## Quickstart

```bash
pip install -r requirements.txt

# Run the test suite (fully offline) — includes a stub-LLM test that proves
# the critic->writer revision loop actually re-enters the writer node, and
# a second test proving the max_revisions cap terminates the graph instead
# of looping forever.
pytest tests/ -v

# Run the demo end-to-end (offline FakeLLM)
python demo.py "Agentic AI systems for financial compliance"
```

Expect printed output showing the agent trace (`[researcher] gathered
notes...`, `[writer] drafted report...`, `[critic] verdict: APPROVE`),
followed by the final generated report and a revision count.

## Project structure

```
agentcrew-multiagent/
├── agentcrew/
│   ├── state.py    # CrewState — the typed dict passed between every node
│   ├── agents.py    # Researcher / Writer / Critic node functions + routing logic
│   ├── llm.py         # LLM interface + FakeLLM / OpenAI / Anthropic implementations
│   └── graph.py       # builds and compiles the LangGraph StateGraph
├── demo.py             # CLI entry point
├── tests/test_graph.py # graph + revision-loop + termination-cap tests
└── requirements.txt
```

## Upgrading to a real model

```bash
pip install openai
export OPENAI_API_KEY=sk-...
export AGENTCREW_LLM=openai
python demo.py "Your topic here"
```

## Tech stack

Python · LangGraph · typed state graphs · conditional edges · pluggable
OpenAI / Anthropic backends

## Possible extensions

- Add a fourth "Fact-Checker" agent that verifies claims against a RAG
  index (pairs naturally with the DocuMind RAG project in this portfolio).
- Persist trace + final report to a database or send to Slack/email.
- Wrap in a FastAPI endpoint for on-demand report generation.
- Add LangSmith tracing for production observability.

## Author

Built by [Josna Deepa Rayana](https://github.com/Josna-R), AI/ML Engineer &
Data Scientist. 
