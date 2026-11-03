"""AgentCrew: a LangGraph multi-agent research & writing crew.

AgentCrew demonstrates a real, parallel fan-out multi-agent architecture built
with LangGraph: a Planner decomposes a topic into subtopics, a dynamic set of
Researcher branches investigate each subtopic concurrently (via LangGraph's
``Send`` API), their findings are aggregated, a Writer/Critic loop refines a
report within a bounded number of revisions, and a Fact-Checker verifies the
final draft is grounded in the research notes.

The whole pipeline runs fully offline by default using ``FakeLLM`` and an
offline ``WebSearchTool`` backed by a small embedded knowledge base, so the
project can be cloned and run with zero API keys and zero network access.
Optional real backends (OpenAI, Anthropic) are available for production use.
"""

__version__ = "1.0.0"
