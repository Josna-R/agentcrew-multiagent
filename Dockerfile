FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# AgentCrew runs fully offline by default (FakeLLM + the embedded, offline
# WebSearchTool knowledge base), so no API keys or network access are
# required for this default entrypoint.
ENTRYPOINT ["python", "demo.py"]
