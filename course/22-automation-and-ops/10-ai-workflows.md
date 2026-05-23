# AI workflows in production

LLMs and embedding models have become a standard part of the operational engineer's toolkit — for parsing free-text documents, summarising logs, classifying unstructured data, generating reports, retrieving relevant context for human reviewers. This chapter is the working patterns for using them in production: batched inference, retrieval-augmented generation (RAG), cost control, and the safety practices that keep AI workflows boring.

## When LLMs earn their keep

For ops:

- **Parsing unstructured text** — extract structured fields from emails, PDFs, news articles.
- **Triage and routing** — classify support tickets, alerts, log lines into buckets.
- **Summarisation** — daily news digests, incident summaries, large-document compression.
- **Code generation for one-shot scripts** — when "write me a quick parser for this format" is faster than implementing.
- **Semantic search** — find the relevant document among thousands by meaning, not keyword.

When NOT to use them:

- Tasks with a closed-form answer (regex, SQL query, deterministic logic).
- High-frequency operations where latency matters.
- Anything where wrong answers are unacceptable without human review.

## The Anthropic SDK pattern

```python
import anthropic
import os


client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def summarise(text: str, max_tokens: int = 500) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": f"Summarise this concisely:\n\n{text}"},
        ],
    )
    return response.content[0].text
```

Three things every production caller should set:

1. **Explicit model name** — pinned, not "latest." Otherwise model upgrades silently change behaviour.
2. **`max_tokens`** — hard cap on output. Prevents runaway responses.
3. **Timeout** — `anthropic.Anthropic(timeout=30.0)`. Default is generous; specify.

## Prompt caching for cost control

For prompts with a large fixed context (a long instruction, a big document) but a small variable part:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=500,
    system=[
        {
            "type": "text",
            "text": LONG_INSTRUCTION_BLOCK,                # 5000 tokens of context
            "cache_control": {"type": "ephemeral"},
        }
    ],
    messages=[{"role": "user", "content": user_question}],
)
```

The cached prefix is processed once and reused for 5 minutes (or until context changes). For a workflow that runs the same instruction with 100 different inputs, prompt caching cuts input costs by ~90%.

## Batch processing

For non-urgent, large-volume LLM jobs (overnight summarisation of 10,000 articles), the **Message Batches API** offers ~50% cost discount and async processing:

```python
import anthropic


client = anthropic.Anthropic()

batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": f"req-{i}",
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": f"Summarise:\n{text}"}],
            },
        }
        for i, text in enumerate(many_documents)
    ]
)

# Poll for completion
batch_id = batch.id
while True:
    status = client.messages.batches.retrieve(batch_id)
    if status.processing_status == "ended":
        break
    time.sleep(60)

results = client.messages.batches.results(batch_id)
for result in results:
    print(result.custom_id, result.result.message.content[0].text)
```

Use for any non-time-sensitive bulk work. 24-hour SLA; usually completes in minutes.

## Structured output

Don't parse free-text responses; ask for JSON:

```python
SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "key_entities": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["sentiment", "key_entities", "summary"],
}


def analyse_news(article: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        tools=[
            {
                "name": "report",
                "description": "Output the structured analysis.",
                "input_schema": SCHEMA,
            }
        ],
        tool_choice={"type": "tool", "name": "report"},
        messages=[
            {"role": "user", "content": f"Analyse this news article:\n\n{article}"}
        ],
    )
    # Extract the tool_use block
    for block in response.content:
        if block.type == "tool_use" and block.name == "report":
            return block.input
    raise RuntimeError("no structured output")
```

`tool_choice` forces the model to produce a structured response matching the schema. Far more reliable than parsing free-text JSON.

## Embeddings + vector stores

For semantic search: convert text to embeddings (vectors), store in a vector database, search by similarity.

```python
# Anthropic doesn't ship embeddings as of writing — common third-party options:
# - voyageai (Anthropic-recommended)
# - sentence-transformers (local, free)
# - OpenAI text-embedding-3 (good for many languages)

from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer("BAAI/bge-small-en-v1.5")
documents = ["doc 1 text", "doc 2 text", ...]
embeddings = model.encode(documents, normalize_embeddings=True)  # (N, dim)


def search(query: str, top_k: int = 5):
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_emb               # cosine since normalised
    top = np.argsort(scores)[::-1][:top_k]
    return [(documents[i], float(scores[i])) for i in top]
```

For < 100k documents, NumPy on disk is fine. For more, use a vector DB:

- **Qdrant**, **Weaviate**, **Pinecone**, **Chroma** — purpose-built.
- **Postgres + `pgvector`** — if you already have Postgres.
- **DuckDB + `vss`** — for analytical workloads.

## RAG — retrieval-augmented generation

Combine retrieval (find relevant docs) + generation (LLM answers using those docs):

```python
def rag_answer(question: str, top_k: int = 5) -> str:
    # 1. Retrieve relevant docs by embedding similarity
    contexts = search(question, top_k=top_k)
    context_text = "\n\n---\n\n".join(d for d, _ in contexts)

    # 2. Generate answer grounded in those docs
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system="Answer the question using only the provided context. If the context doesn't contain the answer, say so.",
        messages=[
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
        ],
    )
    return response.content[0].text
```

That's vanilla RAG in 10 lines. Production refinements:

- **Chunk documents** before embedding (~500 tokens each); index per chunk.
- **Re-rank** retrieved chunks with a cross-encoder for better top-K.
- **Cite sources** — include doc IDs in the response so users can verify.
- **Cache common queries** — many "how do I X?" questions repeat.

## Cost control

Three patterns that keep AI bills sane:

### 1. Tier by capability

Use the smallest model that works. Claude Haiku for simple classification, Sonnet for most reasoning, Opus only when needed.

```python
def classify(text: str) -> str:
    """Three-way classifier; Haiku is fine."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": f"Classify (positive/neutral/negative): {text}"}],
    )
    return response.content[0].text.strip()
```

### 2. Cache aggressively

In-memory or disk cache for prompt → response. Same input ⇒ same output ⇒ no API call.

```python
import hashlib
import json
from pathlib import Path


def cached_llm_call(prompt: str, model: str, max_tokens: int) -> str:
    key = hashlib.sha256(json.dumps({"p": prompt, "m": model, "t": max_tokens}).encode()).hexdigest()
    cache_path = Path(".cache/llm") / f"{key}.txt"
    if cache_path.exists():
        return cache_path.read_text()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text
```

For prompts you'll re-run thousands of times in development.

### 3. Set hard budgets

For batch jobs, count tokens before submitting:

```python
TOKEN_BUDGET = 100_000
estimated_tokens = sum(len(t) // 4 for t in documents) + 500 * len(documents)
if estimated_tokens > TOKEN_BUDGET:
    raise RuntimeError(f"would consume {estimated_tokens} tokens > budget {TOKEN_BUDGET}")
```

Use the Anthropic billing dashboard to set alerts at 50/80/100% of monthly budget.

## A worked example: nightly news triage

```python
from datetime import datetime, timezone
import logging


def triage_overnight_news(news_items: list[dict]) -> list[dict]:
    """For each item: classify, summarise, and tag relevant tickers."""
    results = []
    for item in news_items:
        try:
            analysis = analyse_news(item["body"])           # structured output, defined above
            results.append({
                "id": item["id"],
                "headline": item["headline"],
                "sentiment": analysis["sentiment"],
                "summary": analysis["summary"],
                "entities": analysis["key_entities"],
                "asof": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            logging.exception("failed on item %s", item["id"])
    return results


if __name__ == "__main__":
    news = fetch_overnight_news()
    triaged = triage_overnight_news(news)
    write_to_db(triaged)
```

A simple, idempotent, observable nightly job. 50 articles in ~1 minute, costs pennies.

## Pitfalls

!!! warning "Trusting LLM output for high-stakes decisions"
    LLMs hallucinate. For trading decisions, code that modifies databases, or anything irreversible — humans must review.

!!! warning "Prompt injection"
    If your prompt includes user-controlled text, the user can override your instructions. Defend with: clear separation of instructions vs data, strict structured output, post-validation of model output.

!!! warning "Drift over model versions"
    Pin the model version. A "minor update" sometimes shifts behaviour enough to break a parser downstream.

!!! warning "Privacy and regulation"
    Sending PII / confidential data to an LLM API may violate your data-handling policy. Check before piping logs containing customer data to Claude.

## Bottom line

For production AI workflows:

- **Pin models**, set `max_tokens` and timeouts.
- **Structured output** via tools, not free-text parsing.
- **Cache aggressively**, especially in dev.
- **Batch API** for large non-urgent jobs (~50% cheaper).
- **Prompt caching** for static-instruction workloads.
- **Tier by capability** — Haiku for cheap, Sonnet for default, Opus only when needed.
- **Human-in-the-loop** for anything irreversible.

Continue to **[Backend systems with FastAPI](11-backend-systems.md)**.
