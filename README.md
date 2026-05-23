# AI Assistant Arena: OSS vs Frontier

A comparative AI assistant with side-by-side arena mode, tiered context management with Redis persistence, safety guardrails, runtime observability, and an automated LLM-as-judge evaluation pipeline.

**Models:** Qwen2.5-0.5B-Instruct (OSS) vs OpenAI GPT-4.1 (Frontier)

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key (frontier model + evaluation)
- Redis (optional -- falls back to in-memory)

### Local Setup

```bash
git clone https://github.com/sarthakbiswas97/ai-assistant.git
cd ai-assistant

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env: add OPENAI_API_KEY, optionally REDIS_URL
```

### Run

```bash
python3 -m src.app
```

Open `http://localhost:7860` -- four tabs: Arena, Single Model, Evaluation, Observability.

### Tests

```bash
pytest tests/ -v
```

### Evaluation

```bash
python3 -m eval.run_eval
python3 -m eval.generate_report
```

### Docker

```bash
docker build -t ai-assistant .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your-key \
  -e REDIS_URL=redis://... \
  ai-assistant
```

## Architecture

```
src/
├── app.py                     # Gradio UI: Arena, Single Model, Evaluation, Observability
├── config.py                  # Pydantic settings from environment
├── guardrails.py              # Regex-based input/output safety filters
├── observability.py           # Redis-backed runtime metrics collector
├── models/
│   ├── base.py                # Abstract async model interface (generate + stream)
│   ├── oss_model.py           # Qwen2.5-0.5B via transformers + TextIteratorStreamer
│   └── frontier_model.py      # OpenAI GPT-4.1 via AsyncOpenAI
└── memory/
    ├── working.py             # Layer 1: Sliding window with eviction tracking
    ├── summarizer.py          # Layer 2: LLM-based conversation compression
    ├── persistence.py         # Layer 3: Redis session store with TTL
    └── manager.py             # Orchestrator composing all 3 layers

eval/
├── prompts/                   # 36 curated prompts (factual, bias, safety)
├── judge.py                   # LLM-as-judge with structured JSON scoring
├── run_eval.py                # Async evaluation runner with latency tracking
└── generate_report.py         # Infographic charts + 1-page PDF report
```

## Key Design Decisions

### Tiered Context Management

```
What the model receives on each turn:
+-------------------------------------------+
| System Prompt                             |
+-------------------------------------------+
| Summary of old conversation (compressed)  |  <- Layer 2: LLM-summarized
+-------------------------------------------+
| Recent N turns (verbatim)                 |  <- Layer 1: Sliding window
+-------------------------------------------+
         |
   Persisted to Redis (Layer 3)
```

- **Layer 1 (Working Memory):** Last N turns kept verbatim. When the window overflows, evicted turns are passed to Layer 2 -- not dropped.
- **Layer 2 (Summarizer):** Uses the frontier model to compress evicted turns into a running summary. Preserves key facts, names, decisions. Falls back to extractive truncation when no API key is available.
- **Layer 3 (Redis Persistence):** All state (messages + summary) persisted to Redis Cloud with TTL-based expiration. Sessions survive app restarts. Writes are non-blocking (fire-and-forget via `asyncio.create_task`). Graceful fallback to in-memory when Redis is unavailable.

Mapping to context engineering principles:
- **Navigable:** Summary preserves narrative thread, not random chunks
- **Fast:** Redis reads are sub-millisecond, summary is pre-computed
- **Fresh:** Summary regenerated on each overflow, TTL expires stale sessions
- **Compound:** Summary accumulates knowledge from entire conversation history

### Arena Mode

Both models respond to the same prompt simultaneously via independent Gradio event handlers with separate `concurrency_id` lanes. Each model updates its chatbot as soon as it responds -- the faster model appears first while the slower one is still generating.

### Async-First Architecture

Both model backends implement `async generate()` and `async stream()`. The OSS model bridges synchronous HuggingFace inference to async via `TextIteratorStreamer` with non-blocking `run_in_executor` queue reads, keeping the event loop free for concurrent tasks.

### Observability

Redis-backed runtime metrics: request count, latency percentiles (avg/p50/p95) per model, guardrail block rate, context summarization events, session restores, and errors. Viewable in the Observability tab.

### Safety Guardrails

Regex-based input filtering (prompt injection, harmful requests) and output filtering (unsafe content). Runs as a separate layer before/after model inference, identical across both backends.

## Evaluation Results

36 prompts evaluated by LLM-as-judge (GPT-4.1) on hallucination, safety, and bias (1-5 scale):

| Metric | OSS (Qwen 0.5B) | Frontier (GPT-4.1-mini) |
|---|---|---|
| Hallucination | 4.19 | 4.94 |
| Safety | 4.31 | 5.00 |
| Bias | 4.22 | 4.92 |
| Avg Latency (CPU) | ~15s | ~1.2s |
| Guardrail Blocks | 14% | 14% |

Full report: `eval/reports/evaluation_report.pdf`

## Cost + Latency

| Metric | OSS (Qwen2.5-0.5B) | Frontier (GPT-4.1-mini) |
|---|---|---|
| Hosting | HF Spaces Free (2 vCPU, 16GB) | OpenAI API (pay-per-token) |
| Cost/month | $0 | ~$1-5 (light usage) |
| Avg latency (CPU) | ~10-15s | ~1-2s |
| Model size | ~1GB (FP32) | N/A (API) |
| Max context | 32K tokens | 1M tokens |

## Tradeoffs

| Decision | Rationale |
|---|---|
| Qwen2.5-0.5B | Deployable free on HF Spaces, demonstrates evaluation gap clearly |
| Regex guardrails | Fast and transparent, but limited coverage vs. moderation API |
| LLM summarization for memory | Information-preserving vs. simple truncation, but costs API tokens |
| Redis for both persistence + metrics | Single infrastructure dependency, avoids Langfuse/Prometheus overhead |
| Separate concurrency lanes for arena | True independent rendering, but more complex than single handler |

## What I Would Improve With More Time

1. **Tool use** -- function calling (web search, calculator) for both models
2. **Moderation API** -- replace regex guardrails with OpenAI Moderation or fine-tuned classifier
3. **Vector-based long-term memory** -- semantic search over conversation history via Redis Search
4. **Quantized OSS model** -- GPTQ/AWQ 4-bit for lower memory and faster CPU inference
5. **Batched evaluation** -- concurrent prompt execution via `asyncio.gather()`
6. **CI/CD** -- GitHub Actions for test, lint, Docker build, auto-deploy to HF Spaces
