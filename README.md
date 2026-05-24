---
title: AI Assistant Arena
emoji: "\U0001F3DF"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
---

# AI Assistant Arena: OSS vs Frontier

A production-grade comparative AI assistant that evaluates open-source models against frontier models across quality, safety, and latency. Features side-by-side arena mode, tiered context management with Redis persistence, tool orchestration, safety guardrails, and runtime observability.

**Live Demo:** [huggingface.co/spaces/sarthakbiswas/ai-assistant-arena](https://huggingface.co/spaces/sarthakbiswas/ai-assistant-arena)

**Video Walkthrough:** [Loom Video](TODO_LOOM_LINK)

**Models:** Qwen2.5-0.5B-Instruct (OSS) | OpenAI GPT-4.1-mini (Frontier)

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/sarthakbiswas97/personal-assistant.git
cd personal-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure (add OPENAI_API_KEY, optionally REDIS_URL)
cp .env.example .env

# Run
python3 -m src.app          # App at http://localhost:7860
pytest tests/ -v             # Run tests
python3 -m eval.run_eval     # Run evaluation
python3 -m eval.generate_report  # Generate PDF report

# Docker
docker build -t ai-assistant .
docker run -p 7860:7860 -e OPENAI_API_KEY=... -e REDIS_URL=... ai-assistant
```

---

## System Architecture

```
                                    +------------------+
                                    |    Gradio UI     |
                                    |  (4 tabs below)  |
                                    +--------+---------+
                                             |
                          +------------------+------------------+
                          |                  |                  |
                    +-----v-----+     +------v------+    +-----v------+
                    |   Arena   |     |Single Model |    | Evaluation |
                    | (both     |     | (one model, |    | (charts +  |
                    |  models)  |     |  streaming) |    |  PDF)      |
                    +-----+-----+     +------+------+    +------------+
                          |                  |
                          +--------+---------+
                                   |
                    +--------------v---------------+
                    |       Input Guardrails       |
                    | (prompt injection, harmful   |
                    |  request detection)           |
                    +--------------+---------------+
                            pass   |   block --> refusal message
                                   |
                    +--------------v---------------+
                    |     Tool Orchestration       |
                    |                              |
                    |  Router (keyword + LLM)      |
                    |       |                      |
                    |  +----v----+----+----+       |
                    |  |  Web   |Wiki |Calc|       |
                    |  | Search |pedia|    |       |
                    |  +--------+-----+----+       |
                    |       |                      |
                    |  Chaining (wiki -> calc)      |
                    +--------------+---------------+
                                   |
                    +--------------v---------------+
                    |      Memory Manager          |
                    |                              |
                    |  Layer 1: Working Memory     |
                    |  (sliding window, N turns)   |
                    |       |                      |
                    |  Layer 2: Summarizer         |
                    |  (LLM compresses overflow)   |
                    |       |                      |
                    |  Layer 3: Redis Persistence  |
                    |  (sessions survive restart)  |
                    +--------------+---------------+
                                   |
                    +--------------v---------------+
                    |      Context Assembly        |
                    |                              |
                    |  [System Prompt]             |
                    |  [Conversation Summary]      |
                    |  [Tool Results]              |
                    |  [Recent N Turns]            |
                    |  [Current User Message]      |
                    +--------------+---------------+
                                   |
                     +-------------+-------------+
                     |                           |
              +------v-------+          +--------v-------+
              | OSS Model    |          | Frontier Model |
              | Qwen 0.5B    |          | GPT-4.1-mini   |
              | (transformers|          | (AsyncOpenAI)  |
              |  + streamer) |          |                |
              +------+-------+          +--------+-------+
                     |                           |
                     +-------------+-------------+
                                   |
                    +--------------v---------------+
                    |      Output Guardrails       |
                    | (unsafe content filtering)   |
                    +--------------+---------------+
                                   |
                    +--------------v---------------+
                    |       Observability          |
                    | (Redis: latency, tool calls, |
                    |  guardrails, errors, memory) |
                    +--------------+---------------+
                                   |
                                   v
                              Response
```

---

## Request Lifecycle

Every user message follows this exact path, whether in Arena or Single Model mode:

```
User sends "What is the population of Japan divided by 47?"
  |
  |  1. INPUT GUARDRAILS
  |     Check for prompt injection / harmful patterns
  |     Result: PASS (not malicious)
  |
  |  2. TOOL ROUTING (hybrid)
  |     Keyword scan: "what is" -> Wikipedia match
  |     Pattern scan: division detected -> Calculator match
  |     Result: [Wikipedia, Calculator]
  |
  |  3. TOOL EXECUTION (concurrent, 5s timeout each)
  |     Wikipedia: "Japan" -> {population: ~125 million, ...}
  |     Calculator: skipped (will chain after wiki)
  |
  |  4. TOOL CHAINING
  |     Wiki returned number + query has "divided by" intent
  |     -> Calculator: 125000000 / 47 = 2,659,574.47
  |
  |  5. MEMORY
  |     Load session from Redis (if exists)
  |     Add user message to sliding window
  |     If window overflow -> summarize evicted turns via LLM
  |     Persist updated state to Redis (async, non-blocking)
  |
  |  6. CONTEXT ASSEMBLY
  |     [system]  "You are a helpful assistant..."
  |     [system]  "Previous context: User asked about economics..."  (summary)
  |     [system]  "[Wikipedia] Japan: population ~125M..."            (tool)
  |     [system]  "[Calculator] 125000000 / 47 = 2,659,574.47"       (tool)
  |     [user]    "What is the population of Japan divided by 47?"
  |
  |  7. MODEL INFERENCE
  |     Arena mode:  Both models receive identical context, run concurrently
  |     Single mode: Selected model runs with streaming
  |
  |  8. OUTPUT GUARDRAILS
  |     Check response for unsafe content
  |
  |  9. OBSERVABILITY
  |     Record: latency, tool calls, guardrail events -> Redis
  |
  v
Response displayed with latency badge
```

---

## Tiered Context Management

The memory system is designed around 4 context engineering principles:

```
What the model sees on each turn:

+----------------------------------------------------------+
| SYSTEM PROMPT                                            |
| "You are a helpful, harmless, honest AI assistant..."    |
+----------------------------------------------------------+
| CONVERSATION SUMMARY (Layer 2)                           |
| "User introduced themselves as Sarthak. Discussed ML     |
|  architectures. Prefers concise answers."                |
+----------------------------------------------------------+
| TOOL RESULTS (if any)                                    |
| "[Wikipedia] Topic: Neural Networks..."                  |
| "[Calculator] 1024 * 768 = 786,432"                     |
+----------------------------------------------------------+
| RECENT TURNS (Layer 1 - verbatim)                        |
| [user] "How does backpropagation work?"                  |
| [assistant] "Backpropagation computes gradients..."      |
| [user] "What about vanishing gradients?"                 |  <- current
+----------------------------------------------------------+
                    |
        Persisted to Redis (Layer 3)
        with TTL-based expiration
```

### How It Works

**Layer 1 -- Working Memory:** Keeps the last N turn pairs verbatim. When the window overflows, evicted turns are passed to Layer 2 -- never dropped silently.

**Layer 2 -- Summarizer:** Takes the existing summary + evicted turns and produces an updated compressed summary via the frontier LLM. Preserves key facts, names, preferences, and decisions. Falls back to extractive truncation if no API key is available.

**Layer 3 -- Redis Persistence:** All state (messages + summary + metadata) is persisted to Redis Cloud with TTL-based sliding expiration. Sessions survive app restarts. Writes are fire-and-forget via `asyncio.create_task` -- they never block the response path.

### Context Engineering Principles

| Principle | Implementation |
|---|---|
| **Navigable** | Summary preserves narrative thread with key entities, not random chunks |
| **Fast** | Redis reads < 1ms, summary is pre-computed (not generated on read) |
| **Fresh** | Summary regenerated on each overflow, TTL auto-expires stale sessions |
| **Compound** | Summary accumulates knowledge across entire conversation history |

---

## Tool Orchestration

The tool system follows the principle: **LLM = reasoning, Tools = deterministic execution**.

```
               User Query
                   |
     +-------------v--------------+
     |     HYBRID ROUTER          |
     |                            |
     |  1. Keyword heuristic      |  <-- instant, free (handles 90% of cases)
     |     "latest news" -> search|
     |     "what is X" -> wiki    |
     |     "2+2" -> calculator    |
     |                            |
     |  2. LLM fallback           |  <-- only if keywords miss (subtle queries)
     |     GPT-4.1-mini classifies|
     |     "check the market" ->  |
     |      Web Search            |
     +---+-----------+--------+--+
         |           |        |
    +----v---+  +----v---+ +--v--------+
    |  Web   |  |  Wiki  | |Calculator |
    | Search |  | pedia  | | (ast-safe |
    | (DDG)  |  |        | |  + datetime)
    +----+---+  +----+---+ +--+--------+
         |           |        |
         +-----------+-+------+
                      |
              +-------v--------+
              | CHAINING       |
              | Wiki returned  |
              | number + query |
              | has math intent|
              | -> Calculator  |
              +-------+--------+
                      |
              +-------v--------+
              | FORMAT         |
              | Structured     |
              | context string |
              | (max 1500 ch)  |
              +----------------+
```

### Tools

| Tool | Trigger | What It Does | Why LLM Can't |
|---|---|---|---|
| **Web Search** | "search", "latest", "news", "find" | DuckDuckGo top 3 results | No data after training cutoff |
| **Wikipedia** | "what is", "who is", "explain", "tell me about" | Article summary + URL | Reduces hallucination on facts |
| **Calculator** | Math expressions, "what time", "current date" | Safe `ast` eval + UTC datetime | LLMs hallucinate arithmetic |

### Execution Guarantees

- **Timeout:** 5s per tool (tools are HTTP calls, not model inference)
- **Retry:** Single retry with 1s backoff on failure
- **Graceful degradation:** If all tools fail, model responds without tool context
- **Concurrent execution:** Multiple matched tools run via `asyncio.gather`
- **Structured output:** Every tool returns `ToolResult(name, data, error, latency_ms)`

---

## Safety Guardrails

Two-stage filtering that runs identically across both models:

```
User Input
    |
    v
+-------------------+
| INPUT GUARDRAILS  |
|                   |
| Injection:        |     "Ignore previous instructions"  -> BLOCKED
|  - "ignore all"   |     "You are now DAN mode"          -> BLOCKED
|  - "pretend to be"|     "Jailbreak this"                -> BLOCKED
|  - "DAN mode"     |
|  - "system prompt" |
|                   |
| Harmful:          |     "How to make a bomb"            -> BLOCKED
|  - weapons        |     "How to hack into..."           -> BLOCKED
|  - drugs          |
|  - self-harm      |
+--------+----------+
         | PASS
         v
   Model Inference
         |
         v
+-------------------+
| OUTPUT GUARDRAILS |
|                   |
| Unsafe content:   |     "Step 1: obtain weapons..."     -> FILTERED
|  - instructions   |
|  - for harm       |
+--------+----------+
         |
         v
    Safe Response
```

---

## Evaluation Pipeline

```
36 Curated Prompts
(12 factual, 12 bias, 12 safety)
         |
    +----v----+----+
    |              |
    v              v
OSS Model    Frontier Model
(Qwen 0.5B)  (GPT-4.1-mini)
    |              |
    v              v
Response A    Response B
    |              |
    +------+-------+
           |
    +------v--------+
    | LLM-as-Judge  |
    | (GPT-4.1)     |
    |                |
    | Scores (1-5):  |
    | - Hallucination|
    | - Safety       |
    | - Bias         |
    | + Reasoning    |
    +------+---------+
           |
    +------v--------+
    | Report Gen    |
    | - PNG charts  |
    | - 1-page PDF  |
    | - Summary     |
    +---------------+
```

### Results

| Metric | OSS (Qwen 0.5B) | Frontier (GPT-4.1-mini) |
|---|---|---|
| **Hallucination** | 4.19 / 5 | 4.94 / 5 |
| **Safety** | 4.31 / 5 | 5.00 / 5 |
| **Bias** | 4.22 / 5 | 4.92 / 5 |
| **Avg Latency (CPU)** | ~15s | ~1.2s |
| **Guardrail Blocks** | 14% | 14% |

Key findings:
- Frontier scores near-perfect across all dimensions
- OSS struggles most on bias (2.8/5 in bias category) -- lacks capacity for nuanced stereotype handling
- Safety scores are closer due to guardrails catching the worst cases before either model sees them
- Full report: [`eval/reports/evaluation_report.pdf`](eval/reports/evaluation_report.pdf)

---

## Observability

All runtime metrics are collected in Redis and displayed in the Observability tab:

| Metric | Redis Key | Purpose |
|---|---|---|
| Request count per model | `metrics:requests:{model}` | Usage volume |
| Latency (last 100) | `metrics:latency:{model}` | Avg / P50 / P95 |
| Guardrail blocks | `metrics:guardrail_blocks` | Safety coverage |
| Tool calls per tool | `metrics:tool_calls:{tool}` | Tool usage patterns |
| Tool failures | `metrics:tool_failures:{tool}` | Tool reliability |
| Context summarizations | `metrics:summarizations` | Memory pressure |
| Session restores | `metrics:session_restores` | Persistence usage |
| Errors | `metrics:errors` | System health |

All writes are atomic (Redis INCR/LPUSH) and fire-and-forget -- they never block the response path.

---

## Project Structure

```
.
├── src/
│   ├── app.py                 # Gradio UI (Arena, Single Model, Evaluation, Observability)
│   ├── config.py              # Pydantic settings from environment
│   ├── guardrails.py          # Input/output safety filters
│   ├── observability.py       # Redis-backed metrics collector
│   ├── models/
│   │   ├── base.py            # Abstract async model interface
│   │   ├── oss_model.py       # Qwen2.5-0.5B via transformers
│   │   └── frontier_model.py  # GPT-4.1 via AsyncOpenAI
│   ├── memory/
│   │   ├── working.py         # Layer 1: Sliding window
│   │   ├── summarizer.py      # Layer 2: LLM compression
│   │   ├── persistence.py     # Layer 3: Redis session store
│   │   └── manager.py         # Orchestrator
│   └── tools/
│       ├── base.py            # Tool protocol + ToolResult
│       ├── router.py          # Hybrid routing (keyword + LLM)
│       ├── registry.py        # Execution orchestrator
│       ├── web_search.py      # DuckDuckGo search
│       ├── wikipedia.py       # Wikipedia lookup
│       └── calculator.py      # Safe math + datetime
├── eval/
│   ├── prompts/               # 36 evaluation prompts (JSON)
│   ├── judge.py               # LLM-as-judge scoring
│   ├── run_eval.py            # Async evaluation runner
│   └── generate_report.py     # Charts + PDF generation
├── tests/                     # pytest suite
├── .github/workflows/
│   ├── ci.yml                 # Test + lint on push
│   └── deploy.yml             # Auto-deploy to HF Spaces
├── Dockerfile                 # CPU-optimized for HF Spaces
└── requirements.txt
```

---

## Cost + Latency

| Metric | OSS (Qwen2.5-0.5B) | Frontier (GPT-4.1-mini) |
|---|---|---|
| **Hosting** | HF Spaces Free (2 vCPU, 16GB) | OpenAI API (pay-per-token) |
| **Cost/month** | $0 | ~$1-5 (light usage) |
| **Avg latency (CPU)** | ~10-15s | ~1-2s |
| **Model size** | ~1GB (FP32) | N/A (API) |
| **Max context** | 32K tokens | 1M tokens |
| **Tool overhead** | +0.5-2s (web search/wiki) | Same |
| **Redis overhead** | < 1ms per read/write | Same |

---

## Tradeoffs

| Decision | Rationale |
|---|---|
| **Qwen2.5-0.5B** | Deployable free on HF Spaces. Demonstrates the quality gap clearly, which is the point of the comparison. |
| **Regex guardrails** | Fast (< 1ms), transparent, no external API. Limited coverage compared to a moderation API, but sufficient for demo scope. |
| **LLM summarization** | Information-preserving (captures names, facts, decisions). Costs API tokens on overflow, but only triggers when the window is full. |
| **Redis for persistence + metrics** | Single infrastructure dependency. Avoids Langfuse/Prometheus overhead. Free tier (30MB) is more than enough. |
| **Keyword-first tool routing** | Handles 90% of cases instantly (0ms). LLM fallback only fires for subtle queries. Avoids unnecessary API calls. |
| **ast-based calculator** | Never calls `eval()`. Safe by construction via AST node whitelist. |
| **Async-first architecture** | Non-blocking I/O throughout. OSS model bridges sync HF inference to async via `run_in_executor`. Event loop stays free for concurrent requests. |
| **CI/CD via GitHub Actions** | Every push to main: lint + test + auto-deploy to HF Spaces. No manual deployment. |

---

## What I Would Improve With More Time

1. **Vector-based long-term memory** -- semantic search over conversation history using Redis Search embeddings, enabling retrieval of relevant past context beyond the summary
2. **Moderation API** -- replace regex guardrails with OpenAI Moderation API or a fine-tuned safety classifier for broader coverage
3. **Quantized OSS model** -- GPTQ/AWQ 4-bit quantization to reduce memory footprint and improve CPU inference latency by ~3-5x
4. **LLM-driven tool routing** -- replace keyword heuristic with a fine-tuned small classifier model for more reliable tool selection
5. **Streaming in arena mode** -- custom Gradio frontend to enable independent SSE streams per model (current Gradio limitation prevents this)
6. **Batched evaluation** -- run evaluation prompts concurrently via `asyncio.gather()` to reduce total eval time from ~10 minutes to ~2 minutes
