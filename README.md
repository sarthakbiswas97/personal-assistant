# AI Personal Assistant: OSS vs Frontier Comparison

A comparative AI assistant built with two backends -- an open-source model (Qwen2.5-0.5B-Instruct) and a frontier model (OpenAI GPT-4.1) -- with a shared Gradio interface, safety guardrails, and an automated evaluation pipeline.

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key (for frontier model and evaluation)

### Local Setup

```bash
# Clone and enter the project
git clone https://github.com/<your-username>/ai-assistant.git
cd ai-assistant

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt        # runtime only
pip install -r requirements-dev.txt    # includes testing/linting

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the App

```bash
python3 -m src.app
```

Open `http://localhost:7860` in your browser. Select a model from the dropdown and start chatting.

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Run Evaluation

```bash
# Run both models against all evaluation prompts
python3 -m eval.run_eval

# Generate infographic report
python3 -m eval.generate_report
```

Results are saved to `eval/outputs/` and charts to `eval/reports/`.

### Docker

```bash
docker build -t ai-assistant .
docker run -p 7860:7860 -e OPENAI_API_KEY=your-key ai-assistant
```

## Architecture

```
assistant/
├── src/
│   ├── config.py              # Pydantic settings from environment
│   ├── memory.py              # Sliding window conversation memory
│   ├── guardrails.py          # Input/output safety filters
│   ├── app.py                 # Gradio UI with model selector
│   └── models/
│       ├── base.py            # Abstract async model interface
│       ├── oss_model.py       # Qwen2.5-0.5B via transformers
│       └── frontier_model.py  # GPT-4.1 via OpenAI async SDK
├── eval/
│   ├── prompts/               # 36 curated evaluation prompts
│   │   ├── factual.json       # 12 factual accuracy prompts
│   │   ├── adversarial.json   # 12 bias/stereotype prompts
│   │   └── safety.json        # 12 jailbreak/harmful prompts
│   ├── judge.py               # LLM-as-judge scoring (1-5 scale)
│   ├── run_eval.py            # Async evaluation runner
│   └── generate_report.py     # Matplotlib infographic generation
├── tests/                     # pytest suite (44 tests)
├── Dockerfile                 # CPU-optimized for HF Spaces
└── requirements.txt
```

### Design Decisions

**Async-first architecture.** Both model backends implement an async interface (`async generate()`, `async stream()`). The OSS model bridges synchronous HuggingFace `model.generate()` to async via `TextIteratorStreamer` + background threads. This keeps the Gradio event loop responsive during inference.

**Immutable data throughout.** `Message`, `ConversationSnapshot`, `JudgmentScore`, `GuardrailResponse`, and `Settings` are all frozen dataclasses or Pydantic models. No mutation means no hidden side effects -- easier to debug, safer for concurrent access.

**Lazy model loading.** Models are loaded on first request, not at startup. This avoids loading both backends when only one is needed and keeps the app boot time fast.

**Guardrails as a separate layer.** Input and output safety checks are decoupled from model logic. They run before and after inference respectively, so they work identically across both backends.

**LLM-as-judge evaluation.** Uses GPT-4.1 to score responses on three dimensions (hallucination 1-5, safety 1-5, bias 1-5) with structured JSON output. More reliable than keyword matching for nuanced evaluation.

## Cost + Latency Table

| Metric | OSS (Qwen2.5-0.5B) | Frontier (GPT-4.1-mini) |
|---|---|---|
| **Hosting** | HF Spaces Free (2 vCPU, 16GB) | OpenAI API (pay-per-token) |
| **Cost/month** | $0 | ~$1-5 (light usage) |
| **Avg latency** | ~5-15s (CPU) | ~1-3s |
| **Model size** | ~1GB (FP32) | N/A (API) |
| **Max context** | 32K tokens | 1M tokens |
| **Quality** | Limited (0.5B params) | Strong |

*Latency estimates for CPU inference on HF Spaces free tier. GPU deployment would reduce OSS latency to ~0.5-2s.*

## Tradeoffs

| Decision | Tradeoff |
|---|---|
| Qwen2.5-0.5B (tiny model) | Deployable for free on HF Spaces, but noticeably weaker on complex reasoning. Good for demonstrating the evaluation gap. |
| Regex-based guardrails | Fast and transparent, but limited coverage. A production system would use a dedicated moderation API or fine-tuned classifier. |
| Sliding window memory | Simple and predictable, but loses early context. Could be improved with summarization-based compression. |
| Single-process serving | Fine for demo/evaluation, but won't scale. Production would use vLLM or TGI for the OSS model. |
| LLM-as-judge | More nuanced than rule-based scoring, but introduces evaluator bias and costs API tokens. |

## What I Would Improve With More Time

1. **Semantic memory** -- Replace sliding window with retrieval-augmented memory using embeddings. Summarize old turns instead of dropping them.

2. **Tool use** -- Add function calling capabilities (web search, calculator, code execution) to both assistants.

3. **Moderation API** -- Replace regex guardrails with OpenAI Moderation API or a fine-tuned safety classifier for better coverage and fewer false positives.

4. **Observability** -- Add structured logging with request tracing, latency percentiles, and token usage dashboards (e.g., Langfuse or Phoenix).

5. **Batched evaluation** -- Run evaluation prompts concurrently with `asyncio.gather()` instead of sequentially, reducing eval time significantly.

6. **A/B testing UI** -- Side-by-side comparison mode where both models respond to the same prompt simultaneously.

7. **Quantized OSS model** -- Use GPTQ/AWQ 4-bit quantization to reduce memory footprint and improve latency on CPU.

8. **CI/CD pipeline** -- GitHub Actions for automated testing, linting, Docker build, and HF Spaces deployment on merge to main.
