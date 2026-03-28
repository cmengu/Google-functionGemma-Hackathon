<img src="assets/banner.png" alt="Logo" style="border-radius: 30px; width: 100%;">

## Overview

**Live leaderboard:** [https://cactusevals.ngrok.app](https://cactusevals.ngrok.app)

Hybrid **function-calling** stack for hackathon evals: **on-device** [FunctionGemma](https://github.com/google-deepmind/functiongemma) (via [Cactus](https://cactuscompute.com)) handles most user utterances; **cloud** [Gemini 2.5 Flash](https://ai.google.dev/) runs only when local parsing or validation fails. The goal is **correct tool JSON**, **low latency**, and **high on-device share** on a hidden multi-tool benchmark—without changing the `generate_hybrid(messages, tools)` signature expected by `benchmark.py` / the submit server.

## Submissions

- Your main task is to modify the **internal logic** of the `generate_hybrid` method in `main.py`.
- Do not modify the input or output signature (function arguments and return variables) of the `generate_hybrid` method. Keep the hybrid interface compatible with `benchmark.py`.
- Submit to the leaderboard `python submit.py --team "YourTeamName" --location "YourCity"`, only 1x every 1hr.
- The dataset is a hidden Cactus eval, quite difficult for FunctionGemma by design.
- Use `python benchmark.py` to iterate, but your best score is preserved.
- For transparency, hackers can see live rankings on the [leaderboard](https://cactusevals.ngrok.app).
- Leaderboard will start accepting submissions once event starts.
- The top hackers in each location will make it to judging.

## Architecture: edge vs cloud

| Layer | What runs | Role |
| --- | --- | --- |
| **Edge (on-device)** | FunctionGemma 270M IT, `cactus_complete` with `force_tools`, optional Tool RAG (`tool_rag_top_k` when many tools) | Primary path: hinted system prompt, per-clause calls for compound requests, deterministic arg repair (`_fix_args`) |
| **Cloud** | Gemini 2.5 Flash, `google.genai` with parallel function declarations | **Rescue path:** whole-request fallback when any clause fails validation locally, or single-shot request fails `_validate` after local inference |

**Return shape:** `generate_hybrid` returns a `dict` with at least `function_calls` (list of `{name, arguments}`) and `total_time_ms`. Successful paths set `source` to `"on-device"`; Gemini path sets `"source": "cloud"` (see `main.py`).

### Routing logic (high level)

1. **Multi-action utterances** (`and` / `also` / `then`, comma splits, etc.): split into clauses → **one local** `cactus_complete` per clause → merge calls. If **any** clause fails validation, **one** `_infer_cloud` on the **full** user message (correctness over edge ratio for that case).
2. **Single-action utterances:** one local call with hints → `_fix_args` + `_validate`. If invalid or empty → **cloud** retry with accumulated latency.

Benchmark scoring (from `submit.py` / server) reports **F1**, **avg time**, and **on-device %**—your routing strategy directly trades off the last two against the first.

### What's implemented in `main.py`

| Capability | Implementation |
| --- | --- |
| Tool-aware prompting | `_build_prompt` / `_get_base_prompt` — tool list fingerprinted, rules for times, names, reminders |
| Pre-resolved hints | Regex for alarms, reminders, weather city, `send_message` recipient, `search_contacts` query — reduces small-model ambiguity |
| Multi-tool decomposition | `_is_multi_tool`, `_split_into_clauses`, `_resolve_pronouns` — one tool per local generation |
| Robust JSON extraction | `_fix_json`, `_try_parse` — fences, alternate keys, partial recovery |
| Argument normalization | `_fix_args` — types, 24h alarm fix from raw text, title/time hygiene, name casing from user text |
| Cloud handoff | `_infer_cloud` — Gemini with system instruction aligned to local conventions |

There is no separate LangGraph stack in this repo—the “orchestration” is this **explicit if/else** edge-first router.

## Quick start

1. Fork / clone this repo on a Mac (Cactus targets Mac / edge devices).
2. Install Cactus per [upstream instructions](https://github.com/cactus-compute/cactus): `source ./setup` from the Cactus repo, `cactus build --python`, `cactus download google/functiongemma-270m-it --reconvert`, `cactus auth`.
3. `pip install google-genai`
4. `export GEMINI_API_KEY="..."` for local cloud-fallback testing (eval servers may not expose cloud).
5. `python benchmark.py` — prints per-case results and aggregates.
6. Submit (rate-limited): `python submit.py --team "YourTeamName" --location "YourCity"`

**Hackathon constraint:** only change the **internal logic** of `generate_hybrid` in `main.py`; keep its **parameters and return contract** compatible with `benchmark.py`.

## Project structure

| File | Purpose |
| --- | --- |
| `main.py` | `generate_hybrid`, local + cloud inference, routing (your submission body) |
| `benchmark.py` | Tool definitions, public benchmark cases, scoring driver |
| `submit.py` | Upload `main.py` to leaderboard, poll results |

## Design decisions (hybrid-focused)

1. **Clause-per-call for multi-tool** — FunctionGemma is reliable on **one** tool at a time; splitting utterances avoids overlapping intents in a single JSON blob.
2. **Hints before the model** — Deterministic extraction of time, city, names lowers bad parses without extra network.
3. **`_fix_args` after every path** — Same repair for local and cloud so validation compares apples to apples.
4. **Cloud on first clause failure (multi-tool)** — Partial local merges are avoided when any fragment is wrong; one Gemini call fixes the whole request.
5. **Tool RAG only when `len(tools) > 4`** — Cuts noise when the tool set is small; expands when many tools compete.
6. **`temperature=0` on-device** — Reproducible tool choice for scoring; cloud uses API defaults appropriate for structured calls.

## Graceful degradation

| Dependency | Failure mode | Behaviour |
| --- | --- | --- |
| `GEMINI_API_KEY` / network | `_infer_cloud` raises | Multi-tool: return partial `all_calls` if any clause succeeded; single-tool: return last local attempt |
| Malformed local JSON | `_try_parse` empty | Falls through to cloud when available |
| Cactus runtime | (not handled in `generate_hybrid`) | Fix environment; out of scope for router edits |

## Known limitations

- **Eval environment** may not allow Gemini; strategies that assume cloud will see lower cloud usage on device but must still behave when `_infer_cloud` fails.
- **`generate_hybrid` must stay compatible** with `benchmark.py` — no extra required keys beyond what the harness expects (adding optional keys like `source` is fine if the server ignores them).
- Model weights live under `cactus/weights/` per Cactus layout; paths in `main.py` assume that layout.

## Mermaid diagram — do you need one?

**No.** This project’s control flow is a **short branch** (multi-tool vs single → local → optional cloud). A diagram is **optional** for slides or judge handouts; the **Architecture** table and **Routing logic** bullets are enough for README readability. Reserve Mermaid for future extensions (e.g. LangGraph, multiple models) where the graph is non-obvious.

---

## Challenge context (organizer copy)

- FunctionGemma is strong at tool calling on-device; small models still struggle on harder compound prompts.
- Teams design strategies for **when** to stay local vs call **Gemini Flash**, optimizing **tool correctness**, **speed**, and **edge ratio** (local preferred).
- At least one teammate should use a Mac; Cactus runs on Macs and targets mobile / wearable paths.

## Qualitative judging (rubric)

1. Quality of **hybrid routing** — depth, correctness, creativity.
2. End-to-end products that **execute** function calls for real tasks.
3. Low-latency **voice-to-action** using `cactus_transcribe` where relevant.

---

## Cactus API reference (runtime)

### `cactus_init(model_path, corpus_dir=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Path to model weights directory |
| `corpus_dir` | `str` | (Optional) dir of txt/md files for auto-RAG |

```python
model = cactus_init("weights/lfm2-vl-450m")
model = cactus_init("weights/lfm2-rag", corpus_dir="./documents")
```

### `cactus_complete(model, messages, **options)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | Model handle from `cactus_init` |
| `messages` | `list\|str` | List of message dicts or JSON string |
| `tools` | `list` | Optional tool definitions for function calling |
| `temperature` | `float` | Sampling temperature |
| `top_p` | `float` | Top-p sampling |
| `top_k` | `int` | Top-k sampling |
| `max_tokens` | `int` | Maximum tokens to generate |
| `stop_sequences` | `list` | Stop sequences |
| `include_stop_sequences` | `bool` | Include matched stop sequences in output (default: `False`) |
| `force_tools` | `bool` | Constrain output to tool call format |
| `tool_rag_top_k` | `int` | Select top-k relevant tools via Tool RAG (default: 2, 0 = use all tools) |
| `confidence_threshold` | `float` | Minimum confidence for local generation (default: 0.7, triggers cloud_handoff when below) |
| `callback` | `fn` | Streaming callback `fn(token, token_id, user_data)` |

```python
messages = [{"role": "user", "content": "Hello!"}]
response = cactus_complete(model, messages, max_tokens=100)
print(json.loads(response)["response"])
```

```python
tools = [{
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}]
response = cactus_complete(model, messages, tools=tools)
```

**Response format** (all fields always present):

```json
{
    "success": true,
    "error": null,
    "cloud_handoff": false,
    "response": "Hello! How can I help?",
    "function_calls": [],
    "confidence": 0.85,
    "time_to_first_token_ms": 45.2,
    "total_time_ms": 163.7,
    "prefill_tps": 619.5,
    "decode_tps": 168.4,
    "ram_usage_mb": 245.67,
    "prefill_tokens": 28,
    "decode_tokens": 50,
    "total_tokens": 78
}
```

**Cloud handoff response** (when model detects low confidence):

```json
{
    "success": false,
    "error": null,
    "cloud_handoff": true,
    "response": null,
    "function_calls": [],
    "confidence": 0.18,
    "time_to_first_token_ms": 45.2,
    "total_time_ms": 45.2,
    "prefill_tps": 619.5,
    "decode_tps": 0.0,
    "ram_usage_mb": 245.67,
    "prefill_tokens": 28,
    "decode_tokens": 0,
    "total_tokens": 28
}
```

Built-in `cloud_handoff` in JSON is **separate** from this repo’s `_infer_cloud`; your `generate_hybrid` strategy may use confidence, validation, or other signals—not only Cactus’s flag.

### `cactus_transcribe(model, audio_path, prompt="")`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | Whisper model handle |
| `audio_path` | `str` | Path to audio file (WAV) |
| `prompt` | `str` | Whisper prompt for language/task |

```python
whisper = cactus_init("weights/whisper-small")
prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
response = cactus_transcribe(whisper, "audio.wav", prompt=prompt)
print(json.loads(response)["response"])
cactus_destroy(whisper)
```

### `cactus_embed(model, text, normalize=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | Model handle |
| `text` | `str` | Text to embed |
| `normalize` | `bool` | L2-normalize embeddings (default: False) |

```python
embedding = cactus_embed(model, "Hello world")
print(f"Dimension: {len(embedding)}")
```

### `cactus_reset(model)`, `cactus_stop(model)`, `cactus_destroy(model)`, `cactus_get_last_error()`

Use between unrelated conversations, to cancel generation, to free memory, and to read last error — see Cactus docs for details.

### `cactus_rag_query(model, query, top_k=5)`

Requires `corpus_dir` at init.

```python
model = cactus_init("weights/lfm2-rag", corpus_dir="./documents")
chunks = cactus_rag_query(model, "What is machine learning?", top_k=3)
```

## Quick example

```python
import json
from cactus import cactus_init, cactus_complete, cactus_destroy

model = cactus_init("weights/lfm2-vl-450m")
messages = [{"role": "user", "content": "What is 2+2?"}]
response = json.loads(cactus_complete(model, messages))
print(response["response"])

cactus_destroy(model)
```

## Links

- [Cactus API keys](https://cactuscompute.com/dashboard/api-keys), [Gemini keys](https://aistudio.google.com/api-keys), [Reddit r/cactuscompute](https://www.reddit.com/r/cactuscompute/)
- GCP / hackathon credits: [SF](https://trygcp.dev/claim/cactus-x-gdm-hackathon-sf), [Boston](https://trygcp.dev/claim/cactus-x-gdm-hackathon-boston), [DC](https://trygcp.dev/claim/cactus-x-gdm-hackathon-dc), [London](https://trygcp.dev/claim/cactus-x-gdm-hackathon-london), [Singapore](https://trygcp.dev/claim/cactus-x-gdm-hackathon), [Online](https://trygcp.dev/claim/cactus-x-gdm-hackathon-online)
- Technical reading: [Maths, CS & AI Compendium](https://github.com/HenryNdubuaku/maths-cs-ai-compendium)
