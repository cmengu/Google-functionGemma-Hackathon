<img src="assets/banner.png" alt="Banner" style="border-radius: 30px; width: 100%;">

# FunctionGemma hackathon starter

**Leaderboard:** [cactusevals.ngrok.app](https://cactusevals.ngrok.app)

Minimal repo for the Cactus × Google DeepMind hackathon: turn user chat into **function calls** (tools) using **FunctionGemma on-device** (via [Cactus](https://cactuscompute.com)) and **Gemini 2.5 Flash** when the local path fails validation. You implement the policy inside **`generate_hybrid(messages, tools)`** in [`main.py`](main.py).

---

## What this project does

1. **`benchmark.py`** builds a list of assistant scenarios (weather, alarms, reminders, SMS-style messaging, contact search, music, timers). For each case it calls **`generate_hybrid`** with OpenAI-style **tool definitions** and checks returned **`function_calls`** against gold arguments (F1-style matching).
2. **`main.py`** loads **FunctionGemma 270M IT** from `cactus/weights/functiongemma-270m-it`, runs **`cactus_complete`** with **`force_tools=True`**, repairs JSON and arguments, and optionally calls **`google.genai`** (Gemini) for a second pass.
3. **`submit.py`** uploads **`main.py`** only to the remote eval server and prints score, F1, latency, and on-device percentage.

There is no web app, database, or extra packages beyond what Cactus + `google-genai` need for this flow.

---

## Edge vs cloud

| | **Edge (on-device)** | **Cloud** |
| --- | --- | --- |
| **Stack** | `cactus_init` + `cactus_complete`, model path `cactus/weights/functiongemma-270m-it` | `genai.Client`, model **`gemini-2.5-flash`** |
| **When** | Always tried first (per clause for multi-intent prompts, or once for single-intent) | After **`_validate`** fails on local output, or any multi-tool **clause** fails (then whole message goes to Gemini) |
| **Marked as** | `source: "on-device"` | `source: "cloud"` (or `"empty"` if cloud returns no calls) |

**`benchmark.py`** sets `CACTUS_NO_CLOUD_TELE=1`; scoring uses **`result.get("source", "unknown")`** so your hybrid router should set `source` when possible.

**Tool RAG:** `_infer_local` passes **`tool_rag_top_k: 3`** into `cactus_complete` only when **`len(tools) > 4`**.

**Local inference defaults:** `temperature=0`, `max_tokens` 80 or 112, and Gemma stop sequences including `<end_of_turn>` (see `_infer_local` in `main.py`).

---

## `generate_hybrid` routing (summary)

- **Multi-intent message** (`_is_multi_tool`): split with `_split_into_clauses`, optional **`_resolve_pronouns`**, then for each clause **`_infer_local` → `_fix_args` → `_validate`**. If any clause fails, **`_infer_cloud(messages, tools)`** on the **full** user turn (plus prior `total_time_ms`). If all succeed, concatenate calls and return **`source: "on-device"`**.
- **Single-intent:** one **`_infer_local`**, **`_fix_args`**, **`_validate`**; on failure **`_infer_cloud`** with accumulated time.
- **Cloud unavailable:** exceptions in **`_infer_cloud`** fall back to whatever local produced (multi-tool may return partial `all_calls`).

Main helpers (all in `main.py`): **`_build_prompt` / `_get_base_prompt`**, **`_fix_json` / `_try_parse`**, **`_fix_args`**, **`_validate`**.

---

## Repository layout

| File | Role |
| --- | --- |
| [`main.py`](main.py) | Hybrid router — **only the body of `generate_hybrid` is meant to be your submission** (keep signature and return shape compatible with `benchmark.py`) |
| [`benchmark.py`](benchmark.py) | Seven tool schemas, **30** eval cases (10 easy / 10 medium / 10 hard), `run_benchmark`, `compute_total_score` |
| [`submit.py`](submit.py) | POST `main.py` to `https://cactusevals.ngrok.app` |

`cactus/` is gitignored here; you clone/build Cactus separately per organizer instructions.

---

## Tools (benchmark)

| Name | Purpose |
| --- | --- |
| `get_weather` | `location` |
| `set_alarm` | `hour`, `minute` (24h internally after repair) |
| `send_message` | `recipient`, `message` |
| `create_reminder` | `title`, `time` |
| `search_contacts` | `query` |
| `play_music` | `song` |
| `set_timer` | `minutes` |

---

## Scoring (`benchmark.py`)

Per case: **F1** over `function_calls` vs `expected_calls`, **`total_time_ms`**, **`source`**.

Per difficulty bucket, the combined level score is:

`0.60 × avg_F1 + 0.15 × time_score + 0.25 × on_device_ratio`

where `time_score = max(0, 1 - avg_time_ms / 500)`.

Difficulty weights: **easy 20%**, **medium 30%**, **hard 50%**. Printed **TOTAL SCORE** is in **0–100%**.

The hosted leaderboard (`submit.py`) reports **`score`**, **`f1`**, **`avg_time_ms`**, **`on_device_pct`** from the server.

---

## Quick start

1. Install **Cactus** from [cactus-compute/cactus](https://github.com/cactus-compute/cactus): `source ./setup`, `cactus build --python`, `cactus download google/functiongemma-270m-it --reconvert`, `cactus auth`.
2. In this repo directory (next to your `cactus` tree): `pip install google-genai` and `export GEMINI_API_KEY=...` if you want cloud fallback locally.
3. `python benchmark.py`
4. `python submit.py --team "YourTeam" --location "YourCity"`

Hackathon rule: do not change **`generate_hybrid`**’s parameters or the required return fields expected by **`benchmark.py`**.

---

## Diagrams (Mermaid)

**Not required.** The pipeline is a small if/branch (multi vs single → local → optional Gemini). Tables above are enough; add a diagram only if you introduce more stages or services.

---

## Links

- [Cactus API keys](https://cactuscompute.com/dashboard/api-keys) · [Gemini API keys](https://aistudio.google.com/api-keys) · [r/cactuscompute](https://www.reddit.com/r/cactuscompute/)
- Hackathon credits (if still active): [SF](https://trygcp.dev/claim/cactus-x-gdm-hackathon-sf), [Boston](https://trygcp.dev/claim/cactus-x-gdm-hackathon-boston), [DC](https://trygcp.dev/claim/cactus-x-gdm-hackathon-dc), [London](https://trygcp.dev/claim/cactus-x-gdm-hackathon-london), [Singapore](https://trygcp.dev/claim/cactus-x-gdm-hackathon), [Online](https://trygcp.dev/claim/cactus-x-gdm-hackathon-online)

---

## Minimal Cactus usage (reference)

```python
import json
from cactus import cactus_init, cactus_complete, cactus_destroy

model = cactus_init("weights/lfm2-vl-450m")
response = json.loads(cactus_complete(model, [{"role": "user", "content": "Hello"}], max_tokens=100))
print(response.get("response"))
cactus_destroy(model)
```

This repo’s [`main.py`](main.py) uses **`cactus/python/src`** on `sys.path` and weights at **`cactus/weights/functiongemma-270m-it`**. Full `cactus_complete` options (tools, `force_tools`, `tool_rag_top_k`, `confidence_threshold`, streaming, etc.) are documented in the upstream Cactus README.
