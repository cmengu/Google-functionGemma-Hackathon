<img src="assets/banner.png" alt="Logo" style="border-radius: 30px; width: 100%;">

**Live leaderboard:** [https://cactusevals.ngrok.app](https://cactusevals.ngrok.app)

Edge–cloud **function-calling** demo for the Cactus × DeepMind hackathon. **FunctionGemma** (on-device via [Cactus](https://cactuscompute.com)) parses natural-language assistant commands into structured tool JSON; **Gemini 2.5 Flash** is invoked only when local outputs fail validation—so you optimize **correctness**, **latency**, and **on-device share** on the same harness the eval server uses. No multi-agent graph: one **`generate_hybrid(messages, tools)`** implements the router.

Built for an objective leaderboard (local **`benchmark.py`** + held-out server eval). Architectural choices in **`main.py`** are easy to benchmark, not hand-waved.

## Architecture

Two execution paths share the same **`generate_hybrid`** entry point and are scored on identical scenarios (public suite locally; hidden suite on submit):

| Path | Stack | Role |
| --- | --- | --- |
| **Edge (on-device)** | FunctionGemma 270M IT, `cactus_complete(..., force_tools=True)`, optional Tool RAG | Primary: hinted prompts, **one model call per clause** for multi-intent utterances, `_fix_args` + `_validate` |
| **Cloud rescue** | Gemini 2.5 Flash (`google.genai`), tool declarations mirroring the same schema | **Whole-request fallback** when any local clause fails validation, or single-shot local parse is invalid / empty |

There is no LangGraph orchestrator in this repo—routing is an **explicit branch** in `main.py` (multi-tool vs single-tool → local → optional cloud).

## What’s implemented

Capabilities shipped in **`main.py`** (submission surface):

| Capability | Implementation | Entry |
| --- | --- | --- |
| Tool-conditioned prompts | `_get_base_prompt`, `_build_prompt` — fingerprinted tool list, formatting rules | Every `_infer_local` call |
| Deterministic hints | Regex for alarms, reminders, weather city, recipients, contact search | Injected above user message when patterns match |
| Multi-tool split | `_is_multi_tool`, `_split_into_clauses`, `_resolve_pronouns` | `generate_hybrid` multi-action branch |
| JSON repair | `_fix_json`, `_try_parse` | After `cactus_complete` |
| Argument repair | `_fix_args` — types, 24h alarm from text, title/time cleanup, name casing | Local + cloud outputs |
| Cloud inference | `_infer_cloud` | On validation failure |

## Routing flow (edge → cloud)

```
user message
    → multi-action? ──yes──► split clauses ──► per clause: _infer_local → _fix_args → _validate
    │                              │ any clause fails ──► _infer_cloud(full message)
    │                              └── all ok ──► merge function_calls, source=on-device
    └── no ──► _infer_local → _fix_args → _validate ──ok──► source=on-device
                                      └── fail ──► _infer_cloud → source=cloud
```

**Return contract:** `function_calls`, `total_time_ms`, and optional `source` (`on-device` | `cloud`). The harness grades tool JSON; the server leaderboard also reports **on-device %**.

## Evaluation

### Local benchmark (`benchmark.py`)

| Aspect | Detail |
| --- | --- |
| Cases | **30** scenarios in-repo (`easy` / `medium` / `hard`) — weather, alarms, reminders, messaging, contacts, music, timers; includes **multi-tool** utterances |
| Metrics | Per case: **F1** on function calls (set overlap via `_call_matches`), **wall time** (`total_time_ms`), **source** |
| Aggregate score | Per difficulty, `level_score = 0.60 × avg_F1 + 0.15 × time_score + 0.25 × on_device_ratio` (time_score vs 500ms baseline); difficulties weighted **easy 20% / medium 30% / hard 50%** — see `compute_total_score` in `benchmark.py` |

Run:

```bash
python benchmark.py
```

### Leaderboard submit

Held-out eval + ranking: **`python submit.py --team "YourTeamName" --location "YourCity"`** (rate-limited). Objective score blends **accuracy**, **speed**, and **edge ratio** (local preferred).

### Integration tests

This starter does **not** include a separate pytest integration suite (unlike a full product repo). Treat **`benchmark.py`** as your regression loop before submit.

## Quick start

```bash
git clone https://github.com/cactus-compute/cactus
cd cactus && source ./setup && cd ..
cactus build --python
cactus download google/functiongemma-270m-it --reconvert
cactus auth
pip install google-genai
export GEMINI_API_KEY="your-key"
```

Clone **this** hackathon repo beside Cactus (or adjust `main.py` paths), then:

```bash
cd functiongemma-hackathon
python benchmark.py
python submit.py --team "YourTeamName" --location "YourCity"
```

**Constraint:** Edit only the **body** of **`generate_hybrid`** in `main.py`; keep its signature and return shape compatible with `benchmark.py`.

## Submissions

- Your main task is to modify the **internal logic** of the `generate_hybrid` method in `main.py`.
- Do not change the **parameters** or required **return fields** of `generate_hybrid` (must stay compatible with `benchmark.py`).
- Submit with `python submit.py --team "YourTeamName" --location "YourCity"` (max once per hour).
- Rankings: [leaderboard](https://cactusevals.ngrok.app).
- The top hackers in each location will make it to judging.

## Design decisions

1. **Edge-first** — Default to FunctionGemma to maximize on-device % and keep latency predictable; cloud is a correctness backstop.
2. **One tool per local call for multi-intent** — Small models handle a single tool JSON object more reliably than a list in one shot.
3. **Hints from regex** — Reduces ambiguity (times, cities, names) without an extra network round-trip.
4. **`_fix_args` on both paths** — Local and Gemini outputs go through the same normalizer before validation.
5. **Full-message cloud fallback on partial failure** — Avoid returning a toxic mix of good and bad clauses when any slice fails.
6. **Conditional Tool RAG** — `tool_rag_top_k` when `len(tools) > 4` to limit distraction on small tool sets.
7. **`temperature=0` locally** — Stable tool choice for scoring.
8. **Separate from Cactus `cloud_handoff`** — The JSON `cloud_handoff` flag from `cactus_complete` is optional signal; this repo’s policy is **validation-driven** `_infer_cloud`.
9. **Mac / Cactus layout** — Model path assumes `cactus/weights/functiongemma-270m-it` relative to this repo; align with your install.
10. **Gemini for rescue only** — Keeps cloud cost and privacy surface lower than an always-cloud parser.

## Graceful degradation

| Dependency | Failure mode | System behaviour |
| --- | --- | --- |
| Gemini (`GEMINI_API_KEY`, network) | `_infer_cloud` raises | Multi-tool: may return partial local calls if any clause validated; single-tool: return last local attempt |
| Local parse | Empty / invalid JSON | Retry via cloud when available |
| Cactus runtime | Import / init failure | Outside `generate_hybrid`; fix env before benchmarking |

## Known limitations

- **Eval servers** may not offer Gemini; implementations must tolerate `_infer_cloud` failures.
- **No in-repo pytest graph tests** — use `benchmark.py` as the integration check.
- **Weights path** in `main.py` is environment-specific until Cactus is installed.

## Mermaid diagram — do you need one?

**No.** The control flow is a **short tree** (multi-tool vs single → local → optional cloud). The **Architecture** table and **Routing flow** above are enough for the README. Use Mermaid later if you add **multiple models**, **LangGraph**, or **branching policies** that are hard to scan in text.

## Challenge context

- Teams decide **when** to stay on FunctionGemma vs **Gemini Flash**, trading **F1**, **speed**, and **on-device ratio**.
- At least one teammate should use a **Mac**; Cactus targets Mac + mobile-style deployments.

## Qualitative judging (rubric)

1. Quality of **hybrid routing** — depth, correctness, creativity.
2. End-to-end products that **execute** function calls for real tasks.
3. Low-latency **voice-to-action** using `cactus_transcribe` where relevant.

## Project structure

```
functiongemma-hackathon/
  main.py           # generate_hybrid — your submission logic
  benchmark.py      # Tool defs, 30 cases, F1 + score
  submit.py         # Push main.py to leaderboard
  assets/           # README banner (if present)
```

## Links

- [Cactus API keys](https://cactuscompute.com/dashboard/api-keys), [Gemini keys](https://aistudio.google.com/api-keys), [Reddit r/cactuscompute](https://www.reddit.com/r/cactuscompute/)
- GCP / hackathon credits: [SF](https://trygcp.dev/claim/cactus-x-gdm-hackathon-sf), [Boston](https://trygcp.dev/claim/cactus-x-gdm-hackathon-boston), [DC](https://trygcp.dev/claim/cactus-x-gdm-hackathon-dc), [London](https://trygcp.dev/claim/cactus-x-gdm-hackathon-london), [Singapore](https://trygcp.dev/claim/cactus-x-gdm-hackathon), [Online](https://trygcp.dev/claim/cactus-x-gdm-hackathon-online)
- Technical reading: [Maths, CS & AI Compendium](https://github.com/HenryNdubuaku/maths-cs-ai-compendium)

---

## Cactus API reference (runtime)

### `cactus_init(model_path, corpus_dir=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Path to model weights directory |
| `corpus_dir` | `str` | (Optional) dir of txt/md files for auto-RAG |

### `cactus_complete(model, messages, **options)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | From `cactus_init` |
| `messages` | `list` | Chat turns |
| `tools` | `list` | OpenAI-style tool definitions |
| `force_tools` | `bool` | Constrain to tool JSON |
| `tool_rag_top_k` | `int` | Subset tools by RAG (0 = all) |
| `confidence_threshold` | `float` | Triggers **`cloud_handoff`** in response JSON when confidence low |
| `temperature`, `max_tokens`, `stop_sequences`, `callback` | … | Standard generation |

Response JSON always includes `success`, `function_calls`, `confidence`, `cloud_handoff`, timing and throughput fields — see Cactus docs.

### Other

- `cactus_transcribe`, `cactus_embed`, `cactus_rag_query`, `cactus_reset`, `cactus_stop`, `cactus_destroy`, `cactus_get_last_error` — see [Cactus documentation](https://github.com/cactus-compute/cactus).

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
