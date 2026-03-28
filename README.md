# FunctionGemma Hackathon

A competition entry for the **Cactus Evals** hackathon, where the goal is to build the most accurate and fastest on-device function-calling system using a 270M parameter model running locally via the [Cactus](https://github.com/cactus-compute/cactus) runtime.

Ranked Top 5 on the Evaluation Leaderboard in Singapore

---

## What This Project Does

Given a user's natural language message and a set of available tools (functions), the system must:

1. Identify which tool(s) to call
2. Extract the correct arguments from the message
3. Return structured JSON with the function call(s)

For example:

> **User:** "Set an alarm for 7:30 AM and check the weather in New York."
>
> **Output:**
> ```json
> {
>   "function_calls": [
>     {"name": "set_alarm", "arguments": {"hour": 7, "minute": 30}},
>     {"name": "get_weather", "arguments": {"location": "New York"}}
>   ]
> }
> ```

The system runs **on-device** using `functiongemma-270m-it` (a 270M parameter instruction-tuned model) via the Cactus runtime, with Google Gemini as a cloud fallback.

---

## Supported Tools

| Tool | Description |
|---|---|
| `get_weather` | Get current weather for a city |
| `set_alarm` | Set an alarm (hour + minute in 24h format) |
| `send_message` | Send a message to a named contact |
| `create_reminder` | Create a reminder with a title and time |
| `search_contacts` | Search for a contact by name |
| `play_music` | Play a song or playlist |
| `set_timer` | Set a countdown timer in minutes |

---

## How It Works

### On-Device Inference (`main.py`)

The core logic lives in `generate_hybrid()` and uses several layers of optimisation to squeeze accuracy out of a 270M model:

**1. Hinted Prompts**
Before calling the model, the prompt is pre-populated with hints extracted by rule-based parsing (regex). This includes:
- Time values converted to the correct format (`3:00 PM`, `7:30 AM`)
- Alarm hours converted to 24-hour integers
- Recipient/location/query names copied verbatim from the message

This reduces ambiguity and gives the model maximum context in minimal tokens.

**2. Multi-Tool Clause Splitting**
Requests containing multiple actions (e.g., "set an alarm *and* check the weather") are split into individual clauses. The model is called once per clause — small models handle one tool at a time far more reliably than generating multiple tool calls in one shot.

**3. Argument Post-Processing**
After inference, arguments are normalised and validated:
- Time strings are normalised to `HH:MM AM/PM`
- Alarm hours are recomputed from the original message (the model often gets 24h conversion wrong)
- Reminder titles have time expressions stripped out
- Proper-noun casing is restored for names and locations

**4. Cloud Fallback**
If on-device inference fails validation, the system attempts a call to `gemini-2.5-flash` as a best-effort fallback. In the eval environment the cloud is unavailable, so the system is optimised entirely for on-device accuracy.

---

## Scoring

Submissions are evaluated on three metrics, weighted by difficulty:

| Metric | Weight |
|---|---|
| F1 score (accuracy of tool calls) | 60% |
| Time score (faster → higher, capped at 500ms baseline) | 15% |
| On-device ratio (higher local usage → higher score) | 25% |

**Difficulty weights:**
- Easy (single tool, direct request): 20%
- Medium (2–4 tools presented, must pick the right one): 30%
- Hard (multiple tools required, compound requests): 50%

---

## Project Structure

```
.
├── main.py          # Core function-calling logic (submit this)
├── benchmark.py     # Local benchmark runner with 30 test cases
└── submit.py        # Submission script for the Cactus Evals leaderboard
```

---

## Setup

This project requires the [Cactus](https://github.com/cactus-compute/cactus) runtime and the `functiongemma-270m-it` model weights:

```
cactus/
├── python/src/      # Cactus Python bindings
└── weights/
    └── functiongemma-270m-it/
```

For the cloud fallback, set your Gemini API key:

```bash
export GEMINI_API_KEY=your_key_here
```

---

## Running Locally

**Run the benchmark** (30 test cases across easy / medium / hard):

```bash
python benchmark.py
```

**Submit to the leaderboard:**

```bash
python submit.py --team "YourTeamName" --location "SF"
```

The submit script uploads `main.py`, queues it for evaluation on the server, and polls for results, printing your final score, F1, average latency, and on-device percentage.
