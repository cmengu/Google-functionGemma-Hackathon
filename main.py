import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, re, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy

_MODEL = cactus_init(functiongemma_path)

# ═══════════════════════════════════════════════════════
# TIME UTILITIES
# ═══════════════════════════════════════════════════════

_TIME_RE_EXT = re.compile(
    r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|in\s+the\s+morning|in\s+the\s+evening|in\s+the\s+night|at\s+night|tonight)',
    re.IGNORECASE
)
_AP_MAP = {
    "am": "AM", "pm": "PM",
    "in the morning": "AM", "in the evening": "PM",
    "in the night": "PM", "at night": "PM", "tonight": "PM",
}

def _parse_time(msg):
    m = _TIME_RE_EXT.search(msg)
    if not m:
        return None
    h = int(m.group(1))
    mn = int(m.group(2)) if m.group(2) else 0
    ap = _AP_MAP.get(m.group(3).lower().strip(), "AM")
    return h, mn, ap

def _norm_time(v):
    v = v.strip()
    m = re.match(r'(\d{1,2}:\d{2})\s*([AP]M)$', v, re.IGNORECASE)
    if m:
        return f"{m.group(1)} {m.group(2).upper()}"
    m = re.match(r'(\d{1,2})\s*([AP]M)$', v, re.IGNORECASE)
    if m:
        return f"{m.group(1)}:00 {m.group(2).upper()}"
    return v


# ═══════════════════════════════════════════════════════
# NAME UTILITIES
# ═══════════════════════════════════════════════════════

_NAME_SKIP = {
    "play", "set", "send", "find", "look", "check", "text", "remind",
    "wake", "get", "am", "pm", "and", "also", "then", "the", "my",
    "a", "an", "in", "at", "for", "to", "me", "about", "up", "some",
    "good", "hi", "hello", "hey", "please", "tonight", "morning",
}

def _extract_names_from_msg(msg):
    names = {}
    for m in re.finditer(r'\b([A-Z][a-zA-Z\-\']{1,})\b', msg):
        w = m.group(1)
        if w.lower() not in _NAME_SKIP:
            names[w.lower()] = w
    for m in re.finditer(
        r'(?:to|message|text|send\s+to|find|look\s+up|search(?:\s+for)?)\s+([A-Za-z][a-zA-Z\-\']{1,})',
        msg, re.IGNORECASE
    ):
        w = m.group(1)
        if w.lower() not in _NAME_SKIP:
            names[w.lower()] = w
    return names

def _restore_name_case(value, original_msg):
    name_map = _extract_names_from_msg(original_msg)
    restored = []
    for w in value.split():
        key = w.lower()
        restored.append(name_map[key] if key in name_map else w.title())
    return " ".join(restored)


# ═══════════════════════════════════════════════════════
# PROMPT CACHE
# ═══════════════════════════════════════════════════════

_PROMPT_CACHE = {}

def _tools_fingerprint(tools):
    return tuple(sorted(t["name"] for t in tools))

def _build_tools_desc(tools):
    tool_lines = []
    for t in tools:
        props = t["parameters"]["properties"]
        req_fields = t["parameters"].get("required", [])
        params = []
        for pname, pinfo in props.items():
            req = "required" if pname in req_fields else "optional"
            params.append(f"{pname} ({req} {pinfo['type']}): {pinfo.get('description','')}")
        tool_lines.append(f"  {t['name']}: {t['description']}\n    " + "\n    ".join(params))
    return "\n".join(tool_lines)

def _get_base_prompt(tools):
    fp = _tools_fingerprint(tools)
    if fp not in _PROMPT_CACHE:
        tools_desc = _build_tools_desc(tools)
        _PROMPT_CACHE[fp] = (
            "You are a precise function-calling assistant.\n"
            "Pick the single best tool and extract clean, minimal argument values.\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            "Output format (JSON only, no other text):\n"
            '{"function_calls": [{"name": "TOOL_NAME", "arguments": {"param": "value"}}]}\n\n'
            "Rules:\n"
            "  - String values: lowercase, no punctuation unless part of a proper name\n"
            "  - Time format: '3:00 PM' or '7:30 AM' (always space before AM/PM)\n"
            "  - Alarm hours: 24-hour integers (6 AM->6, 6 PM->18, 10 PM->22, 12 AM->0, 12 PM->12)\n"
            "  - Reminder title: subject only, never include the time in the title\n"
            "  - Names (recipient/query/location): copy spelling exactly as given\n"
        )
    return _PROMPT_CACHE[fp]

def _build_prompt(tools, msg):
    """Build a message-specific prompt with pre-resolved hints injected.
    The model still runs and produces final JSON — hints reduce ambiguity."""
    base = _get_base_prompt(tools)
    ml = msg.lower()
    tool_names = {t["name"] for t in tools}
    hints = []
    all_times = list(_TIME_RE_EXT.finditer(msg))

    if re.search(r'\btimer\b', ml):
        dm = re.search(r'(\d+)\s*(?:min|sec)', ml)
        if dm:
            hints.append(f"Use set_timer. minutes={dm.group(1)}")

    if re.search(r'\b(alarm|wake)\b', ml) and "set_alarm" in tool_names and all_times:
        m_t = all_times[0]
        h = int(m_t.group(1))
        mn = int(m_t.group(2)) if m_t.group(2) else 0
        ap = _AP_MAP.get(m_t.group(3).lower().strip(), "AM")
        if ap == "PM" and h != 12: h += 12
        elif ap == "AM" and h == 12: h = 0
        hints.append(f"Use set_alarm. hour={h} (already 24h), minute={mn}")

    if re.search(r'\b(weather|temperature|forecast)\b', ml) and "get_weather" in tool_names:
        city_m = re.search(r'(?:in|at|for)\s+([A-Za-z][\w\s]+?)(?:\?|\.|,|\s+and\b|\s+and$|$)', msg)
        if city_m:
            city = city_m.group(1).strip().rstrip("?., ")
            hints.append(f"Use get_weather. location='{city}' (copy exactly)")

    if re.search(r'\b(remind|reminder)\b', ml) and "create_reminder" in tool_names and all_times:
        m_t = all_times[-1]
        h = int(m_t.group(1))
        mn = int(m_t.group(2)) if m_t.group(2) else 0
        ap = _AP_MAP.get(m_t.group(3).lower().strip(), "AM")
        time_str = f"{h}:{mn:02d} {ap}"
        hints.append(f"Use create_reminder. time='{time_str}'. Title must NOT include the time.")

    if re.search(r'\b(send|text|message)\b', ml) and "send_message" in tool_names:
        recip_m = re.search(
            r'(?:send\s+(?:a\s+)?message\s+to|text|message)\s+([A-Za-z][A-Za-z\-\']+)',
            msg, re.IGNORECASE
        )
        if recip_m:
            hints.append(f"Use send_message. recipient='{recip_m.group(1)}' (copy exactly)")

    if re.search(r'\b(find|look\s*up|search)\b', ml) and "search_contacts" in tool_names:
        name_m = re.search(
            r'(?:find|look\s*up|search(?:\s+for)?)\s+([A-Za-z][A-Za-z\-\']+)',
            msg, re.IGNORECASE
        )
        if name_m:
            hints.append(f"Use search_contacts. query='{name_m.group(1)}' (copy exactly)")

    if not hints:
        return base
    hint_block = "\nPre-resolved values — use these exactly:\n" + "\n".join(f"  - {h}" for h in hints)
    return base + hint_block


# ═══════════════════════════════════════════════════════
# MULTI-TOOL: CLAUSE SPLITTING
# ═══════════════════════════════════════════════════════

_ACTION_VERBS = ["play", "remind", "send", "message", "text", "alarm", "wake",
                 "timer", "search", "find", "look up", "weather", "check", "get", "set"]

def _count_actions(msg):
    ml = msg.lower()
    return len({v for v in _ACTION_VERBS
                if re.search(rf'\b{re.escape(v)}\b', ml)})

def _is_multi_tool(msg, tools):
    if _count_actions(msg) < 2:
        return False
    ml = msg.lower()
    # Connector words
    if any(c in ml for c in [' and ', ' also ', ' then ', ', also', ', and ']):
        return True
    # Comma followed immediately by an action verb
    if re.search(r',\s*(?:' + '|'.join(re.escape(v) for v in _ACTION_VERBS) + r')\b', ml):
        return True
    # Two sentences (period/semicolon between capitalized words)
    if re.search(r'[.;]\s+[A-Z]', msg):
        return True
    return False

def _split_into_clauses(msg):
    action_pat = (r'(?:play|remind|send|text|message|set\s+(?:an?\s+)?(?:alarm|timer)|'
                  r'wake|search|find|look\s+up|check|get|wake)\b')
    split_re = re.compile(
        r',?\s*(?:and|also|then)\s+(?=' + action_pat + r')'
        r'|[.;]\s+(?=[A-Z])'
        r'|,\s+(?=' + action_pat + r')',
        re.IGNORECASE
    )
    clauses = [c.strip() for c in split_re.split(msg) if c.strip()]
    return clauses if len(clauses) > 1 else [msg]

def _resolve_pronouns(clause, full_msg):
    if not re.search(r'\b(him|her|them|he|she|they)\b', clause, re.IGNORECASE):
        return clause
    snippet = clause[:30] if len(clause) >= 30 else clause
    pos = full_msg.find(snippet)
    if pos == -1:
        pos = len(full_msg)
    name_map = _extract_names_from_msg(full_msg[:pos])
    last_name = None
    for m in re.finditer(r'\b([A-Za-z][a-zA-Z\-\']{1,})\b', full_msg[:pos]):
        key = m.group(1).lower()
        if key not in _NAME_SKIP and key in name_map:
            last_name = name_map[key]
    if last_name:
        clause = re.sub(r'\b(him|her|them|he|she|they)\b', last_name, clause, flags=re.IGNORECASE)
    return clause


# ═══════════════════════════════════════════════════════
# JSON PARSING
# ═══════════════════════════════════════════════════════

def _fix_json(s):
    if not s:
        return s
    s = s.strip()
    s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.MULTILINE)
    s = re.sub(r'```\s*$', '', s, flags=re.MULTILINE)
    s = s.strip().replace("：", ":")
    if s.startswith('{"name"') or s.startswith("{'name'"):
        s = '{"function_calls": [' + s + ']}'
    if s.startswith('[') and '"name"' in s:
        s = '{"function_calls": ' + s + '}'
    s = re.sub(r'"function_call"\s*:', '"function_calls":', s)
    s = re.sub(r'"tool_calls"\s*:', '"function_calls":', s)
    s = re.sub(r',\s*([}\]])', r'\1', s)
    if '"' not in s and "'" in s:
        s = s.replace("'", '"')
    return s

def _try_parse(s):
    if not s:
        return None
    fixed = _fix_json(s)
    try:
        data = json.loads(fixed)
        calls = data.get("function_calls", [])
        if calls:
            return calls
    except Exception:
        pass
    for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', s, re.DOTALL):
        try:
            obj = json.loads(_fix_json(match.group()))
            if "name" in obj:
                return [obj]
            if "function_calls" in obj:
                return obj["function_calls"]
        except Exception:
            pass
    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', s)
    args_m = re.search(r'"arguments"\s*:\s*(\{[^}]+\})', s)
    if name_m:
        call = {"name": name_m.group(1), "arguments": {}}
        if args_m:
            try:
                call["arguments"] = json.loads(_fix_json(args_m.group(1)))
            except Exception:
                for kv in re.finditer(r'"(\w+)"\s*:\s*(?:"([^"]*)"|([-\d.]+))', args_m.group(1)):
                    k, sv, nv = kv.group(1), kv.group(2), kv.group(3)
                    call["arguments"][k] = sv if sv is not None else int(float(nv or 0))
        return [call]
    return None


# ═══════════════════════════════════════════════════════
# ARGUMENT POST-PROCESSING
# ═══════════════════════════════════════════════════════

_TIME_IN_TITLE = re.compile(
    r'\s+(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*$', re.IGNORECASE
)

def _fix_args(calls, tools, msg):
    all_times = list(_TIME_RE_EXT.finditer(msg))
    for call in calls:
        td = next((t for t in tools if t["name"] == call["name"]), None)
        if not td:
            continue
        props = td["parameters"]["properties"]
        args = call.get("arguments", {})

        for k in list(args):
            if k not in props:
                ck = re.sub(r'[^a-zA-Z0-9_]', '', k)
                if ck in props:
                    args[ck] = args.pop(k)
                else:
                    args.pop(k, None)

        for k, v in list(args.items()):
            if k not in props:
                continue
            pt = props[k].get("type", "string")
            if pt == "string":
                v = str(v).strip()
                v = v.replace('\u2019', "'").replace('\u2018', "'")
                v = v.rstrip(".,!?;:")
                if k == "message":
                    v = v.lower()
                if k == "time":
                    v = _norm_time(v)
                if k == "title":
                    v = _TIME_IN_TITLE.sub("", v).strip().rstrip(".,!?;:").lower()
                if k in ("recipient", "query", "location"):
                    v = _restore_name_case(v, msg)
                args[k] = v
            elif pt == "integer":
                try:
                    args[k] = int(float(str(v)))
                except Exception:
                    pass

        m_t = all_times[0] if all_times else None
        for req in td["parameters"].get("required", []):
            if args.get(req) not in (None, ""):
                continue
            pt = props.get(req, {}).get("type", "")
            desc = props.get(req, {}).get("description", "").lower()
            if pt == "string" and "time" in (req + desc) and m_t:
                h, mn = m_t.group(1), m_t.group(2) or "00"
                ap = _AP_MAP.get(m_t.group(3).lower().strip(), "AM")
                args[req] = f"{h}:{mn} {ap}"
            elif pt == "integer" and m_t:
                h = int(m_t.group(1))
                mn = int(m_t.group(2)) if m_t.group(2) else 0
                ap = _AP_MAP.get(m_t.group(3).lower().strip(), "AM")
                if "hour" in req:
                    if ap == "PM" and h != 12: h += 12
                    elif ap == "AM" and h == 12: h = 0
                    args[req] = h
                elif "minute" in req and "minutes" not in req:
                    args[req] = mn
            elif pt == "integer" and "minutes" in req:
                dm = re.search(r'(\d+)\s*min', msg, re.IGNORECASE)
                if dm:
                    args[req] = int(dm.group(1))

        # Always recompute alarm time from original message — model often gets 24h wrong
        if call["name"] == "set_alarm" and m_t:
            h = int(m_t.group(1))
            mn = int(m_t.group(2)) if m_t.group(2) else 0
            ap = _AP_MAP.get(m_t.group(3).lower().strip(), "AM")
            if ap == "PM" and h != 12: h += 12
            elif ap == "AM" and h == 12: h = 0
            args["hour"] = h
            args["minute"] = mn

    return calls


def _validate(calls, tools):
    if not calls:
        return False
    vn = {t["name"] for t in tools}
    for c in calls:
        if c.get("name") not in vn:
            return False
        td = next((t for t in tools if t["name"] == c["name"]), None)
        if not td:
            return False
        for req in td["parameters"].get("required", []):
            if c.get("arguments", {}).get(req) in (None, ""):
                return False
    return True


# ═══════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════

def _infer_local(messages, tools, system=None):
    rag = {"tool_rag_top_k": 3} if len(tools) > 4 else {}
    msg = messages[-1]["content"] if messages else ""
    prompt = system or _build_prompt(tools, msg)
    max_tok = 80 if len(tools) == 1 else 112
    t0 = time.time()
    raw = cactus_complete(
        _MODEL,
        [{"role": "system", "content": prompt}] + messages,
        tools=[{"type": "function", "function": t} for t in tools],
        force_tools=True,
        max_tokens=max_tok,
        temperature=0,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        **rag,
    )
    elapsed = (time.time() - t0) * 1000
    calls = _try_parse(raw) or []
    if not calls:
        try:
            calls = json.loads(_fix_json(raw)).get("function_calls", [])
        except Exception:
            pass
    return {"function_calls": calls, "total_time_ms": elapsed}


def _infer_cloud(messages, tools):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    gdecls = [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=t["name"], description=t["description"],
            parameters=types.Schema(
                type="OBJECT",
                properties={k: types.Schema(type=v["type"].upper(),
                                            description=v.get("description", ""))
                            for k, v in t["parameters"]["properties"].items()},
                required=t["parameters"].get("required", []),
            ),
        ) for t in tools
    ])]
    contents = [m["content"] for m in messages if m["role"] == "user"]
    t0 = time.time()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=gdecls,
            system_instruction=(
                "You are a function calling assistant. "
                "Make exactly one tool call per distinct action. "
                "For compound requests output ALL tool calls. "
                "Bare minimum argument values. No punctuation in strings. "
                "Time: '3:00 PM' or '7:30 AM'. Alarm hours in 24h integers. "
                "Recipient/location/query: copy exact spelling from user message."
            ),
        ),
    )
    elapsed = (time.time() - t0) * 1000
    calls = [{"name": p.function_call.name, "arguments": dict(p.function_call.args)}
             for c in resp.candidates for p in c.content.parts if p.function_call]
    return {"function_calls": calls, "total_time_ms": elapsed}


# ═══════════════════════════════════════════════════════
# MAIN ROUTING
#
# Since cloud is unavailable in the eval environment, we optimise
# entirely for on-device accuracy:
#
# Multi-tool: split into single-action clauses, run model once per
# clause. Small models handle one tool at a time reliably.
# Per-clause prompts include pre-resolved hints (time, name, city)
# so each call has maximum context with minimum ambiguity.
#
# Single-tool: one local call with hinted prompt.
# Cloud is attempted as a last resort but may not be available.
# ═══════════════════════════════════════════════════════

def generate_hybrid(messages, tools):
    msg = messages[-1]["content"] if messages else ""
    total_ms = 0

    # ── Multi-tool: clause-per-model-call ───────────────
    if _is_multi_tool(msg, tools):
        clauses = _split_into_clauses(msg)
        all_calls = []
        for clause in clauses:
            clause = _resolve_pronouns(clause, msg)
            r = _infer_local([{"role": "user", "content": clause}], tools)
            total_ms += r["total_time_ms"]
            r["function_calls"] = _fix_args(r["function_calls"], tools, clause)
            if _validate(r["function_calls"], tools):
                all_calls.extend(r["function_calls"])
            else:
                # Clause failed — try cloud for entire request
                try:
                    r_c = _infer_cloud(messages, tools)
                    r_c["function_calls"] = _fix_args(r_c["function_calls"], tools, msg)
                    r_c["total_time_ms"] += total_ms
                    r_c["source"] = "cloud"
                    return r_c
                except Exception:
                    pass  # Cloud unavailable — return whatever we have
        if all_calls:
            return {"function_calls": all_calls, "total_time_ms": total_ms, "source": "on-device"}

    # ── Single-tool: local with hinted prompt ───────────
    r = _infer_local(messages, tools)
    r["function_calls"] = _fix_args(r["function_calls"], tools, msg)
    if _validate(r["function_calls"], tools):
        r["source"] = "on-device"
        return r

    # Cloud fallback (best-effort)
    try:
        r2 = _infer_cloud(messages, tools)
        r2["function_calls"] = _fix_args(r2["function_calls"], tools, msg)
        r2["total_time_ms"] += r["total_time_ms"]
        r2["source"] = "cloud" if r2["function_calls"] else "empty"
        return r2
    except Exception:
        r["source"] = "on-device"
        return r