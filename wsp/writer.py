"""Writer-model client with an append-only request ledger.

Transports
  * ``responses``: the OpenAI Responses API over HTTP in background mode (create -> poll -> cancel).
  * ``anthropic``: the Anthropic Messages API through the ``anthropic`` SDK, streaming.

Configuration (see README "Writer configuration")
  * Keys: ``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY``.
  * Model ids and endpoints default to the vendors' public ones (``MODELS`` below). Optional overrides,
    as environment variables or in a JSON file (``$WSP_WRITER_CONFIG`` or ``writer_config.local.json``
    in the repository root; ``writer_config.example.json`` documents the format):
      WSP_MODEL_ID_<KEY>         model id for a writer key, e.g. WSP_MODEL_ID_ASTRA_6
      WSP_RESPONSES_BASE_URL     base URL of the Responses API (``/responses`` is appended)
      WSP_ANTHROPIC_BASE_URL     base URL of the Messages API

Rules
  * A ledger line is written BEFORE every submission.
  * Automatic retry only when the request provably never started (HTTP error before an id/stream).
  * A request whose fate is unknown is recorded as `unknown` and never resubmitted automatically.
  * A background response whose id never becomes retrievable (HTTP 404 on every poll) counts as not
    started and is resubmitted, at most three submissions in total.
  * Every turn is stateless: the full visible transcript is resent; no reasoning is carried over.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from .manifest import ROOT, RUNS, sha256_bytes

LEDGER = RUNS / "ledger.jsonl"
CONFIG_ENV = "WSP_WRITER_CONFIG"
DEFAULT_CONFIG = ROOT / "writer_config.local.json"
_lock = threading.Lock()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    transport: str          # responses | anthropic
    label: str              # name used in the paper
    api_id: str             # the vendor's model id
    effort: str = "high"
    max_output_tokens: int = 128_000

    @property
    def model_id(self) -> str:
        """The model id sent to the API: ``api_id`` unless overridden (see module docstring)."""
        env = "WSP_MODEL_ID_" + self.key.upper().replace("-", "_").replace(".", "_")
        return os.environ.get(env) or _config().get("models", {}).get(self.key, {}).get("model_id") or self.api_id


MODELS = {m.key: m for m in [
    ModelSpec("astra-6", "responses", "GPT-6 Astra", "gpt-6-astra"),
    ModelSpec("sol-6.1", "responses", "GPT-6.1 Sol", "gpt-6.1-sol"),
    ModelSpec("opus-5.5", "anthropic", "Claude Opus 5.5", "claude-opus-5-5"),
    ModelSpec("terra-6.1", "responses", "GPT-6.1 Terra", "gpt-6.1-terra"),
    ModelSpec("luna-6", "responses", "GPT-6 Luna", "gpt-6-luna"),
    ModelSpec("sonnet-5.5", "anthropic", "Claude Sonnet 5.5", "claude-sonnet-5-5"),
    ModelSpec("haiku-5.5", "anthropic", "Claude Haiku 5.5", "claude-haiku-5-5"),
    ModelSpec("sol-5.6", "responses", "GPT-5.6 Sol", "gpt-5.6-sol"),
    ModelSpec("opus-5", "anthropic", "Claude Opus 5", "claude-opus-5"),
]}


@lru_cache(maxsize=1)
def _config() -> dict:
    path = Path(os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG)
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _transport_cfg(name: str) -> dict:
    return dict(_config().get("transports", {}).get(name, {}))


@dataclass
class CallResult:
    call_id: str
    model: str
    status: str                      # completed | incomplete | refusal | error | unknown
    text: str = ""
    usage: dict = field(default_factory=dict)
    response_id: str | None = None
    wall_s: float = 0.0
    detail: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def ledger(event: str, **kw) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    rec = {"t": time.time(), "event": event, **kw}
    with _lock, open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- Responses API
RESPONSES_BASE_URL = "https://api.openai.com/v1"


def _responses_base() -> str:
    base = (os.environ.get("WSP_RESPONSES_BASE_URL") or _transport_cfg("responses").get("base_url")
            or RESPONSES_BASE_URL)
    return base.rstrip("/")


def _request(method: str, path: str, body: dict | None, timeout: float = 120):
    """One HTTP request to the Responses API."""
    url = _responses_base() + path
    data = None if body is None else json.dumps(body).encode()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    headers = {"content-type": "application/json", "authorization": f"Bearer {key}"}
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:2000]}


def _responses_text(d: dict) -> str:
    out = []
    for item in d.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    out.append(c.get("text", ""))
    return "".join(out)


def _call_responses(spec: ModelSpec, messages: list[dict], call_id: str, effort: str,
                    max_tokens: int, wall_limit: float) -> CallResult:
    body = {"model": spec.model_id, "input": messages, "max_output_tokens": max_tokens,
            "reasoning": {"effort": effort}, "tools": [], "store": True, "background": True}   # no tools
    for lost_try in range(3):         # a response whose id never becomes retrievable is resubmitted
        res = _call_responses_once(spec, body, call_id, wall_limit)
        if res is not None:
            return res
        ledger("lost", call_id=call_id, lost_try=lost_try)
    return CallResult(call_id, spec.key, "error", detail="response not retrievable (HTTP 404) after three submissions")


def _call_responses_once(spec: ModelSpec, body: dict, call_id: str, wall_limit: float):
    """One submission. Returns None if the response id never became retrievable (>= LOST_404 polls
    with HTTP 404 before any successful poll); otherwise a CallResult."""
    LOST_404 = 20
    t0 = time.time()
    for attempt in range(90):         # create: retry only on errors that prove no response exists
        try:
            st, d = _request("POST", "/responses", body)
        except Exception as exc:      # network failure on create: fate unknown
            ledger("unknown", call_id=call_id, phase="create", error=repr(exc)[:500])
            return CallResult(call_id, spec.key, "unknown", detail=f"create transport error: {exc!r}"[:500])
        if st == 200:
            break
        ledger("create_error", call_id=call_id, status=st, error=str(d)[:500], attempt=attempt)
        if st not in (429, 500, 502, 503, 504):
            return CallResult(call_id, spec.key, "error", detail=f"HTTP {st}: {d}"[:1000])
        time.sleep(min(90, 15 * 2 ** attempt))
    else:
        return CallResult(call_id, spec.key, "error", detail="create retries exhausted")
    rid = d["id"]
    ledger("created", call_id=call_id, response_id=rid)
    status = d.get("status")
    n404, seen = 0, False
    while status in ("queued", "in_progress"):
        if time.time() - t0 > wall_limit:
            try:
                _request("POST", f"/responses/{rid}/cancel", {})
            except Exception:
                pass
            ledger("cancelled", call_id=call_id, response_id=rid, reason="wall limit")
            return CallResult(call_id, spec.key, "error", response_id=rid, wall_s=time.time() - t0,
                              detail=f"cancelled after wall limit {wall_limit}s")
        time.sleep(15)
        try:
            st, d2 = _request("GET", f"/responses/{rid}", None)
        except Exception:
            continue                  # transient poll failure; the remote request keeps running
        if st != 200:
            ledger("poll_error", call_id=call_id, response_id=rid, status=st)
            n404 += int(st == 404)
            if n404 >= LOST_404 and not seen:
                try:
                    _request("POST", f"/responses/{rid}/cancel", {})
                except Exception:
                    pass
                return None
            time.sleep(30)
            continue
        n404, seen = 0, True
        d, status = d2, d2.get("status")
    usage = d.get("usage") or {}
    u = {"input": usage.get("input_tokens"), "output": usage.get("output_tokens"),
         "reasoning": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
         "cached": (usage.get("input_tokens_details") or {}).get("cached_tokens")}
    st_map = {"completed": "completed", "incomplete": "incomplete", "failed": "error",
              "cancelled": "error"}
    detail = None
    if status == "incomplete":
        detail = json.dumps(d.get("incomplete_details"))
    elif status in ("failed", "cancelled"):
        detail = json.dumps(d.get("error"))[:1000]
    return CallResult(call_id, spec.key, st_map.get(status, "unknown"), _responses_text(d), u, rid,
                      time.time() - t0, detail)


# --------------------------------------------------------------------------- Anthropic Messages API
@lru_cache(maxsize=1)
def _anthropic_client():
    """The Messages API client (key from ANTHROPIC_API_KEY); SDK-internal retries are off, retries are
    decided here."""
    from anthropic import Anthropic
    kw = {}
    base = os.environ.get("WSP_ANTHROPIC_BASE_URL") or _transport_cfg("anthropic").get("base_url")
    if base:
        kw["base_url"] = base
    return Anthropic(max_retries=0, timeout=7200.0, **kw)


def _call_anthropic(spec: ModelSpec, messages: list[dict], call_id: str, effort: str,
                    max_tokens: int, wall_limit: float, task_budget: int | None = None) -> CallResult:
    import anthropic
    t0 = time.time()
    client = _anthropic_client()
    for attempt in range(90):
        started = False
        try:
            oc = {"effort": effort}
            kw = {}
            api = client.messages
            if task_budget:
                oc["task_budget"] = {"type": "tokens", "total": task_budget}
                kw["betas"] = ["task-budgets-2026-03-13"]
                api = client.beta.messages
            with api.stream(model=spec.model_id, max_tokens=max_tokens,
                            thinking={"type": "adaptive"}, output_config=oc,
                            messages=messages, **kw) as stream:
                started = True
                ledger("created", call_id=call_id, attempt=attempt)
                box: dict = {}

                def _consume():
                    try:
                        box["msg"] = stream.get_final_message()
                    except Exception as e:          # re-raised in the caller thread
                        box["exc"] = e
                worker = threading.Thread(target=_consume, daemon=True)
                worker.start()
                worker.join(timeout=max(1.0, wall_limit - (time.time() - t0)))
                if worker.is_alive():               # enforce the wall limit, never resubmit
                    try:
                        stream.close()
                    except Exception:
                        pass
                    ledger("unknown", call_id=call_id, phase="wall_limit", wall_s=round(time.time() - t0, 1))
                    return CallResult(call_id, spec.key, "unknown", wall_s=time.time() - t0,
                                      detail=f"wall limit {wall_limit}s exceeded; stream closed; not resubmitted")
                if "exc" in box:
                    raise box["exc"]
                msg = box["msg"]
            break
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIConnectionError) as exc:
            if started:
                ledger("unknown", call_id=call_id, phase="stream", error=repr(exc)[:500])
                return CallResult(call_id, spec.key, "unknown", wall_s=time.time() - t0,
                                  detail=f"stream interrupted: {exc!r}"[:500])
            ledger("create_error", call_id=call_id, error=repr(exc)[:500], attempt=attempt)
            time.sleep(min(90, 15 * 2 ** attempt))
        except anthropic.APIStatusError as exc:
            if started:
                ledger("unknown", call_id=call_id, phase="stream", error=repr(exc)[:500])
                return CallResult(call_id, spec.key, "unknown", wall_s=time.time() - t0,
                                  detail=f"stream error: {exc!r}"[:500])
            return CallResult(call_id, spec.key, "error", detail=repr(exc)[:1000])
    else:
        return CallResult(call_id, spec.key, "error", detail="create retries exhausted")
    text = "".join(b.text for b in msg.content if b.type == "text")
    u = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens,
         "cached": getattr(msg.usage, "cache_read_input_tokens", None)}
    status = {"end_turn": "completed", "max_tokens": "incomplete", "refusal": "refusal"}.get(
        msg.stop_reason, "completed" if text else "incomplete")
    detail = None
    if msg.stop_reason == "refusal":
        detail = json.dumps(getattr(msg, "stop_details", None), default=str)[:500]
    elif msg.stop_reason not in ("end_turn",):
        detail = f"stop_reason={msg.stop_reason}"
    return CallResult(call_id, spec.key, status, text, u, msg.id, time.time() - t0, detail)


# --------------------------------------------------------------------------- public API
def call(model_key: str, messages: list[dict], tag: str, out_dir: Path | None = None,
         effort: str | None = None, max_tokens: int | None = None,
         wall_limit: float = 5400, task_budget: int | None = None) -> CallResult:
    """messages: [{'role': 'user'|'assistant', 'content': str}, ...] (visible text only)."""
    spec = MODELS[model_key]
    effort = effort or spec.effort
    if effort not in ("low", "medium", "high", "xhigh"):
        raise ValueError(f"effort must be one of low, medium, high, xhigh, got {effort!r}")
    max_tokens = max_tokens or spec.max_output_tokens
    model_id = spec.model_id
    key_env = {"responses": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[spec.transport]
    if not os.environ.get(key_env):
        raise RuntimeError(f"{key_env} is not set")           # before the ledger line
    call_id = uuid.uuid4().hex[:12]
    payload = json.dumps(messages, sort_keys=True).encode()
    ledger("submit", call_id=call_id, tag=tag, model=model_key, model_id=model_id,
           transport=spec.transport, effort=effort, max_tokens=max_tokens,
           prompt_sha256=sha256_bytes(payload), input_chars=sum(len(m["content"]) for m in messages),
           task_budget=task_budget)
    try:
        if spec.transport == "responses":
            res = _call_responses(spec, messages, call_id, effort, max_tokens, wall_limit)
        elif spec.transport == "anthropic":
            res = _call_anthropic(spec, messages, call_id, effort, max_tokens, wall_limit, task_budget)
        else:
            raise ValueError(f"unknown transport {spec.transport!r}")
    except Exception as exc:
        res = CallResult(call_id, model_key, "unknown", detail=f"client exception: {exc!r}"[:500])
    ledger("finish", call_id=call_id, tag=tag, model=model_key, status=res.status, usage=res.usage,
           response_id=res.response_id, wall_s=round(res.wall_s, 1), detail=res.detail)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{tag}.call.json").write_text(json.dumps(
            {"request": {"model": model_id, "effort": effort, "max_tokens": max_tokens,
                         "messages": messages}, "result": res.to_json()}, indent=1))
    return res


def extract_code(text: str) -> str | None:
    """Return the last ```python fenced block (or last fenced block) in a response."""
    import re
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.S)
    return blocks[-1] if blocks else None
