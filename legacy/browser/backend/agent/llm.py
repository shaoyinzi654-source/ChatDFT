"""A small OpenAI-compatible chat client, and nothing more.

Why not the ``openai`` package?  It is not installed here and adding a
dependency for one POST request is not worth it.  Every provider this app
talks to (SenseNova, DeepSeek, OpenAI) exposes the same
``POST /v1/chat/completions`` shape, so this is ~120 lines instead of a
wheel.

Configuration, in priority order:

    CHATDFT_LLM_BASE / CHATDFT_LLM_KEY / CHATDFT_LLM_MODEL   (environment)
    data/llm.json                                            (project-local)

The key never belongs in the repository, so the file is the fallback and
the environment wins.  ``available`` is False when neither is set, and
every caller must handle that instead of crashing.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(ROOT, "data", "llm.json")

DEFAULT_BASE = "https://token.sensenova.cn/v1"
DEFAULT_MODEL = "sensenova-6.8-flash-lite"
# Tried in order when the primary model is rate limited or unreachable.  A
# design run that dies because one endpoint is busy is worse than a run that
# quietly produces the same answer through the other model.
DEFAULT_FALLBACKS = ["deepseek-v4-flash"]

# Fields a reply may arrive in.  Reasoning models on this endpoint put the
# answer under ``reasoning_content`` (or ``reasoning``) and leave ``content``
# null, which used to be read as "the model said nothing".
_TEXT_FIELDS = ("content", "reasoning_content", "reasoning", "text")

_RESET_RE = re.compile(r"reset at ([0-9][^\"',]*)", re.I)
_LIMIT_RE = re.compile(r"limit[^\"',]{0,40}", re.I)

# Mean seconds per successful request, per model, shared by every client in
# the process.  Two models on the same endpoint can differ by 5x for the
# same answer (measured: 103 s vs 18 s for one candidate list), so once both
# have been timed the faster one is asked first.  A model that has never
# completed a request keeps its configured place, so the primary is always
# tried before anything is reordered.
_LATENCY: Dict[str, float] = {}


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or returns nothing usable."""


class _Transient(RuntimeError):
    """Internal: this failure may succeed on another model or later."""


def _http_hint(status: int, text: str) -> str:
    """Turn a rate-limit body into something a human can act on."""
    head = (text or "").strip().replace("\n", " ")[:220]
    if status == 429:
        reset = _RESET_RE.search(head)
        hint = "429 rate limited"
        if reset:
            hint += f" (resets {reset.group(1).strip()})"
        return f"{hint}: {head}"
    return f"HTTP {status}: {head}"


def _message_text(body: Any) -> str:
    """Pull the assistant text out of a chat-completions payload.

    Handles the three shapes seen on this endpoint:

        {"choices":[{"message":{"content":"..."}}]}              normal
        {"choices":[{"message":{"content":null,
                                "reasoning_content":"..."}}]}    reasoning model
        {"choices":[{"message":{"reasoning":"..."}}]}             older reasoning
    """
    try:
        msg = body["choices"][0]["message"] or {}
    except (KeyError, IndexError, TypeError):
        msg = {}
    if not isinstance(msg, dict):
        return ""
    for field in _TEXT_FIELDS:
        value = msg.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # some gateways answer with a list of content parts
    parts = msg.get("content")
    if isinstance(parts, list):
        chunks = [p.get("text", "") for p in parts
                  if isinstance(p, dict) and isinstance(p.get("text"), str)]
        if any(chunks):
            return "".join(chunks).strip()
    return ""


def _shape(body: Any) -> str:
    """Compact description of a payload, for error messages only."""
    try:
        msg = body["choices"][0]["message"]
        keys = sorted(msg.keys()) if isinstance(msg, dict) else "?"
        return f"message keys {keys}, finish_reason " \
               f"{body['choices'][0].get('finish_reason')!r}"
    except (KeyError, IndexError, TypeError):
        return f"top keys {sorted(body.keys()) if isinstance(body, dict) else '?'}"


def _load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


class LLMClient:
    """Minimal chat-completions client with retries and JSON repair."""

    def __init__(self, base: Optional[str] = None, key: Optional[str] = None,
                 model: Optional[str] = None, timeout: float = 90.0,
                 retries: int = 2):
        cfg = _load_config()
        self.base = (base or os.environ.get("CHATDFT_LLM_BASE")
                     or cfg.get("base") or DEFAULT_BASE).rstrip("/")
        self.key = (key or os.environ.get("CHATDFT_LLM_KEY")
                    or cfg.get("key") or "")
        self.model = (model or os.environ.get("CHATDFT_LLM_MODEL")
                      or cfg.get("model") or DEFAULT_MODEL)
        self.timeout = float(cfg.get("timeout") or timeout)
        self.retries = int(cfg.get("retries") if cfg.get("retries") is not None
                           else retries)
        fallbacks = cfg.get("fallback_models")
        if not isinstance(fallbacks, list):
            fallbacks = DEFAULT_FALLBACKS
        self.fallbacks = [m for m in fallbacks
                          if isinstance(m, str) and m and m != self.model]

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self.key)

    def models(self) -> List[str]:
        order = [self.model] + list(self.fallbacks)
        timed = sorted((m for m in order if m in _LATENCY),
                       key=lambda m: _LATENCY[m])
        fresh = [m for m in order if m not in _LATENCY]
        return fresh + timed

    def describe(self) -> Dict[str, Any]:
        """Config summary with the key masked, safe to hand to the front end."""
        k = self.key
        tail = k[-4:] if len(k) > 8 else ""
        return {
            "base": self.base,
            "model": self.model,
            "models": self.models(),
            "latency": {m: round(v, 1) for m, v in _LATENCY.items()},
            "configured": bool(k),
            "key_hint": ("sk-..." + tail) if tail else "",
        }

    # ------------------------------------------------------------------
    def _post(self, model: str, payload: Dict[str, Any],
              timeout: Optional[float] = None) -> Any:
        import requests

        body = dict(payload)
        body["model"] = model
        try:
            resp = requests.post(
                f"{self.base}/chat/completions",
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"},
                json=body,
                timeout=(10, timeout or self.timeout),
            )
        except Exception as exc:                          # network / timeout
            raise _Transient(f"request failed: {exc}") from None
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _Transient(_http_hint(resp.status_code, resp.text))
        if resp.status_code != 200:
            # 400/401/404 will not fix themselves by asking again.
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError:
            raise LLMError(f"unreadable response: {resp.text[:300]}") from None

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7,
             max_tokens: int = 4000, json_mode: bool = False,
             timeout: Optional[float] = None) -> str:
        """Return the assistant's text.  Raises LLMError on failure.

        Every configured model is tried in turn; rate limits and 5xx move on
        to the next one instead of failing the whole design run.

        ``timeout`` overrides the configured per-request budget.  A caller on
        an interactive path (the chat planner) wants a shorter one than a
        design run that is already expected to take minutes.
        """
        if not self.key:
            raise LLMError("no language model configured (set CHATDFT_LLM_KEY "
                           "or data/llm.json)")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            # Some providers only honour the plain form; send both spellings.
            payload["response_format"] = {"type": "json_object"}

        problems: List[str] = []
        for model in self.models():
            for attempt in range(self.retries + 1):
                try:
                    text = self._ask(model, payload, timeout)
                except _Transient as exc:
                    problems.append(f"{model}: {exc}")
                    time.sleep(0.8 * (attempt + 1))
                    continue
                except LLMError:
                    raise
                self.last_model = model
                return text
            # exhausted this model, fall through to the next

        raise LLMError("; ".join(problems[-4:]) or "request failed")

    def _ask(self, model: str, payload: Dict[str, Any],
             timeout: Optional[float] = None) -> str:
        """One attempt at one model.  Raises _Transient if it is worth
        retrying on a different model, LLMError if it is not."""
        started = time.time()
        body = self._post(model, payload, timeout)
        text = _message_text(body)
        if not text:
            raise _Transient(f"empty reply ({_shape(body)})")
        previous = _LATENCY.get(model)
        seconds = time.time() - started
        _LATENCY[model] = seconds if previous is None else \
            0.5 * previous + 0.5 * seconds
        return text

    # ------------------------------------------------------------------
    def json(self, messages: List[Dict[str, str]], temperature: float = 0.7,
             max_tokens: int = 4000,
             validate: Optional[Any] = None,
             timeout: Optional[float] = None) -> Any:
        """Chat, then parse.  Models wrap JSON in prose and fences, so the
        extraction is defensive rather than trusting ``response_format``.

        ``validate(payload) -> bool`` lets the caller say what shape it
        needs.  A reply that parses but does not match is treated like a
        failure, so the next model gets a turn instead of the caller
        quietly receiving an empty result -- which is what happened with a
        reasoning model that spent its whole budget thinking out loud and
        never emitted the object.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        problems: List[str] = []
        for model in self.models():
            for attempt in range(self.retries + 1):
                if attempt:
                    payload["temperature"] = 0.0
                try:
                    text = self._ask(model, payload, timeout)
                except _Transient as exc:
                    problems.append(f"{model}: {exc}")
                    time.sleep(0.8 * (attempt + 1))
                    continue
                except LLMError:
                    raise
                try:
                    parsed = loads_loose(text)
                except ValueError as exc:
                    problems.append(f"{model}: {exc}")
                    continue
                if validate is not None and not validate(parsed):
                    problems.append(f"{model}: reply did not match the "
                                    "requested schema")
                    continue
                self.last_model = model
                return parsed
        raise LLMError("; ".join(problems[-4:]) or "request failed")

    # ------------------------------------------------------------------
    def probe(self) -> Dict[str, Any]:
        """Ask each configured model one trivial question.  Used by the
        smoke test and by the front end to explain which model is live."""
        out: Dict[str, Any] = {"base": self.base, "models": []}
        for model in self.models():
            entry: Dict[str, Any] = {"model": model, "ok": False}
            try:
                body = self._post(model, {
                    "model": model,
                    "messages": [{"role": "user",
                                  "content": 'Reply with exactly: {"ok": true}'}],
                    "temperature": 0, "max_tokens": 64, "stream": False,
                    "response_format": {"type": "json_object"},
                })
                text = _message_text(body)
                entry["ok"] = bool(text and "ok" in text.lower())
                entry["shape"] = _shape(body)
                entry["preview"] = (text or "")[:120]
            except (LLMError, _Transient) as exc:
                entry["error"] = str(exc)[:200]
            out["models"].append(entry)
        return out


def loads_loose(text: str) -> Any:
    """Parse JSON out of a reply that may be fenced, prefixed or chatty.

    `````json { ... }``` `` is the common case; some models also emit
    "Here is the JSON you asked for:" first.  Fall back to slicing between
    the outermost matching brackets, which survives trailing commentary.
    """
    if not text:
        raise ValueError("empty reply")
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    try:
        return json.loads(body)
    except ValueError:
        pass

    # A reasoning model writes a thousand words of chain-of-thought and then
    # the object.  Slicing between the FIRST "{" and the LAST "}" picks up
    # any brace in the prose and produces garbage, so every balanced span is
    # tried in turn and the first one that actually parses wins.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = 0
        while True:
            i = body.find(opener, start)
            if i < 0:
                break
            j = _balanced_end(body, i, opener, closer)
            if j is not None:
                try:
                    return json.loads(body[i:j + 1])
                except ValueError:
                    pass
            start = i + 1
    raise ValueError(f"no JSON object found in reply: {text[:200]!r}")


def _balanced_end(text: str, start: int, opener: str, closer: str) -> Optional[int]:
    """Index of the bracket closing the one at ``start``, or None.

    Depth counting with string and escape awareness, because a "}" inside a
    SMILES or a rationale would otherwise end the object early.
    """
    depth = 0
    in_string = False
    escaped = False
    for k in range(start, len(text)):
        ch = text[k]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return k
    return None


class StubClient:
    """Drop-in client that replays canned replies.  Used by the gates so a
    design run can be audited without a network or a key."""

    def __init__(self, replies: Any, models: Optional[List[str]] = None):
        # a list of replies, or one callable(messages) -> str
        self.replies = replies
        self.calls: List[List[Dict[str, str]]] = []
        self.model = (models or ["stub"])[0]
        self.fallbacks = list((models or ["stub"])[1:])
        self.last_model = self.model

    @property
    def available(self) -> bool:
        return True

    def models(self) -> List[str]:
        return [self.model] + list(self.fallbacks)

    def describe(self) -> Dict[str, Any]:
        return {"base": "stub", "model": self.model, "models": self.models(),
                "configured": True, "key_hint": ""}

    def chat(self, messages, temperature=0.7, max_tokens=4000, json_mode=False):
        self.calls.append(list(messages))
        if callable(self.replies):
            return self.replies(messages)
        if isinstance(self.replies, list):
            idx = min(len(self.calls) - 1, len(self.replies) - 1)
            out = self.replies[idx]
        else:
            out = self.replies
        return out if isinstance(out, str) else json.dumps(out)

    def json(self, messages, temperature=0.7, max_tokens=4000, validate=None):
        payload = loads_loose(self.chat(messages, temperature=temperature,
                                        max_tokens=max_tokens))
        if validate is not None and not validate(payload):
            raise LLMError("stub reply did not match the requested schema")
        return payload


def default_client() -> LLMClient:
    return LLMClient()


def _main(argv: Optional[List[str]] = None) -> int:
    """``python -m backend.agent.llm`` -- report which models answer."""
    import argparse

    ap = argparse.ArgumentParser(description="probe the language model endpoint")
    ap.add_argument("--request", help="send this prompt instead of the ping")
    ap.add_argument("--model", help="override the primary model")
    args = ap.parse_args(argv)

    client = default_client()
    if args.model:
        client.model = args.model
    print("config:", client.describe())
    if not client.available:
        print("no key configured")
        return 1

    if args.request:
        try:
            print("reply:", client.chat([{"role": "user",
                                          "content": args.request}]))
        except LLMError as exc:
            print("ERROR:", exc)
            return 1
        return 0

    info = client.probe()
    bad = 0
    for entry in info["models"]:
        if entry["ok"]:
            print(f"  OK    {entry['model']:<28} {entry.get('shape','')}")
            print(f"        {entry.get('preview','')!r}")
        else:
            bad += 1
            print(f"  FAIL  {entry['model']:<28} {entry.get('error','')}")
    return 1 if bad == len(info["models"]) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
