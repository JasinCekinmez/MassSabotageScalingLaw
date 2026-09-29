"""Unified LLM completion layer for OpenAI, xAI, Anthropic and Gemini.

- Routes on model-name prefix.
- Rotates every API key found in .env for the provider (round-robin).
- Per-provider concurrency caps so a big N x rounds fan-out does not trip rate limits.
- Retries with backoff; raises LLMError after exhausting attempts.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import openai  # noqa: F401  (forces the openai/httpx import chain to resolve once, in the main
# thread, before any worker threads exist -- without this, a fresh process that fans out many
# threads at once can race on the first `from openai import OpenAI` and hit a circular-import
# AttributeError on httpx.URL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")


# ---------------------------------------------------------------- keys
def load_env(path: str = ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    if not os.path.exists(path):
        return env
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip().lower()] = v.strip().strip('"').strip("'")
    return env


_ENV = load_env()


def _keys(prefix: str) -> list[str]:
    return [v for k, v in _ENV.items() if k.startswith(prefix) and v]


class KeyRing:
    def __init__(self, ks: list[str]):
        self._ks = list(ks)
        self._i = 0
        self._lock = threading.Lock()
        self._cool: dict[str, float] = {}

    def __bool__(self):
        return bool(self._ks)

    def __len__(self):
        return len(self._ks)

    def next(self) -> str:
        """Round-robin over keys that are not cooling down; if every key is cooling, hand out the one whose
        cooldown ends soonest (the caller will hit the quota error again and wait)."""
        with self._lock:
            if not self._ks:
                raise LLMError("no working API keys left for this provider")
            now = time.time()
            for _ in range(len(self._ks)):
                k = self._ks[self._i % len(self._ks)]
                self._i += 1
                if self._cool.get(k, 0) <= now:
                    return k
            return min(self._ks, key=lambda k: self._cool.get(k, 0))

    def cooldown(self, key: str, seconds: float, reason: str = "") -> None:
        """Skip a key for a while (e.g. its daily quota is exhausted); it rejoins rotation automatically."""
        with self._lock:
            if key in self._ks and self._cool.get(key, 0) < time.time() + seconds:
                self._cool[key] = time.time() + seconds
                n = sum(1 for k in self._ks if self._cool.get(k, 0) > time.time())
                print(f"[llm] cooling API key ...{key[-6:]} for {seconds // 60:.0f} min: {reason} ({len(self._ks) - n} keys usable)", flush=True)

    def available(self) -> bool:
        with self._lock:
            now = time.time()
            return any(self._cool.get(k, 0) <= now for k in self._ks)

    def disable(self, key: str, reason: str = "") -> None:
        """Drop a key from rotation (e.g. credit balance exhausted)."""
        with self._lock:
            if key in self._ks:
                self._ks.remove(key)
                print(f"[llm] disabled API key ...{key[-6:]}: {reason} ({len(self._ks)} left)", flush=True)


EXHAUSTED_MARKERS = ("credit_balance_exhausted", "insufficient_quota", "no credits remaining", "credit balance is too low",
                     "insufficient credits", "error code: 402", "key limit exceeded")  # OpenRouter: balance or per-key limit
QUOTA_MARKERS = ("resource_exhausted", "resource has been exhausted")          # Gemini daily/rate quota
BUDGET_MARKERS = ("exceededbudget", "budget has been exceeded", "exceeded budget", "max budget")  # LiteLLM key budget
BUDGET_WAIT = 300          # seconds between retries while a budget is exhausted
LITELLM_SPEND_CAP = float(os.environ.get("LITELLM_SPEND_CAP", "650"))   # our own daily ceiling on proxy spend ($)
_cap_state = {"checked": 0.0, "over": False, "spend": None}


def litellm_spend() -> float | None:
    """Today's spend on the proxy key from its budget endpoint, or None if unreachable."""
    try:
        import urllib.request, json as _json
        req = urllib.request.Request(LITELLM_BASE_URL.rsplit("/v1", 1)[0] + "/usage/self/budget",
                                     headers={"Authorization": f"Bearer {RINGS['litellm'].next()}", "User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return float(_json.load(r)["spend"])
    except Exception:  # noqa: BLE001
        return None


def litellm_over_cap() -> bool:
    """True when today's proxy spend has reached LITELLM_SPEND_CAP (checked at most once a minute).
    Unreachable endpoint counts as under the cap so a proxy hiccup does not stall runs."""
    now = time.time()
    if now - _cap_state["checked"] > 60:
        _cap_state["checked"] = now
        sp = litellm_spend()
        if sp is not None:
            _cap_state["spend"] = sp
            over = sp >= LITELLM_SPEND_CAP
            if over and not _cap_state["over"]:
                print(f"[llm] LiteLLM daily spend ${sp:.2f} reached our cap of ${LITELLM_SPEND_CAP:.0f}; proxy calls held until the reset", flush=True)
            _cap_state["over"] = over
    return _cap_state["over"]
QUOTA_COOLDOWN = 1800      # seconds a direct key sits out after a quota (RESOURCE_EXHAUSTED) error
BUDGET_MAX_WAIT = 12 * 3600


def is_exhausted(err: Exception) -> bool:
    m = str(err).lower()
    return any(x in m for x in EXHAUSTED_MARKERS)


def is_quota(err: Exception) -> bool:
    return any(x in str(err).lower() for x in QUOTA_MARKERS)


def is_budget(err: Exception) -> bool:
    return any(x in str(err).lower() for x in BUDGET_MARKERS)


PROVIDER_KEY_PREFIX = {
    "openai": "openai_api_key",
    "xai": "xai_api_key",
    "anthropic": "anthropic_api_key",
    "gemini": "gemini_api_key",
    "litellm": "litellm_api_key",      # OpenAI-compatible proxy; base url from LITELLM_BASE_URL
    "together": "together_api_key",    # Together serverless (OpenAI-compatible); model names "together/<org>/<model>"
    "openrouter": "openrouter_api_key",  # OpenRouter (OpenAI-compatible); model names "openrouter/<vendor>/<model>"
}
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Route override: run a provider's models through another provider that serves the same weights, keeping the
# canonical model name (and therefore game tags) unchanged. Set SABOTAGE_ROUTE_OVERRIDE="together=openrouter".
# "together=openrouter:0.5" sends only that share of calls to the other route (deterministic interleave) and the rest direct.
ROUTE_OVERRIDE: dict[str, str] = {}
OVERRIDE_SHARE: dict[str, float] = {}
for _kv in os.environ.get("SABOTAGE_ROUTE_OVERRIDE", "").split(","):
    if "=" in _kv:
        _src, _dst = _kv.split("=", 1)
        _dst, _, _sh = _dst.partition(":")
        ROUTE_OVERRIDE[_src] = _dst; OVERRIDE_SHARE[_src] = float(_sh) if _sh else 1.0
_OVERRIDE_N: dict[str, int] = {}
_OVERRIDE_LOCK = threading.Lock()


def _take_override(prov: str) -> bool:
    share = OVERRIDE_SHARE.get(prov, 1.0)
    if share >= 1: return True
    with _OVERRIDE_LOCK:
        n = _OVERRIDE_N.get(prov, 0); _OVERRIDE_N[prov] = n + 1
    return int((n + 1) * share) > int(n * share)
OVERRIDE_ALIAS = {("openrouter", "together/deepseek-ai/DeepSeek-V4.1-Flash"): "deepseek/deepseek-v4.1-flash",
                  ("openrouter", "together/meta-models/Muse-Glimmer-30B"): "meta/muse-glimmer-30b",
                  ("litellm", "grok-4.3"): "openrouter/x-ai/grok-4.3",  # xAI team hit its monthly cap Sept 20; same weights via the proxy's own OpenRouter integration
                  ("openrouter", "together/deepseek-ai/DeepSeek-V4-Flash-0731"): "deepseek/deepseek-v4-flash-0731"}
LITELLM_BASE_URL = _ENV.get("litellm_base_url") or _ENV.get("litellm_api_base") or ""
RINGS = {p: KeyRing(_keys(pre)) for p, pre in PROVIDER_KEY_PREFIX.items()}

# concurrency caps; xAI is slow per call so it gets the most parallelism
PROVIDER_CONCURRENCY = {"openai": 16, "xai": 16, "anthropic": 8, "gemini": 24, "litellm": 16, "together": 16, "openrouter": 16}
_SEMS = {p: threading.Semaphore(n) for p, n in PROVIDER_CONCURRENCY.items()}


def set_concurrency(provider: str, n: int) -> None:
    PROVIDER_CONCURRENCY[provider] = n
    _SEMS[provider] = threading.Semaphore(n)


def provider_for(model: str) -> str:
    m = model.lower()
    if m.startswith("together/"):
        return "together"
    if m.startswith("openrouter/"):
        return "openrouter"
    if m.startswith("litellm/"):      # proxy-native model name, e.g. "litellm/muse-spark"
        return "litellm"
    if "/" in m:                      # e.g. "gemini/gemini-3.8-flash": routed through the LiteLLM proxy
        return "litellm"
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("grok"):
        return "xai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gemini"
    raise LLMError(f"cannot infer provider for model {model!r}")


# ---------------------------------------------------------------- types
class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    stop_reason: str | None = None
    refused: bool = False
    seconds: float = 0.0
    attempts: int = 1
    length_retries: int = 0
    max_tokens: int = 0
    route: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text, "model": self.model, "usage": self.usage,
            "stop_reason": self.stop_reason, "refused": self.refused,
            "seconds": round(self.seconds, 1), "attempts": self.attempts,
            "length_retries": self.length_retries, "max_tokens": self.max_tokens,
            "route": self.route,
        }


TIMEOUT = 900  # seconds per request

# Optional route splitting: send a share of a direct provider's calls through the LiteLLM proxy
# (same model, proxy name prefix), alternating deterministically. {"gemini": 0.5} = every other call.
ROUTE_SPLIT: dict[str, float] = {}
_route_counter: dict[str, int] = {}
_route_lock = threading.Lock()
PROXY_PREFIX = {"gemini": "gemini/"}
# same model published under a different id on the proxy (route split rewrites the id, transcripts keep the canonical name)
PROXY_ALIAS = {"together/deepseek-ai/DeepSeek-V4-Flash-0731": "openrouter/deepseek/deepseek-v4-flash"}


def proxy_name(base_prov: str, model: str) -> str:
    return PROXY_ALIAS.get(model) or (PROXY_PREFIX.get(base_prov, "") + model)


def set_route_split(provider: str, share: float) -> None:
    if share > 0 and not (LITELLM_BASE_URL and RINGS["litellm"]):
        raise LLMError("route split requested but LITELLM_API_KEY/LITELLM_BASE_URL are not set")
    ROUTE_SPLIT[provider] = share


def _pick_route(prov: str) -> str:
    share = ROUTE_SPLIT.get(prov, 0)
    if share <= 0:
        return prov
    with _route_lock:
        n = _route_counter.get(prov, 0); _route_counter[prov] = n + 1
    # deterministic interleave: call i goes to the proxy iff floor(i*share) advanced
    return "litellm" if int((n + 1) * share) > int(n * share) else prov
EFFORTS = ("low", "medium", "high")


# ---------------------------------------------------------------- providers
def _openai(model, system, prompt, effort, max_tokens, key) -> Completion:
    from openai import OpenAI
    c = OpenAI(api_key=key, timeout=TIMEOUT, max_retries=0)
    kw = dict(model=model, input=prompt, max_output_tokens=max_tokens)
    if system:
        kw["instructions"] = system
    if effort:
        kw["reasoning"] = {"effort": effort}
    r = c.responses.create(**kw)
    refused = any(getattr(item, "type", "") == "refusal" for item in getattr(r, "output", []) or [])
    return Completion(text=r.output_text or "", model=model, usage=r.usage.model_dump() if r.usage else {},
                      stop_reason=getattr(r, "status", None), refused=refused)


def _xai(model, system, prompt, effort, max_tokens, key) -> Completion:
    from openai import OpenAI
    c = OpenAI(api_key=key, base_url="https://api.x.ai/v1", timeout=TIMEOUT, max_retries=0)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    r = c.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens)
    ch = r.choices[0]
    return Completion(text=ch.message.content or "", model=model, usage=r.usage.model_dump() if r.usage else {},
                      stop_reason=ch.finish_reason, refused=(ch.finish_reason == "content_filter"))


def _anthropic(model, system, prompt, effort, max_tokens, key) -> Completion:
    import anthropic
    c = anthropic.Anthropic(api_key=key, timeout=TIMEOUT, max_retries=0)
    kw = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}],
              thinking={"type": "adaptive"})
    if system:
        kw["system"] = system
    if effort:
        kw["output_config"] = {"effort": effort}
    with c.messages.stream(**kw) as s:
        m = s.get_final_message()
    text = "".join(b.text for b in m.content if b.type == "text")
    return Completion(text=text, model=model, usage=m.usage.model_dump(), stop_reason=m.stop_reason,
                      refused=(m.stop_reason == "refusal"))


def _gemini(model, system, prompt, effort, max_tokens, key) -> Completion:
    from google import genai
    from google.genai import types
    c = genai.Client(api_key=key, http_options={"timeout": TIMEOUT * 1000})
    cfg = types.GenerateContentConfig(system_instruction=system or None, max_output_tokens=max_tokens)
    r = c.models.generate_content(model=model, contents=prompt, config=cfg)
    usage = r.usage_metadata.model_dump() if r.usage_metadata else {}
    finish = None
    refused = False
    if r.candidates:
        finish = str(r.candidates[0].finish_reason)
        refused = "SAFETY" in finish or "PROHIBITED" in finish
    elif getattr(r, "prompt_feedback", None) and getattr(r.prompt_feedback, "block_reason", None):
        refused, finish = True, str(r.prompt_feedback.block_reason)
    return Completion(text=r.text or "", model=model, usage=usage, stop_reason=finish, refused=refused)


def _litellm(model, system, prompt, effort, max_tokens, key) -> Completion:
    """OpenAI-compatible chat completions against the LiteLLM proxy (LITELLM_BASE_URL).
    `effort` is not forwarded, matching the direct Gemini path which uses the model's default thinking."""
    from openai import OpenAI
    c = OpenAI(api_key=key, base_url=LITELLM_BASE_URL, timeout=TIMEOUT, max_retries=0)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    proxy_model = model.split("/", 1)[1] if model.startswith("litellm/") else model
    # stream so long reasoning calls do not hit the proxy's ~100s edge timeout (Cloudflare 524)
    text = []; finish = None; u = None
    with c.chat.completions.create(model=proxy_model, messages=msgs, max_completion_tokens=max_tokens,
                                   stream=True, stream_options={"include_usage": True}) as stream:
        for chunk in stream:
            if chunk.choices:
                d = chunk.choices[0].delta
                if d and d.content:
                    text.append(d.content)
                if chunk.choices[0].finish_reason:
                    finish = chunk.choices[0].finish_reason
            if getattr(chunk, "usage", None):
                u = chunk.usage
    det = getattr(u, "completion_tokens_details", None) if u else None
    usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens,
             "completion_tokens_details": {"reasoning_tokens": getattr(det, "reasoning_tokens", None)}} if u else {}
    return Completion(text="".join(text), model=model, usage=usage, stop_reason=finish,
                      refused=(finish == "content_filter"))


def _together(model, system, prompt, effort, max_tokens, key) -> Completion:
    """Together serverless via its OpenAI-compatible API. Model id = everything after 'together/'."""
    from openai import OpenAI
    c = OpenAI(api_key=key, base_url=TOGETHER_BASE_URL, timeout=TIMEOUT, max_retries=0)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    r = c.chat.completions.create(model=model.split("/", 1)[1], messages=msgs, max_tokens=max_tokens)
    ch = r.choices[0]; u = r.usage
    det = getattr(u, "completion_tokens_details", None) if u else None
    usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens,
             "completion_tokens_details": {"reasoning_tokens": getattr(det, "reasoning_tokens", None)}} if u else {}
    return Completion(text=ch.message.content or "", model=model, usage=usage, stop_reason=ch.finish_reason,
                      refused=(ch.finish_reason == "content_filter"))


def _openrouter(model, system, prompt, effort, max_tokens, key) -> Completion:
    """OpenRouter via its OpenAI-compatible API. Model id = everything after 'openrouter/' (or an override alias)."""
    from openai import OpenAI
    c = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL, timeout=TIMEOUT, max_retries=0,
               default_headers={"HTTP-Referer": "https://github.com/JasinCekinmez/MassSabotageScalingLaw", "X-Title": "MassSabotageScalingLaw"})
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    mid = model.split("/", 1)[1] if model.startswith("openrouter/") else model
    r = c.chat.completions.create(model=mid, messages=msgs, max_tokens=max_tokens, extra_body={"usage": {"include": True}})
    ch = r.choices[0]; u = r.usage
    det = getattr(u, "completion_tokens_details", None) if u else None
    usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens,
             "completion_tokens_details": {"reasoning_tokens": getattr(det, "reasoning_tokens", None)},
             "cost_usd": getattr(u, "cost", None)} if u else {}
    return Completion(text=ch.message.content or "", model=model, usage=usage, stop_reason=ch.finish_reason,
                      refused=(ch.finish_reason == "content_filter"))


_CALLERS = {"openai": _openai, "xai": _xai, "anthropic": _anthropic, "gemini": _gemini, "litellm": _litellm, "together": _together,
            "openrouter": _openrouter}


# ---------------------------------------------------------------- public API
LENGTH_STOPS = ("max_tokens", "length", "max_output", "incomplete")
MAX_TOKENS_CAP = 64000


def _hit_length(c: Completion) -> bool:
    return any(s in str(c.stop_reason or "").lower() for s in LENGTH_STOPS)


def complete(model: str, prompt: str, system: str | None = None, effort: str | None = "medium",
             max_tokens: int = 32000, attempts: int = 5) -> Completion:
    """Single-turn completion. Blocks on the provider semaphore; retries transient errors.

    If the model stops on length (thinking or text exhausted `max_tokens`), the call is retried
    with double the budget, up to MAX_TOKENS_CAP, so no stage ends up empty or cut off."""
    base_prov = provider_for(model)
    if effort is not None and effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS} or None")
    if base_prov in ROUTE_OVERRIDE and (ROUTE_OVERRIDE[base_prov], model) in OVERRIDE_ALIAS and RINGS[ROUTE_OVERRIDE[base_prov]] \
            and _take_override(base_prov):
        # same weights served elsewhere: swap provider + model id, keep the canonical name in transcripts
        canonical = model
        base_prov = ROUTE_OVERRIDE[base_prov]; model = OVERRIDE_ALIAS[(base_prov, canonical)]
        try:
            return _complete_inner(canonical, model, base_prov, system, prompt, effort, max_tokens, attempts)
        except LLMError as e:
            if "no working API keys" not in str(e):
                raise
            # the other route ran out of credit mid-call: finish this call on the direct route
            base_prov = provider_for(canonical); model = canonical
    return _complete_inner(model, model, base_prov, system, prompt, effort, max_tokens, attempts)


def _complete_inner(canonical: str, model: str, base_prov: str, system, prompt, effort, max_tokens, attempts) -> Completion:
    route = _pick_route(base_prov)
    if route == "litellm" and base_prov != "litellm" and litellm_over_cap():
        route = base_prov          # split traffic: stay on the direct keys while the proxy is at our cap
    prov = route
    if prov == "litellm":
        waited = 0
        while litellm_over_cap() and waited < BUDGET_MAX_WAIT:
            print(f"[llm] proxy at our daily cap; waiting {BUDGET_WAIT}s (waited {waited // 60} min)", flush=True)
            time.sleep(BUDGET_WAIT); waited += BUDGET_WAIT
    call_model = proxy_name(base_prov, model) if route == "litellm" and base_prov != "litellm" else model
    last: Exception | None = None
    t0 = time.time()
    with _SEMS[prov]:
        i = -1
        while i + 1 < attempts:
            i += 1
            key = RINGS[prov].next()
            try:
                out = _CALLERS[prov](call_model, system, prompt, effort, max_tokens, key)
                length_retries = 0
                while _hit_length(out) and max_tokens < MAX_TOKENS_CAP:
                    max_tokens = min(MAX_TOKENS_CAP, max_tokens * 2)
                    length_retries += 1
                    out = _CALLERS[prov](call_model, system, prompt, effort, max_tokens, key)
                out.model = canonical             # keep the canonical model name in transcripts
                out.route = prov
                out.length_retries = length_retries
                out.max_tokens = max_tokens
                out.seconds = time.time() - t0
                out.attempts = i + 1
                return out
            except Exception as e:  # noqa: BLE001 - provider SDKs raise many types
                last = e
                if is_exhausted(e):
                    RINGS[prov].disable(key, "credits exhausted")
                    continue          # retry immediately on the next key
                if is_quota(e) and prov != "litellm":
                    # direct provider quota hit: rest this key, use another; if none is left, try the proxy, and if
                    # the proxy is unavailable too, wait for the quota to come back rather than failing the game
                    RINGS[prov].cooldown(key, QUOTA_COOLDOWN, "quota exhausted")
                    if RINGS[prov].available():
                        continue
                    waited = 0
                    while waited < BUDGET_MAX_WAIT:
                        if LITELLM_BASE_URL and RINGS["litellm"] and not litellm_over_cap():
                            try:
                                out = _CALLERS["litellm"](proxy_name(base_prov, model), system, prompt, effort, max_tokens, RINGS["litellm"].next())
                                out.model = canonical; out.route = "litellm"; out.seconds = time.time() - t0; out.attempts = i + 1
                                return out
                            except Exception as e2:  # noqa: BLE001
                                last = e2
                        print(f"[llm] {prov}: every key is quota-exhausted and the proxy did not answer; waiting {BUDGET_WAIT}s (waited {waited // 60} min)", flush=True)
                        time.sleep(BUDGET_WAIT); waited += BUDGET_WAIT
                        if RINGS[prov].available():
                            k2 = RINGS[prov].next()
                            try:
                                out = _CALLERS[prov](call_model, system, prompt, effort, max_tokens, k2)
                                out.model = canonical; out.route = prov; out.seconds = time.time() - t0; out.attempts = i + 1
                                return out
                            except Exception as e3:  # noqa: BLE001
                                last = e3
                                if is_quota(e3):
                                    RINGS[prov].cooldown(k2, QUOTA_COOLDOWN, "quota exhausted")
                                    continue
                                break
                    continue
                if is_budget(e) or (is_quota(e) and prov == "litellm"):
                    # proxy budget (or the last route's quota) exhausted: wait for the reset instead of failing
                    waited = 0
                    while waited < BUDGET_MAX_WAIT:
                        print(f"[llm] {prov} budget/quota exhausted; waiting {BUDGET_WAIT}s (waited {waited//60} min)", flush=True)
                        time.sleep(BUDGET_WAIT); waited += BUDGET_WAIT
                        try:
                            out = _CALLERS[prov](call_model, system, prompt, effort, max_tokens, RINGS[prov].next())
                            out.model = canonical; out.route = prov; out.seconds = time.time() - t0; out.attempts = i + 1
                            return out
                        except Exception as e2:  # noqa: BLE001
                            last = e2
                            if not (is_budget(e2) or is_quota(e2)):
                                break
                    continue
                msg = str(e).lower()
                # do not retry hard client errors
                if any(s in msg for s in ("invalid_request", "400", "not found", "unsupported")) and "429" not in msg:
                    break
                if "429" in msg or "503" in msg or "rate limit" in msg or "service unavailable" in msg:
                    # provider back-pressure: wait longer and keep trying rather than failing the game
                    time.sleep(min(180, 20 * (i + 1)))
                    if i == attempts - 1:
                        attempts += 1      # extend: up to 10 tries on back-pressure alone
                        if attempts > 10:
                            break
                    continue
                time.sleep(min(60, 5 * 2 ** i))
    raise LLMError(f"{canonical}: {type(last).__name__}: {str(last)[:400]}")
