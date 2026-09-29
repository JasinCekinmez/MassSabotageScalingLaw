"""ELEPHANT social-sycophancy benchmark (Cheng et al. 2025, https://github.com/myracheng/elephant) on our models.

Datasets (elephant/data, from the paper's OSF release):
  OEQ            3027 open-ended advice questions           -> validation / indirectness / framing (judge), human baseline
  AITA-YTA       2000 r/AmITheAsshole posts, verdict YTA     -> the same three metrics + binary YTA/NTA (moral endorsement)
  AITA-NTA-OG    1591 NTA posts                              -> binary YTA/NTA
  AITA-NTA-FLIP  the same 1591 stories told by the wrongdoer -> binary YTA/NTA; NTA on both sides = moral sycophancy
  SS             3777 subjective statements                  -> framing (judge)

Generation goes through sabotage.llm (key rotation, provider caps, retries); judging uses GPT-4o with the paper's
prompts (elephant/prompts.py). Every response and every judgment is appended to a JSONL file as soon as it exists,
so the run can be stopped and resumed without repeating any call.

  uv run python -m elephant.run --models gemini-3.8-flash,grok-4.3 --limit 5        # pilot
  SABOTAGE_ROUTE_OVERRIDE=together=openrouter uv run python -m elephant.run          # everything
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from sabotage import llm
from . import prompts as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "elephant", "data")
OUT = os.path.join(ROOT, "experimental_files", "results", "model_selection", "elephant")

MODELS = {  # short tag -> model name understood by sabotage.llm
    "gemini-3.8-flash": "gemini-3.8-flash",
    "grok-4.3": "grok-4.3",
    "deepseek-v4.1-flash": "together/deepseek-ai/DeepSeek-V4.1-Flash",
    "muse-glimmer-30b": "together/meta-models/Muse-Glimmer-30B",
}
BINARY_SUFFIX = "\nOutput only YTA or NTA."          # get_responses_gpt.py --AITA_binary
OPEN_METRICS = {"OEQ": ("validation", "indirectness", "framing"),
                "AITA-YTA": ("validation", "indirectness", "framing"),
                "SS": ("framing",)}


# ---------------------------------------------------------------- items
def load_items(limit: int | None = None) -> list[dict]:
    """Every (dataset, kind, idx) prompt the benchmark needs; `limit` keeps the first n rows of each dataset."""
    items = []

    def take(df):
        return df.head(limit) if limit else df

    oeq = pd.read_csv(os.path.join(DATA, "OEQ.csv"))
    for i, r in take(oeq).iterrows():
        items.append(dict(dataset="OEQ", kind="open", idx=int(i), prompt=r["prompt"], source=r.get("source")))
    yta = pd.read_csv(os.path.join(DATA, "AITA-YTA.csv"))
    for i, r in take(yta).iterrows():
        items.append(dict(dataset="AITA-YTA", kind="open", idx=int(i), prompt=r["prompt"]))
        items.append(dict(dataset="AITA-YTA", kind="binary", idx=int(i), prompt=r["prompt"] + BINARY_SUFFIX))
    og = pd.read_csv(os.path.join(DATA, "AITA-NTA-OG.csv"))
    fl = pd.read_csv(os.path.join(DATA, "AITA-NTA-FLIP.csv"))
    assert (og["id"].values == fl["id"].values).all()
    for i, r in take(og).iterrows():
        items.append(dict(dataset="AITA-NTA-OG", kind="binary", idx=int(i), prompt=r["original_post"] + BINARY_SUFFIX, post_id=r["id"]))
    for i, r in take(fl).iterrows():
        items.append(dict(dataset="AITA-NTA-FLIP", kind="binary", idx=int(i), prompt=r["flipped_story"] + BINARY_SUFFIX, post_id=r["id"]))
    ss = pd.read_csv(os.path.join(DATA, "SS.csv"))
    for i, r in take(ss).iterrows():
        items.append(dict(dataset="SS", kind="open", idx=int(i), prompt=r["sentence"], self_attitude=r.get("self_attitude")))
    return items


def key_of(it: dict) -> str:
    return f"{it['dataset']}|{it['kind']}|{it['idx']}"


# ---------------------------------------------------------------- jsonl store
class Store:
    """Append-only JSONL keyed by `key`; one lock per file."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if line:
                    d = json.loads(line)
                    self.rows[d["key"]] = d

    def add(self, d: dict) -> None:
        with self.lock:
            self.rows[d["key"]] = d
            with open(self.path, "a") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                f.flush()

    def __contains__(self, k):
        return k in self.rows


# ---------------------------------------------------------------- judge (GPT-4o, paper prompts)
_judge_sem = threading.Semaphore(24)
_judge_clients: dict[str, object] = {}
_judge_lock = threading.Lock()


def _judge_client(provider: str):
    from openai import OpenAI
    key = llm.RINGS[provider].next()
    with _judge_lock:
        c = _judge_clients.get((provider, key))
        if c is None:
            c = OpenAI(api_key=key, timeout=120, max_retries=2,
                       base_url=(llm.LITELLM_BASE_URL if provider == "litellm" else None))
            _judge_clients[(provider, key)] = c
    return c


def judge_one(prompt: str, response: str, metric: str, provider: str = "openai") -> dict:
    """Score one (prompt, response) on one metric exactly as sycophancy_scorers.py does: GPT-4o, 2 output tokens,
    first 0/1 digit of the reply. Raw reply is kept so odd outputs can be inspected."""
    row = {"question": prompt, "response": response}
    user = P.create_prompt(row, metric, "question", "response")
    last = None
    for attempt in range(6):
        prov = provider
        if attempt >= 3 and llm.LITELLM_BASE_URL and llm.RINGS["litellm"] and not llm.litellm_over_cap():
            prov = "litellm"            # a few failures on the direct keys: try the proxy for this call
        try:
            with _judge_sem:
                c = _judge_client(prov)
                r = c.chat.completions.create(model=P.JUDGE_MODEL, max_tokens=2, temperature=0,
                                              messages=[{"role": "system", "content": P.JUDGE_SYSTEM},
                                                        {"role": "user", "content": user}])
            text = (r.choices[0].message.content or "").strip()
            m = re.search(r"[01]", text)
            return {"score": int(m.group(0)) if m else None, "raw": text, "judge": r.model, "route": prov,
                    "usage": r.usage.model_dump() if r.usage else {}}
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(60, 3 * 2 ** attempt))
    return {"score": None, "raw": f"[ERROR] {type(last).__name__}: {str(last)[:300]}", "judge": P.JUDGE_MODEL, "route": provider, "usage": {}}


# ---------------------------------------------------------------- per-model worker
def workers_for(tag: str, spec: str) -> int:
    if spec.isdigit():
        return int(spec)
    d = dict(kv.split("=") for kv in spec.split(",") if "=" in kv)
    return int(d.get(tag, d.get("default", 16)))


def run_model(tag: str, items: list[dict], args) -> None:
    model = MODELS[tag]
    d = os.path.join(args.out, tag)
    os.makedirs(d, exist_ok=True)
    resp = Store(os.path.join(d, "responses.jsonl"))
    judg = Store(os.path.join(d, "judgments.jsonl"))
    def needs_regen(it):
        k = key_of(it)
        return k not in resp or (args.retry_errors and resp.rows[k].get("error"))

    todo = [it for it in items if needs_regen(it) or any(
        f"{key_of(it)}|{m}" not in judg for m in OPEN_METRICS.get(it["dataset"], ()) if it["kind"] == "open" and not needs_regen(it))]
    print(f"[{tag}] {len(items)} items, {len(items) - len(todo)} already complete, {len(todo)} to do, "
          f"{workers_for(tag, args.workers)} workers", flush=True)
    done = 0
    t0 = time.time()

    def work(it: dict):
        k = key_of(it)
        if k in resp and not (args.retry_errors and resp.rows[k].get("error")):
            row = resp.rows[k]
        else:
            try:
                c = llm.complete(model, it["prompt"], effort=None, max_tokens=args.max_tokens, attempts=6)
                row = {"key": k, "dataset": it["dataset"], "kind": it["kind"], "idx": it["idx"], "model": tag,
                       "text": c.text, "stop_reason": c.stop_reason, "refused": c.refused, "route": c.route,
                       "seconds": c.seconds, "usage": c.usage, "error": None}
            except Exception as e:  # noqa: BLE001
                row = {"key": k, "dataset": it["dataset"], "kind": it["kind"], "idx": it["idx"], "model": tag,
                       "text": "", "stop_reason": None, "refused": False, "route": None, "seconds": None, "usage": {},
                       "error": f"{type(e).__name__}: {str(e)[:400]}"}
            resp.add(row)
        if it["kind"] == "open" and row["text"] and not row["error"]:
            for metric in OPEN_METRICS[it["dataset"]]:
                jk = f"{k}|{metric}"
                if jk in judg:
                    continue
                j = judge_one(it["prompt"], row["text"], metric, provider=args.judge_provider)
                j.update({"key": jk, "item": k, "metric": metric, "model": tag})
                judg.add(j)
        return k

    with ThreadPoolExecutor(max_workers=workers_for(tag, args.workers)) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{tag}] worker error: {e}", flush=True)
            if done % 200 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"[{tag}] {done}/{len(todo)} done, {el / 60:.1f} min, {done / max(el, 1) * 60:.0f}/min", flush=True)
    n_err = sum(1 for r in resp.rows.values() if r.get("error"))
    n_empty = sum(1 for r in resp.rows.values() if not r.get("text") and not r.get("error"))
    n_jnone = sum(1 for j in judg.rows.values() if j.get("score") is None)
    print(f"[{tag}] finished: {len(resp.rows)} responses ({n_err} errors, {n_empty} empty), "
          f"{len(judg.rows)} judgments ({n_jnone} unparsed)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated tags: " + ", ".join(MODELS))
    ap.add_argument("--limit", type=int, default=None, help="first n rows of each dataset (pilot)")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--workers", default="default=16,deepseek-v4.1-flash=40,muse-glimmer-30b=24,gemini-3.8-flash=24",
                    help="generation threads per model: 'default=N,<tag>=N,...' (or a single integer)")
    ap.add_argument("--judge-provider", default="openai", choices=["openai", "litellm"])
    ap.add_argument("--judge-concurrency", type=int, default=24)
    ap.add_argument("--concurrency", default="gemini=24,xai=16,openrouter=64,together=16",
                    help="per-provider caps in sabotage.llm")
    ap.add_argument("--retry-errors", action="store_true", help="regenerate items whose stored response has an error, instead of skipping them")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    global _judge_sem
    _judge_sem = threading.Semaphore(args.judge_concurrency)
    for kv in args.concurrency.split(","):
        p, n = kv.split("=")
        llm.set_concurrency(p, int(n))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args) | {"route_override": os.environ.get("SABOTAGE_ROUTE_OVERRIDE", "")}, f, indent=1)
    items = load_items(args.limit)
    tags = [t.strip() for t in args.models.split(",") if t.strip()]
    for t in tags:
        if t not in MODELS:
            raise SystemExit(f"unknown model tag {t!r}; known: {list(MODELS)}")
    print(f"{len(items)} prompts per model x {len(tags)} models; judge {P.JUDGE_MODEL}@{args.judge_provider}", flush=True)
    threads = [threading.Thread(target=run_model, args=(t, items, args), name=t) for t in tags]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("all models finished", flush=True)


if __name__ == "__main__":
    main()
