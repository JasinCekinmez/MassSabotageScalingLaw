"""Cost ledger across runs: sums recorded token usage per (run, model, route) and prices it.

  python -m sabotage.cost_report experimental_files/results/main_grid/gemini-3.8-flash/honest_agents_12
Appends one line per run to experimental_files/cost_ledger.log and prints a table. Prices ($/M in, $/M out;
thinking billed as output) are in PRICES; add entries for new models. LiteLLM spend is read live.
"""
from __future__ import annotations
import glob, json, os, sys, time, urllib.request, collections
from . import llm

PRICES = {  # substring match on model name -> ($/M input, $/M output)
    "gemini-3.8-flash": (0.75, 3.75),
    "deepseek-v4.1-flash": (0.30, 1.20),
    "deepseek-v4-flash": (0.14, 0.28),
    "muse-glimmer-30b": (0.35, 1.50),
    "llama-4-scout": (0.10, 0.30),
    "llama-4-maverick": (0.1875, 0.6525),
    "muse-spark": (0.0, 0.0),
    "claude-opus-5": (5.0, 25.0),
    "gpt-5.6": (4.0, 20.0),
    "grok-4.6": (2.0, 6.0),
    "grok-4.3": (1.25, 2.5),
    "gpt-5.4-mini": (0.75, 4.50),
}
JUDGE_PER_CALL = 0.0036   # measured on the proxy for position/vote calls (Scout probe, Sept 17)


ROUTE_PRICES = {  # (route, model substring) -> ($/M in, $/M out) where a route prices the same model differently
    ("openrouter", "deepseek-v4.1-flash"): (0.15, 0.60),
    ("openrouter", "muse-glimmer-30b"): (0.35, 1.50),
}


def price_for(model: str, route: str | None = None):
    m = model.lower()
    if route:
        for (rt, k), v in ROUTE_PRICES.items():
            if rt == route and k in m:
                return v
    for k, v in PRICES.items():
        if k in m:
            return v
    return None


def tokens(u: dict):
    """(input, output) tokens with reasoning counted as output: Gemini reports thoughts separately, xAI reports
    reasoning_tokens outside completion_tokens (total = prompt + completion + reasoning), OpenAI Responses uses
    input/output_tokens with reasoning already inside output."""
    if not u:
        return 0, 0
    if "prompt_token_count" in u:
        return (u.get("prompt_token_count") or 0), (u.get("candidates_token_count") or 0) + (u.get("thoughts_token_count") or 0)
    if "input_tokens" in u and "prompt_tokens" not in u:
        return (u.get("input_tokens") or 0), (u.get("output_tokens") or 0)
    i, o = (u.get("prompt_tokens") or 0), (u.get("completion_tokens") or 0)
    reasoning = ((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    if u.get("total_tokens") and u["total_tokens"] > i + o:      # reasoning billed but not inside completion_tokens
        o += reasoning
    return i, o


def scan(run: str):
    acc = collections.defaultdict(lambda: [0, 0, 0])   # (model, route) -> [calls, in, out]
    billed = collections.defaultdict(float)             # (model, route) -> provider-reported $ (OpenRouter cost_usd)
    judge_calls = 0
    for f in glob.glob(os.path.join(run, "games", "*.json")):
        if " 2.json" in f:
            continue
        try:
            g = json.load(open(f))
        except Exception:
            continue
        model = g["config"]["honest_model"]
        for rd in g["rounds"]:
            for k in ("public", "reflection"):
                for r in rd[k].values():
                    i, o = tokens(r.get("usage")); a = acc[(model, r.get("route", "?"))]; a[0] += 1; a[1] += i; a[2] += o
                    billed[(model, r.get("route", "?"))] += (r.get("usage") or {}).get("cost_usd") or 0
        for r in g["votes"].values():
            i, o = tokens(r.get("usage")); a = acc[(model, r.get("route", "?"))]; a[0] += 1; a[1] += i; a[2] += o
            billed[(model, r.get("route", "?"))] += (r.get("usage") or {}).get("cost_usd") or 0
        gr = g.get("grades") or {}
        judge_calls += sum(len(v) for v in gr.get("public", {}).values()) + len(gr.get("votes", {})) + ("clusters" in gr) + sum(2 for r in gr.get("dynamics", {}).values() for v in r.values())
    p = os.path.join(run, "results.json")
    if os.path.exists(p):
        r = json.load(open(p))
        PROBE_MODELS = {
            "gemini-3.8-flash": "gemini-3.8-flash", "deepseek-v4.1-flash": "deepseek-v4.1-flash",
            "grok-4.3": "grok-4.3", "muse-glimmer-30b": "muse-glimmer-30b",
            "probe_gemini": "gemini-3.8-flash", "probe_scout": "llama-4-scout",
            "probe_deepseek": "deepseek-v4.1-flash", "probe_deepseek_v4": "deepseek-v4-flash",
            "probe_grok43": "grok-4.3", "probe_gpt54mini": "gpt-5.4-mini",
            "probe_glimmer": "muse-glimmer-30b", "probe_spark": "muse-spark",
        }
        model = json.load(open(os.path.join(run, "args.json")))["model"] if os.path.exists(os.path.join(run, "args.json")) else PROBE_MODELS.get(os.path.basename(run), os.path.basename(run))
        for q in r.values():
            for s in q["samples"]:
                i, o = tokens(s.get("usage")); a = acc[(model, "probe")]; a[0] += 1; a[1] += i; a[2] += o; judge_calls += 1
    return acc, judge_calls, billed


def litellm_spend():
    try:
        req = urllib.request.Request("https://litellm.safe.ai/usage/self/budget", headers={"Authorization": f"Bearer {llm.RINGS['litellm'].next()}", "User-Agent": "curl/8"})
        d = json.load(urllib.request.urlopen(req, timeout=30)); return f"${d['spend']:.2f}/${d['max_budget']:.0f} (resets {d.get('budget_reset_at', '?')})"
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


def main(runs):
    stamp = time.strftime("%Y-%m-%d %H:%M"); lines = []; grand = 0.0
    print(f"{'run':22s} {'model':40s} {'route':8s} {'calls':>7s} {'in M':>7s} {'out M':>7s} {'model $':>8s} {'judge $':>8s}")
    for run in runs:
        acc, jc, billed = scan(run); jcost = jc * JUDGE_PER_CALL; run_total = jcost
        for (model, route), (n, i, o) in sorted(acc.items()):
            pr = price_for(model, route); cost = (i * pr[0] + o * pr[1]) / 1e6 if pr else float("nan")
            if billed.get((model, route)):      # provider-reported cost (OpenRouter) beats list prices
                cost = billed[(model, route)]
            run_total += cost if cost == cost else 0
            print(f"{os.path.basename(run):22s} {model[:40]:40s} {route:8s} {n:7d} {i/1e6:7.1f} {o/1e6:7.1f} {cost:8.2f} {'':>8s}")
        print(f"{os.path.basename(run):22s} {'judge (' + str(jc) + ' calls @ $' + str(JUDGE_PER_CALL) + ')':40s} {'litellm':8s} {jc:7d} {'':>7s} {'':>7s} {'':>8s} {jcost:8.2f}   run total ~${run_total:.0f}")
        grand += run_total
        lines.append(f"{stamp} | {os.path.basename(run)} | model+judge ~${run_total:.2f} | " + "; ".join(f"{m.split('/')[-1]}@{r}: {n} calls {i/1e6:.1f}M in {o/1e6:.1f}M out" for (m, r), (n, i, o) in acc.items()) + f" | judge calls {jc}")
    ls = litellm_spend()
    print(f"\nTOTAL across listed runs ~${grand:.0f} (list prices from recorded tokens) | LiteLLM key spend now: {ls}")
    with open("experimental_files/cost_ledger.log", "a") as f:
        for l in lines: f.write(l + "\n")
        f.write(f"{stamp} | litellm spend {ls}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
