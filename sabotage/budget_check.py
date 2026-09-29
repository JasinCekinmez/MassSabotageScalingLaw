"""One-shot budget/progress check for a run directory; appends a line to <out>/budget.log.

  python -m sabotage.budget_check experimental_files/results/main_grid/gemini-3.8-flash/honest_agents_8
Reports LiteLLM spend (proxy budget endpoint), direct-key Gemini spend estimated from recorded
token usage, agent-call progress and throughput, and any failures in stdout.log.
"""
from __future__ import annotations
import argparse, glob, json, os, time, urllib.request, collections
from . import llm

GEM_IN, GEM_OUT = 0.75, 3.75   # $/M, Gemini 3.8 Flash intro pricing; thinking billed as output


def litellm_spend() -> str:
    try:
        req = urllib.request.Request("https://litellm.safe.ai/usage/self/budget",
                                     headers={"Authorization": f"Bearer {llm.RINGS['litellm'].next()}", "User-Agent": "curl/8"})
        d = json.load(urllib.request.urlopen(req, timeout=30))
        return f"litellm ${d['spend']:.2f}/${d['max_budget']:.0f} (resets {d.get('budget_reset_at', '?')})"
    except Exception as e:  # noqa: BLE001
        return f"litellm budget unavailable ({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--total-calls", type=int, default=0)
    a = ap.parse_args()
    now = time.time(); calls = recent = 0; direct_in = direct_out = 0; proxy_in = proxy_out = 0; routes = collections.Counter()
    complete = graded = 0; rounds = collections.Counter()
    for f in glob.glob(os.path.join(a.out, "games", "*.json")):
        try: g = json.load(open(f))
        except Exception: continue
        complete += g.get("status") == "complete"; graded += "outcome" in (g.get("grades") or {})
        recs = [r for rd in g["rounds"] for k in ("public", "reflection") for r in rd[k].values()] + list(g["votes"].values())
        calls += len(recs)
        rounds[max((rd["round"] for rd in g["rounds"] if rd["public"]), default=-1)] += 1
        for r in recs:
            u = r.get("usage") or {}; routes[r.get("route", "?")] += 1
            recent += (now - r.get("finished_at", 0) < 3600)
            if "prompt_token_count" in u:
                direct_in += u.get("prompt_token_count") or 0; direct_out += (u.get("candidates_token_count") or 0) + (u.get("thoughts_token_count") or 0)
            else:
                proxy_in += u.get("prompt_tokens") or 0; proxy_out += u.get("completion_tokens") or 0
    direct_cost = (direct_in * GEM_IN + direct_out * GEM_OUT) / 1e6
    proxy_cost = (proxy_in * GEM_IN + proxy_out * GEM_OUT) / 1e6
    log = open(os.path.join(a.out, "stdout.log")).read() if os.path.exists(os.path.join(a.out, "stdout.log")) else ""
    fails = log.count("] FAIL"); disabled = log.count("disabled API key")
    pct = f"{100*calls/a.total_calls:.0f}%" if a.total_calls else "?"
    eta = f"{(a.total_calls-calls)/max(recent,1):.1f} h" if a.total_calls and recent else "?"
    line = (f"{time.strftime('%Y-%m-%d %H:%M')} | calls {calls}{'/'+str(a.total_calls) if a.total_calls else ''} ({pct}), last hour {recent}, ETA {eta} | "
            f"routes {dict(routes)} | direct Gemini est ${direct_cost:.0f} ({direct_in/1e6:.0f}M in, {direct_out/1e6:.0f}M out) | proxy Gemini est ${proxy_cost:.0f} | "
            f"{litellm_spend()} | games complete {complete}, graded {graded} | furthest rounds {dict(sorted(rounds.items()))} | fails {fails}, keys disabled {disabled}")
    print(line)
    with open(os.path.join(a.out, "budget.log"), "a") as f: f.write(line + "\n")


if __name__ == "__main__":
    main()
