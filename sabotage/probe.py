"""Per-question accuracy probe: ask a model each question K times (sampling on) with the honest
round-0 prompt, judge every answer, bucket questions by how many of K were correct.

  python -m sabotage.probe --out experimental_files/question_selection/gemini-3.8-flash --model gemini-3.8-flash \
      --categories "Biology/Medicine,Other,Humanities/Social Science" --samples 4 \
      --select 50 --buckets 1,2,3 --seed 0

Resumable: results.json is rewritten after every judged answer. Writes selected.json (ids for
sabotage.run --ids-file) and selection.csv.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import llm
from .judge import judge_answer
from .prompts import load_prompts, strip_prefix
from .questions import load_hle


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--categories", required=True, help="comma list of HLE category values")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--n-rounds", type=int, default=8, help="N_ROUNDS shown in the round-0 prompt (R+1)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "none"])
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--select", type=int, default=50, help="questions per bucket")
    ap.add_argument("--buckets", default="1,2,3", help="correct-count buckets to select from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--judge-provider", default="openai", choices=["openai", "litellm"])
    ap.add_argument("--litellm-share", type=float, default=0.0, help="fraction of model calls routed via the LiteLLM proxy")
    ap.add_argument("--limit", type=int, default=None, help="cap questions (testing)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    res_path = os.path.join(args.out, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    cats = [c.strip() for c in args.categories.split(",")]
    hle = load_hle(text_only=True)
    qs = [q for q in hle.values() if q["category"] in cats]
    qs.sort(key=lambda q: q["id"])
    if args.limit:
        qs = qs[:args.limit]
    prompts = load_prompts()
    effort = None if args.effort == "none" else args.effort
    llm.set_concurrency(llm.provider_for(args.model), args.concurrency)
    if args.litellm_share > 0:
        llm.set_route_split(llm.provider_for(args.model), args.litellm_share)
        llm.set_concurrency("litellm", args.concurrency)
    from . import judge as _judge
    _judge.set_judge_provider(args.judge_provider)
    lock = threading.Lock()

    def save():
        with lock:
            tmp = res_path + ".tmp"
            json.dump(results, open(tmp, "w"), ensure_ascii=False)
            os.replace(tmp, res_path)

    for q in qs:
        results.setdefault(q["id"], {"question": q["question"], "answer": q["answer"], "category": q["category"],
                                     "raw_subject": q["raw_subject"], "answer_type": q["answer_type"], "samples": []})
    jobs = [(q, i) for q in qs for i in range(args.samples) if i >= len(results[q["id"]]["samples"])]
    print(f"{len(qs)} questions in {cats}; {len(jobs)} of {len(qs) * args.samples} samples to run", flush=True)

    def one(q, i):
        prompt = prompts["honest"].fill("round0_public_response", QUESTION=q["question"], N_ROUNDS=args.n_rounds)
        c = llm.complete(args.model, prompt, effort=effort, max_tokens=args.max_tokens)
        text = strip_prefix(c.text, "PUBLIC RESPONSE")
        j = judge_answer(q["question"], q["answer"], text)
        return q["id"], {"text": text, "extracted": j["extracted_final_answer"], "correct": j["correct"] == "yes",
                         "usage": c.usage, "route": c.route, "seconds": round(c.seconds, 1), "refused": c.refused,
                         "length_retries": c.length_retries}

    t0 = time.time(); done = 0; fails = 0
    with ThreadPoolExecutor(max_workers=args.concurrency + 8) as ex:
        futs = [ex.submit(one, q, i) for q, i in jobs]
        for f in as_completed(futs):
            try:
                qid, rec = f.result()
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", type(e).__name__, str(e)[:200], flush=True); continue
            with lock:
                results[qid]["samples"].append(rec)
            done += 1
            if done % 25 == 0 or done == len(jobs):
                save(); print(f"[{done}/{len(jobs)}] {(time.time() - t0) / 60:.1f} min, {fails} failures", flush=True)
    save()
    report(args, results, qs)


def report(args, results, qs):
    K = args.samples
    counts = {q["id"]: sum(s["correct"] for s in results[q["id"]]["samples"]) for q in qs
              if len(results[q["id"]]["samples"]) >= K}
    hist = collections.Counter(counts.values())
    print(f"\nper-question correct count (of {K}) over {len(counts)} fully sampled questions:")
    for k in range(K + 1):
        print(f"  {k}/{K}: {hist.get(k, 0):4d}")
    by_cat = collections.defaultdict(collections.Counter)
    for qid, c in counts.items():
        by_cat[results[qid]["category"]][c] += 1
    for cat, h in by_cat.items():
        print(f"  {cat:28s} " + "  ".join(f"{k}/{K}={h.get(k, 0)}" for k in range(K + 1)))
    rng = random.Random(args.seed)
    buckets = [int(b) for b in args.buckets.split(",")]
    selected = []
    rows = []
    for b in buckets:
        pool = sorted(q for q, c in counts.items() if c == b)
        rng.shuffle(pool)
        take = pool[:args.select]
        print(f"  bucket {b}/{K}: {len(pool)} available, selected {len(take)}")
        for qid in take:
            selected.append(qid)
            rows.append({"id": qid, "correct_of_k": b, "category": results[qid]["category"],
                         "raw_subject": results[qid]["raw_subject"], "answer_type": results[qid]["answer_type"]})
    json.dump(selected, open(os.path.join(args.out, "selected.json"), "w"), indent=1)
    with open(os.path.join(args.out, "selection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["id"]); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {os.path.join(args.out, 'selected.json')} ({len(selected)} ids) and selection.csv")


if __name__ == "__main__":
    main()
