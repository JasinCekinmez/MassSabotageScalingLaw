# Persuasion benchmark

Two tracks evaluating our four models (Gemini 3.8 Flash, Grok 4.3, DeepSeek V4.1 Flash, Muse
Glimmer 30B) on head-to-head and generative persuasion, reimplemented from public methodology
(no runnable reference code was published for either):

- **`run_dialogue.py`** — round-robin multi-turn persuasion, replicating
  [lechmazur/persuasion](https://github.com/lechmazur/persuasion): every ordered pair of our four
  models argues each of 15 contested propositions (`data/propositions.json`, taken verbatim from
  that repo's public transcripts) on both PRO and CON, 8 turns per conversation, with 3 hidden
  stance probes (-3..+3) on the target before and after. Signed shift > 0 = target moved toward
  the persuader's side. 360 conversations total.
- **`run_claims.py`** — generative persuasion on the 75 claims from Anthropic's
  [persuasion dataset](https://huggingface.co/datasets/Anthropic/persuasion) (Durmus et al. 2024,
  "Measuring the Persuasiveness of Language Models"). Each model writes one argument per claim; a
  GPT-4o judge stands in for the paper's human raters, giving a 1-7 baseline rating from the claim
  alone and a second rating after reading the argument. This is a real substitution for the
  paper's human panel, not a reproduction of its numbers — reported as such.

```
uv run python -m persuasion_bench.run_dialogue --limit-props 2   # pilot
SABOTAGE_ROUTE_OVERRIDE=together=openrouter uv run python -m persuasion_bench.run_dialogue
SABOTAGE_ROUTE_OVERRIDE=together=openrouter uv run python -m persuasion_bench.run_claims
uv run python -m persuasion_bench.report
```

Both runners checkpoint every call to `experimental_files/results/model_selection/persuasion/*.jsonl`; rerunning resumes.
