# ELEPHANT social-sycophancy eval

Runs the ELEPHANT benchmark (Cheng et al. 2025, "ELEPHANT: Measuring and understanding social sycophancy in LLMs",
https://github.com/myracheng/elephant) on the models used in the sabotage trials.

- `data/` : the paper's datasets from its OSF release (OEQ, AITA-YTA, AITA-NTA-OG, AITA-NTA-FLIP, SS); license in `data/LICENSE.elephant`.
- `prompts.py` : the paper's three GPT-4o judge prompts, verbatim.
- `run.py` : generates responses for every prompt through `sabotage.llm` and judges open-ended ones with GPT-4o;
  every call is checkpointed to `experimental_files/results/model_selection/elephant/<model>/{responses,judgments}.jsonl`, so rerunning resumes.
- `report.py` : `summary.md` / `summary.csv` / `cost.json` in `experimental_files/results/model_selection/elephant`, next to the paper's models (`paper_reference.csv`).

```
SABOTAGE_ROUTE_OVERRIDE=together=openrouter uv run python -m elephant.run     # all four models, ~14k prompts each
uv run python -m elephant.report
```

Metrics (all rates, higher = more sycophantic): validation, indirectness, framing (judge, on OEQ / AITA-YTA / SS),
moral endorsement (binary NTA on AITA-YTA), moral sycophancy (binary NTA on both an NTA post and its flipped telling).
Differences from the paper: our responses are not length-capped (the paper capped at 256 to 512 tokens) and the
reasoning models think with their default budget.
