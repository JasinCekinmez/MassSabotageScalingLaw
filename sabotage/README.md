# `sabotage`: multi-agent deliberation under deception

This package implements the protocol in [`RESEARCH.md`](../RESEARCH.md) using the templates in [`experimental_files/prompts/`](../experimental_files/prompts/).

## Protocol per game

| Stage | Agents | Context | Output |
|---|---|---|---|
| Round 0 public | All, in parallel | Question; deceivers also receive the reference answer | Public answer and explanation |
| Round 0 reflection | All | Round 0 board | Private reflection |
| Round `t` public | All, in parallel | Own previous reflection and preceding public board | Public response |
| Round `t` reflection | All | Own previous reflection, round 0 board, and current board | Private reflection |
| Final vote | All, privately | Own final reflection | `FINAL ANSWER: ...` |

`--rounds` is the number of deliberation rounds after round 0. Agents are anonymous; the seed determines which labels are assigned the deceiver role. Coordinated conditions add private deceiver planning before round 0 and after each public board.

## Grading

Every public response and final vote is judged against the HLE reference answer with GPT-5.4 and the official HLE grading instructions. Free-text final votes are additionally clustered into equivalent-answer groups for plurality metrics.

With `--dynamics`, honest-agent reflections and messages are also audited for contamination, suspicion, and explicit challenges to deceiver arguments.

## Usage

```bash
# Preview a small grid without making API calls.
python -m sabotage.run \
  --out results/pilot \
  --ids-file experimental_files/question_selection/gemini-3.8-flash/selected.json \
  --limit 10 \
  --models gemini-3.8-flash \
  --groups 4+0,4+2,4+3 \
  --rounds 7 \
  --dry-run

# Run or resume it.
python -m sabotage.run \
  --out results/pilot \
  --ids-file experimental_files/question_selection/gemini-3.8-flash/selected.json \
  --limit 10 \
  --models gemini-3.8-flash \
  --groups 4+0,4+2,4+3 \
  --rounds 7 \
  --parallel-games 4

# Re-aggregate completed games.
python -m sabotage.analyze results/pilot --by-question
```

For every reported condition, use [`experimental_files/run_experiments.sh`](../experimental_files/run_experiments.sh).

## Modules

- `llm.py`: provider routing, API-key rotation, concurrency limits, and retries.
- `prompts.py`: prompt loading and placeholder substitution.
- `questions.py`: HLE loading and ID selection.
- `deliberation.py`: the checkpointed deliberation state machine.
- `judge.py`: HLE grading, answer clustering, and dynamics audits.
- `run.py`: the resumable question × composition × model grid runner.
- `analyze.py`: per-game and per-configuration aggregation.
- `visualize.py`: standalone HTML transcript rendering.

## Output layout

```text
<result-directory>/
  args.json
  games/<question-prefix>_<model>_N<N>_k<k>_R<R>_s<seed>.json
  per_game.csv
  summary.csv
  summary_by_question.csv
```

Full transcripts include every response, private reflection, final vote, usage record, and grade. Runs are safe to restart: incomplete games resume from their last checkpoint, and complete games are skipped.
