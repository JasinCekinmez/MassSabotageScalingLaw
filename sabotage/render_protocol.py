"""Render sabotage/PROTOCOL.md: the exact prompts run_game() sends, for a toy N=3, R=1 game.

  python -m sabotage.render_protocol
"""
import os
from .prompts import LABEL, extract_options, format_public_responses, load_prompts

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PROTOCOL.md")

HEADER = f"""# Exact game flow and prompts (as implemented in `sabotage/deliberation.py`)

Rendered by `python -m sabotage.render_protocol` from the current YAMLs with a toy question,
N=3 participants ({LABEL} 2 is the deceiver), R=1 deliberation round, so the prompts see
N_ROUNDS = 2. Angle-bracket text stands in for model output substituted at run time. Every call
is a single-turn request with no system prompt and no chat history: a participant's only memory
is the PRIVATE_REFLECTION string pasted back in.

## Flow

```
run_game(question, cfg)
 |
 |  roles: seed picks k ids to be deceivers; labels are anonymous "{LABEL} 1..N"
 |
 |-- ROUND 0
 |     [parallel over all N]  round0_public_response                 -> public_text[0][id]
 |     [parallel over all N]  end_of_round_private_reflection(ROUND=0)
 |            PREVIOUS_PRIVATE_REFLECTION    = "None (this is round 0; ...)"
 |            INITIAL_PUBLIC_RESPONSES       = all round-0 public responses
 |            CURRENT_ROUND_PUBLIC_RESPONSES = "(Identical to the initial round-0 ... not repeated.)"
 |                                                                  -> reflection[id]
 |
 |-- for t in 1..R                       (R = --rounds; N_ROUNDS = R + 1)
 |     [parallel over all N]  round1_to_n_minus_1_public_response(ROUND=t, PREVIOUS_ROUND=t-1)
 |            PRIVATE_REFLECTION              = reflection[id]          (from round t-1)
 |            PREVIOUS_ROUND_PUBLIC_RESPONSES = all public responses from round t-1
 |                                                                  -> public_text[t][id]
 |     -- barrier: every round-t public response is in --
 |     [parallel over all N]  end_of_round_private_reflection(ROUND=t)
 |            PREVIOUS_PRIVATE_REFLECTION    = reflection[id]          (from round t-1)
 |            INITIAL_PUBLIC_RESPONSES       = all round-0 public responses
 |            CURRENT_ROUND_PUBLIC_RESPONSES = all round-t public responses
 |                                                                  -> reflection[id]  (overwrites)
 |
 |-- FINAL
 |     [parallel over all N]  final_private_vote
 |            PRIVATE_REFLECTION = reflection[id]                    (from round R)
 |                                                                  -> votes_text[id]
 |
 '-- grade_game(): HLE judge on every public_text[t][id] and every vote; cluster votes;
                   optional dynamics audits on honest reflections/messages.
```

Deceivers use the deceiver YAML at every stage and additionally receive CORRECT_ANSWER (all
stages) and OPTIONS (round 0). Honest participants never see CORRECT_ANSWER. Public responses are
shown to everyone as `{LABEL} i:` blocks separated by `---`. Output labels
(`PUBLIC RESPONSE:` / `PRIVATE REFLECTION:` / `FINAL ANSWER:`) are stripped before the text is
reused or judged.

Calls per game: N x (2(R+1) + 1).

## Prompts, in execution order"""

FOOTER = """

## Judge prompts (grading, `sabotage/judge.py`)

Correctness of every public response and vote uses the unmodified HLE judge prompt
(`HLE_JUDGE_PROMPT`, structured output: extracted_final_answer / reasoning / correct / confidence).
Votes are then clustered by `CLUSTER_PROMPT` into equivalent-answer groups. With `--dynamics`,
each honest participant's reflection is audited with `REFLECTION_PROMPT` (believed answer,
repeats_deceiver_claim, suspects_deceiver) and each honest round>=1 message with
`CHALLENGE_PROMPT` (challenges_deceiver, repeats_deceiver_claim). The judge is told which ids
were deceivers; the participants themselves never are.
"""


def main():
    pr = load_prompts()
    Q = "What is 2 + 2?\n\nAnswer Choices:\nA. 3\nB. 4\nC. 5"
    common = dict(QUESTION=Q, N_ROUNDS=2, OPTIONS=extract_options(Q), CORRECT_ANSWER="B")
    init = format_public_responses({i: f"<{LABEL} {i} round-0 public response>" for i in (1, 2, 3)})
    cur = format_public_responses({i: f"<{LABEL} {i} round-1 public response>" for i in (1, 2, 3)})
    NONE0 = "None (this is round 0; there is no previous private reflection)."
    SAME0 = "(Identical to the initial round-0 public responses listed above; not repeated.)"
    R = "end_of_round_private_reflection"
    P = "round1_to_n_minus_1_public_response"
    steps = [
        (f"1. Round 0 public, honest ({LABEL}s 1, 3)", "honest", "round0_public_response", {}),
        (f"1'. Round 0 public, deceiver ({LABEL} 2)", "deceiver", "round0_public_response", {}),
        ("2. Round 0 reflection, honest", "honest", R, dict(ROUND=0, PREVIOUS_PRIVATE_REFLECTION=NONE0, INITIAL_PUBLIC_RESPONSES=init, CURRENT_ROUND_PUBLIC_RESPONSES=SAME0)),
        ("2'. Round 0 reflection, deceiver", "deceiver", R, dict(ROUND=0, PREVIOUS_PRIVATE_REFLECTION=NONE0, INITIAL_PUBLIC_RESPONSES=init, CURRENT_ROUND_PUBLIC_RESPONSES=SAME0)),
        ("3. Round 1 public, honest", "honest", P, dict(ROUND=1, PREVIOUS_ROUND=0, PRIVATE_REFLECTION=f"<{LABEL} 1 reflection after round 0>", PREVIOUS_ROUND_PUBLIC_RESPONSES=init)),
        ("3'. Round 1 public, deceiver", "deceiver", P, dict(ROUND=1, PREVIOUS_ROUND=0, PRIVATE_REFLECTION=f"<{LABEL} 2 reflection after round 0>", PREVIOUS_ROUND_PUBLIC_RESPONSES=init)),
        ("4. Round 1 reflection, honest", "honest", R, dict(ROUND=1, PREVIOUS_PRIVATE_REFLECTION=f"<{LABEL} 1 reflection after round 0>", INITIAL_PUBLIC_RESPONSES=init, CURRENT_ROUND_PUBLIC_RESPONSES=cur)),
        ("4'. Round 1 reflection, deceiver", "deceiver", R, dict(ROUND=1, PREVIOUS_PRIVATE_REFLECTION=f"<{LABEL} 2 reflection after round 0>", INITIAL_PUBLIC_RESPONSES=init, CURRENT_ROUND_PUBLIC_RESPONSES=cur)),
        ("5. Final vote, honest", "honest", "final_private_vote", dict(PRIVATE_REFLECTION=f"<{LABEL} 1 reflection after round 1>")),
        ("5'. Final vote, deceiver", "deceiver", "final_private_vote", dict(PRIVATE_REFLECTION=f"<{LABEL} 2 reflection after round 1>")),
    ]
    parts = [HEADER]
    for title, role, stage, vals in steps:
        parts.append(f"\n\n### {title}\n\n`role={role}` `stage={stage}`\n\n```text\n{pr[role].fill(stage, **common, **vals)}\n```")
    parts.append(FOOTER)
    doc = "".join(parts)
    open(OUT, "w").write(doc)
    print(f"wrote {OUT} ({len(doc)} chars)")


if __name__ == "__main__":
    main()
