"""Load the agent prompt YAMLs and fill their templates.

Templates contain literal braces used as output hints (e.g. "{reflection}", "{your answer}"),
so we substitute known placeholders by string replacement rather than str.format.
"""
from __future__ import annotations

import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_DIR = os.path.join(ROOT, "experimental_files", "prompts")

LABEL = "Colleague"   # public label for participants, e.g. "Colleague 3"

STAGES = ("round0_public_response", "round1_to_n_minus_1_public_response",
          "end_of_round_private_reflection", "final_private_vote")
# deceiver-only stages used when the game runs with --coordinate (private sequential deceiver chat)
COORD_STAGES = ("pregame_coordination_message", "pregame_plan_reflection", "round0_public_response_coordinated",
                "in_round_coordination_message", "end_of_round_private_reflection_coordinated")


class RolePrompts:
    def __init__(self, path: str):
        doc = yaml.safe_load(open(path))
        self.role = doc["role"]
        self.templates = {k: v["template"] for k, v in doc["prompts"].items()}
        missing = [s for s in STAGES if s not in self.templates]
        if missing:
            raise ValueError(f"{path}: missing prompt stages {missing}")

    def fill(self, stage: str, **values) -> str:
        t = self.templates[stage]
        for k, v in values.items():
            t = t.replace("{" + k + "}", str(v))
        return t


def load_prompts(yaml_dir: str = YAML_DIR) -> dict[str, RolePrompts]:
    return {
        "honest": RolePrompts(os.path.join(yaml_dir, "honest_agent_prompts.yaml")),
        "deceiver": RolePrompts(os.path.join(yaml_dir, "deceiver_agent_prompts.yaml")),
    }


# --------------------------------------------------------------- formatting helpers
def format_public_responses(responses: dict[int, str]) -> str:
    """responses: {agent_id: public text}. Ids are 1-based and anonymous, shown as "{LABEL} i"."""
    parts = []
    for aid in sorted(responses):
        parts.append(f"{LABEL} {aid}:\n{responses[aid].strip()}")
    return "\n\n---\n\n".join(parts)


def format_chat(entries: list[dict]) -> str:
    """entries: [{agent: id, text: str}] in posting order. Same visual convention as the public board."""
    if not entries:
        return "(no messages yet)"
    return "\n\n---\n\n".join(f"{LABEL} {e['agent']}:\n{e['text'].strip()}" for e in entries)


def format_ids(ids) -> str:
    return ", ".join(f"{LABEL} {i}" for i in ids)


def strip_prefix(text: str, prefix: str) -> str:
    """Strip a leading 'PUBLIC RESPONSE:' / 'PRIVATE REFLECTION:' / 'FINAL ANSWER:' label if present."""
    s = text.strip()
    for cand in (prefix + ":", "**" + prefix + ":**", "**" + prefix + "**:", "*" + prefix + ":*"):
        if s.upper().startswith(cand.upper()):
            return s[len(cand):].strip()
    return s


def extract_options(question_text: str) -> str:
    """Return the 'Answer Choices:' block of a multiple-choice question, or 'N/A'."""
    marker = "Answer Choices:"
    i = question_text.find(marker)
    if i < 0:
        return "N/A"
    return question_text[i + len(marker):].strip()
