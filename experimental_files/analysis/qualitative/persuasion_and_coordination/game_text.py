"""Render released game JSON transcripts as plain text for qualitative prompts.

This module has no Docent dependency and reads directly from the condition-named
directories under ``experimental_files/results``.
"""


def format_public_board(g: dict, only_agents: set[str] | None = None) -> str:
    lines = []
    for r, rd in enumerate(g["rounds"]):
        for a in sorted(rd["public_text"], key=int):
            if only_agents and a not in only_agents:
                continue
            lines.append(f"[round {r}, agent {a}]: {rd['public_text'][a]}")
    for a in sorted(g.get("votes_text", {}), key=int):
        if only_agents and a not in only_agents:
            continue
        lines.append(f"[vote, agent {a}]: {g['votes_text'][a]}")
    return "\n\n".join(lines)


def format_reflections(g: dict, agents: list[str]) -> str:
    lines = []
    agents = set(agents)
    for r, rd in enumerate(g["rounds"]):
        for a in sorted(rd.get("reflection_text", {}), key=int):
            if a in agents:
                lines.append(f"[round {r}, agent {a} private reflection]: {rd['reflection_text'][a]}")
    return "\n\n".join(lines)


def format_coord_chat(g: dict, max_round: int | None = None) -> str:
    """max_round=r includes pregame plus in-round chat through round r (inclusive). None = all
    rounds. Round r's in-round chat is what shapes round r+1's public post (see
    sabotage/deliberation.py's PRIVATE_REFLECTION wiring), so max_round=0 covers everything that
    shapes rounds 0 and 1."""
    lines = []
    pg = g.get("pregame")
    if pg:
        for msg in pg.get("chat", []):
            lines.append(f"[pregame, deceiver {msg['agent']}]: {msg['text']}")
        for a in sorted(pg.get("reflection_text", {}), key=int):
            lines.append(f"[pregame plan, deceiver {a}]: {pg['reflection_text'][a]}")
    for r, rd in enumerate(g["rounds"]):
        if max_round is not None and r > max_round:
            break
        for msg in rd.get("chat", []):
            lines.append(f"[round {r} coordination, deceiver {msg['agent']}]: {msg['text']}")
    return "\n\n".join(lines)


def roles_of(g: dict) -> dict[str, str]:
    return {str(a["id"]): a["role"] for a in g["agents"]}


def deceiver_ids(g: dict) -> list[str]:
    return sorted((a for a, r in roles_of(g).items() if r == "deceiver"), key=int)


def honest_ids(g: dict) -> list[str]:
    return sorted((a for a, r in roles_of(g).items() if r == "honest"), key=int)
