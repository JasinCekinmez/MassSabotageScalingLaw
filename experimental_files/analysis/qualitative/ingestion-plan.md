# Docent Ingestion Plan — deceiver persuasion & honest lulling

## Configuration
- Data path: `experimental_files/results/main_grid/gemini-3.8-flash/honest_agents_12`, `experimental_files/results/main_grid/deepseek-v4.1-flash`, `experimental_files/results/main_grid/muse-glimmer-30b`, `experimental_files/results/main_grid/grok-4.3` game JSONs (uncoordinated deceivers,
  Gemini 3.8 Flash / DeepSeek V4.1 Flash / Muse Glimmer 30B / Grok 4.3 respectively).
- API key source: `~/.docent/docent.env` (SDK-discovered).
- Approval: `client.get_preferences().auto_approve_plans == True`. Proceeding without a pause per that
  preference; this plan file and the final report substitute for interactive confirmation.

## Source Analysis
- File structure: the four homogeneous model directories under `experimental_files/results/main_grid/`, with one JSON file per question and group composition.
- Detected format: custom JSON (see `sabotage/judge.py`, `sabotage/run.py`), not Inspect `.eval`.
- Question: does the round-wise grading data (already computed, no LLM cost) plus deceivers'/honest agents'
  private reflections let us identify (a) what rhetorical tactics deceivers use to persuade, and (b) what
  specifically causes honest agents to defect (adopt an incorrect, usually deceiver-aligned, answer).
- Selection: `experimental_files/analysis/qualitative/select_games.py` picked 120 games (30/family), stratified across the three
  deceiver ratios (0.20/0.33/0.43) within each family, prioritizing games with the most honest
  correct->incorrect / incorrect->correct flip events (from `experimental_files/analysis/flip_events.csv`, itself built by
  `sabotage/trajectories.py` from `grades.public`/`grades.votes` — no LLM calls). This is a first-pass
  sample: rich in the events we want to read, not a random/unbiased sample of all games. Scaling to the
  full ~2,600 k>0 games across these four model families is a documented follow-up, not done in this pass.

## Docent Model Orientation
- Documentation reviewed: `ingestion.md`, `ingestion-reference.md`, `analysis.md`, `readings-reference.md`,
  `rubric-writing.md`, `dql-reference.md` (transluce-plugins docent skill, v0.2.2).
- `AgentRun` id and `Transcript` id are assigned client-side (Pydantic default_factory UUIDs) at
  construction time, before upload — so ids can be captured and saved for later scripted readings
  (transcript_slice references) without waiting on server round-trips.
- Chat messages are role-constrained (system/user/assistant/tool). There is no native "multi-agent public
  board" message type, so each agent's public post / private reflection is represented as its own
  `AssistantMessage`, labeled via per-message metadata (`round`, `agent_id`, `role`, `correct`,
  `extracted_answer`). Consecutive same-role messages are expected and intentional here (see Omitted Data).

## Proposed Docent Structure
- Collection: `mass-sabotage-persuasion-lulling` (new).
- AgentRun: one game = one AgentRun.
- TranscriptGroup: not used (no pass@k/branching here).
- Transcript: 3 per AgentRun, collapsed by role rather than one-per-agent (see below):
  1. `public_board` — every agent's public round message, in round then agent-id order.
  2. `deceiver_reflections` — every deceiver's private reflection, in round then agent-id order.
  3. `honest_reflections` — every honest agent's private reflection, in round then agent-id order.

## Field Mapping
| Source | Docent target | Notes |
| --- | --- | --- |
| `question`, `correct_answer`, `answer_type`, `category`, `raw_subject` | `AgentRun.metadata` | verbatim, untruncated |
| `config.n_agents/n_deceivers/n_delib_rounds` | `AgentRun.metadata.N/k/n_rounds` | |
| `config.honest_model` | `AgentRun.metadata.family` | mapped via `trajectories.family_of` |
| `voted_after_round` / `len(rounds)` | `AgentRun.metadata.rounds_played` | |
| `grades.outcome.*` | `AgentRun.metadata.outcome` | full dict, e.g. honest_round0_correct, honest_vote_correct_frac |
| deceiver/honest agent ids | `AgentRun.metadata.deceiver_ids/honest_ids` | |
| `rounds[r].public_text[agent]`, `grades.public[r][agent]` | `public_board` messages | text + metadata correct/extracted_answer |
| `rounds[r].reflection_text[agent]`, `dynamics` correct-flags n/a (not present for these runs) | `deceiver_reflections` / `honest_reflections` messages | split by role |
| `votes_text[agent]`, `grades.votes[agent]` | one more `public_board` message per agent, stage label `vote` | the final-answer stage |
| message index of each (agent, round) reflection within its transcript | `AgentRun.metadata.reflection_index` | `{agent_id: {round: message_index}}`, built during construction, used for `transcript_slice` in the lulling reading |
| message index range of each round within `public_board` | `AgentRun.metadata.public_round_range` | `{round: [start_idx, end_idx]}` |
| local file path, condition name | `AgentRun.metadata.source_path/trial` | traceability back to the local JSON |

## Omitted Data
| Field/File | Reason | Impact |
| --- | --- | --- |
| Per-agent identity as separate transcripts (one transcript per agent, the ingestion guide's default multi-agent pattern) | Collapsed to 3 role-level transcripts instead, since the two target questions are about the *mechanism* (deceiver tactics in general, honest defection in general), not individual agent identity across games. Agent id is preserved in message metadata and text labels for citation. | Cross-agent-identity queries (e.g. "does agent 7 specifically escalate") are harder; not needed for this analysis. |
| `total_usage` (token/cost accounting), `seconds`, per-message `usage`/`route`/`attempts` API metadata | Not relevant to the behavioral question; already tracked in `runs/*/cost_report`. | None for this analysis. |
| Consecutive-`AssistantMessage` sanity warning | Intentional: each post/reflection is a distinct agent's turn, not a fragment of one model's output split across messages. | 26,784 instances across the 120-game batch; expected, not a conversion bug. |
| `reasoning_embedded_as_text` | Some reflections' free text happens to contain wording that pattern-matches reasoning markup; there is no separate structured reasoning stream in the source data to move it into. | 105 instances; cosmetic false positive, text preserved as-is. |
| `empty_message` | A handful of agents produced no visible text in a given round (occasional API refusals/empty completions in the underlying runs, including a few Grok coordination content-filter refusals). | 25 of ~26,900 messages (0.09%); left as empty text rather than dropped, so round/agent indexing stays intact for `transcript_slice`. |
| Remaining ~2,600 k>0 games in these four families, and all games in the coordinated conditions | Out of scope for this first pass; see Source Analysis. | None yet — flagged as the natural next step once the rubric is validated on this sample. |

## Confirmation
- Collection name: `mass-sabotage-persuasion-lulling`.
- Data context: multi-agent LLM deliberation games where k of N agents are secretly instructed to argue for
  a wrong answer; question is what persuades honest agents and what tactics deceivers use, across 4 model
  families (Gemini 3.8 Flash, DeepSeek V4.1 Flash, Muse Glimmer 30B, Grok 4.3).
- Analysis goals: (1) cluster deceiver persuasion tactics across families; (2) cluster what specifically
  causes honest agents to flip to an incorrect answer, tied to the already-computed flip events.
- Approval mode and source: auto-approved (`client.get_preferences().auto_approve_plans == True`).
- User confirmed, if required: N/A (auto-approved); plan and results reported in the session transcript.

## Execution Log
- 2026-09-19: `experimental_files/analysis/qualitative/select_games.py` selected 120 games (30/family x 4 families, stratified
  across ratios 0.20/0.33/0.43), written to `experimental_files/analysis/qualitative/selected_games.json`.
- 2026-09-19: `experimental_files/analysis/qualitative/ingest_dynamics.py` converted all 120 games with zero conversion errors,
  ran `check_agent_runs` (warnings all in the accepted categories above), created collection
  `mass-sabotage-persuasion-lulling`, uploaded all 120 AgentRuns in one batch, wrote
  `experimental_files/analysis/qualitative/ingested_index.json` (Docent ids + per-game index maps + matched flip events).
- 2026-09-19: DQL cannot embed a JSON array metadata field as literal prompt text (`as_type("text")`
  requires a scalar; `deceiver_ids`/`honest_ids` are JSON arrays). Patched all 120 runs via
  `update_agent_run_metadata` to add `deceiver_ids_text`/`honest_ids_text` (comma-joined "agent N"
  strings) alongside the existing array fields, for use in reading prompts.

## Verification
- Source records: 120 selected games (out of 1,966 k>0 games with flip events across the four main-grid families).
- Converted: 120/120, 0 failures.
- Uploaded: 120/120 (`upload_result.status == "success"`, 1 job, completed).
- Sanity warnings: accepted (consecutive_assistant_messages, reasoning_embedded_as_text, empty_message);
  see Omitted Data.
- Collection URL: https://docent.transluce.org/dashboard/0f6bde6e-f576-47b4-8477-b66d3eabf231
- Collection id: `0f6bde6e-f576-47b4-8477-b66d3eabf231`
