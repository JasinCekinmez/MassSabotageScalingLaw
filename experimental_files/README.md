# Experimental files

This directory is the release boundary for the paper experiments. Results are grouped by scientific condition rather than collection order.

- `prompts/` contains the exact honest, deceiver, and coordinated-deceiver templates.
- `question_selection/` contains the four-attempt HLE probes, selected question IDs, and difficulty buckets for each reported model family.
- `results/main_grid/` contains homogeneous-agent experiments.
- `results/coordinated_deceivers/` contains the private-coordination ablation.
- `results/heterogeneous/` contains all four honest/deceiver cross-model pairings.
- `results/model_selection/` contains the ELEPHANT and persuasion replications used to select models for the heterogeneous experiment.
- `analysis/` contains the main notebook, qualitative coding, frozen figure data, and exact plotting scripts.

The manuscript's inferential statistics are reproduced by
`analysis/main_statistics.py`, `analysis/coordination_statistics.py`,
`analysis/heterogeneous_statistics.py`, and `analysis/behavioral_statistics.py`.

[`manifest.json`](manifest.json) is the machine-readable mapping from paper conditions to result directories. [`run_experiments.sh`](run_experiments.sh) gives each condition a stable name and records the canonical reproduction commands.

Some result directories were filled incrementally and resumably. Their `args.json` records the most recent invocation, while `run_experiments.sh` records the complete intended condition. Full transcripts and aggregated CSVs are committed for auditability.

Exploratory GPT/Claude deliberation runs, smoke tests, temporary traces, and collection logs are not part of the release. They remain preserved locally under the ignored `local_archive/pre_release/` directory.
