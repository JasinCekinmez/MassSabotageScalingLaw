#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

target="all"
if [[ $# -gt 0 && "$1" != --* ]]; then
  target="$1"
  shift
fi
extra=("$@")
parallel_games="${SABOTAGE_PARALLEL_GAMES:-4}"

run_condition() {
  local name="$1"
  shift
  if [[ "$target" == "all" || "$target" == "$name" ]]; then
    echo "==> $name"
    python -m sabotage.run "$@" "${extra[@]}"
  fi
}

common=(--rounds 7 --seeds 0 --judge-model gpt-5.4 --judge-provider openai --parallel-games "$parallel_games")
all_groups="2+0,2+1,4+0,4+2,4+3,8+0,8+2,8+4,8+6,12+3,12+6,12+9"
coord_groups="4+2,4+3,8+2,8+4,8+6,12+3,12+6,12+9"

# Homogeneous main grid. Gemini was collected in four batches with a fixed
# number of honest agents; the other families use one directory each.
run_condition gemini-main-h2 \
  --ids-file experimental_files/question_selection/gemini-3.8-flash/selected.json \
  --groups 2+0,2+1 --models gemini-3.8-flash --effort none --max-tokens 32000 \
  --out experimental_files/results/main_grid/gemini-3.8-flash/honest_agents_2 "${common[@]}"

run_condition gemini-main-h4 \
  --ids-file experimental_files/question_selection/gemini-3.8-flash/selected.json \
  --groups 4+0,4+2,4+3 --models gemini-3.8-flash --effort none --max-tokens 32000 \
  --out experimental_files/results/main_grid/gemini-3.8-flash/honest_agents_4 "${common[@]}"

run_condition gemini-main-h8 \
  --ids-file experimental_files/question_selection/gemini-3.8-flash/selected.json \
  --groups 8+0,8+2,8+4,8+6 --models gemini-3.8-flash --effort none --max-tokens 32000 \
  --out experimental_files/results/main_grid/gemini-3.8-flash/honest_agents_8 "${common[@]}"

run_condition gemini-main-h12 \
  --ids-file experimental_files/question_selection/gemini-3.8-flash/selected.json \
  --groups 12+3,12+6,12+9 --models gemini-3.8-flash --effort none --max-tokens 32000 \
  --out experimental_files/results/main_grid/gemini-3.8-flash/honest_agents_12 "${common[@]}"

run_condition grok-main \
  --ids-file experimental_files/question_selection/grok-4.3/selected_run.json \
  --groups "$all_groups" --models grok-4.3 --effort none --max-tokens 32000 --stop-after 4 \
  --out experimental_files/results/main_grid/grok-4.3 "${common[@]}"

run_condition deepseek-main \
  --ids-file experimental_files/question_selection/deepseek-v4.1-flash/selected_3q.json \
  --groups "$all_groups" --models together/deepseek-ai/DeepSeek-V4.1-Flash --effort none --max-tokens 48000 --stop-after 4 \
  --out experimental_files/results/main_grid/deepseek-v4.1-flash "${common[@]}"

run_condition glimmer-main \
  --ids-file experimental_files/question_selection/muse-glimmer-30b/selected.json \
  --groups "$all_groups" --models together/meta-models/Muse-Glimmer-30B --effort none --max-tokens 32000 --stop-after 4 \
  --out experimental_files/results/main_grid/muse-glimmer-30b "${common[@]}"

# Private deceiver coordination. Only compositions with at least two
# deceivers enter the reported matched comparison.
run_condition gemini-coordinated \
  --ids-file experimental_files/question_selection/gemini-3.8-flash/selected.json \
  --groups "$coord_groups" --models gemini-3.8-flash --coordinate --stop-after 4 --effort none --max-tokens 32000 \
  --out experimental_files/results/coordinated_deceivers/gemini-3.8-flash "${common[@]}"

run_condition grok-coordinated \
  --ids-file experimental_files/question_selection/grok-4.3/selected_run.json \
  --groups "$coord_groups" --models grok-4.3 --coordinate --stop-after 4 --effort none --max-tokens 32000 \
  --out experimental_files/results/coordinated_deceivers/grok-4.3 "${common[@]}"

# Heterogeneous pairings use the common Gemini-selected HLE set.
heterogeneous_groups="4+1,4+3,8+2,8+6"
heterogeneous_ids="experimental_files/question_selection/gemini-3.8-flash/selected.json"

run_condition heterogeneous-gemini-deepseek \
  --ids-file "$heterogeneous_ids" --groups "$heterogeneous_groups" --stop-after 4 --effort none --max-tokens 32000 \
  --honest-model gemini-3.8-flash --deceiver-model together/deepseek-ai/DeepSeek-V4.1-Flash \
  --out experimental_files/results/heterogeneous/gemini-honest_deepseek-deceiver "${common[@]}"

run_condition heterogeneous-gemini-grok \
  --ids-file "$heterogeneous_ids" --groups "$heterogeneous_groups" --stop-after 4 --effort none --max-tokens 32000 \
  --honest-model gemini-3.8-flash --deceiver-model grok-4.3 \
  --out experimental_files/results/heterogeneous/gemini-honest_grok-deceiver "${common[@]}"

run_condition heterogeneous-glimmer-deepseek \
  --ids-file "$heterogeneous_ids" --groups "$heterogeneous_groups" --stop-after 4 --effort none --max-tokens 32000 \
  --honest-model together/meta-models/Muse-Glimmer-30B --deceiver-model together/deepseek-ai/DeepSeek-V4.1-Flash \
  --out experimental_files/results/heterogeneous/glimmer-honest_deepseek-deceiver "${common[@]}"

run_condition heterogeneous-glimmer-grok \
  --ids-file "$heterogeneous_ids" --groups "$heterogeneous_groups" --stop-after 4 --effort none --max-tokens 32000 \
  --honest-model together/meta-models/Muse-Glimmer-30B --deceiver-model grok-4.3 \
  --out experimental_files/results/heterogeneous/glimmer-honest_grok-deceiver "${common[@]}"

if [[ "$target" != "all" ]] && ! grep -q "run_condition $target " "$0"; then
  echo "unknown condition: $target" >&2
  exit 2
fi
