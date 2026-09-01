#!/usr/bin/env bash
set -euo pipefail

# Deterministically evaluate saved reward-ablation policies using geometric
# goal reaching. Evaluation artifacts are written separately from training.
SOURCE_ROOT="${SOURCE_ROOT:-/home/guanglin/data_arcgym1/reward_ablation_results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/reward_ablation_results/evaluations_old_seed0}"
TEST_EPISODES="${TEST_EPISODES:-5}"
EVALUATION_SEEDS_CSV="${EVALUATION_SEEDS_CSV:-0,1}"
EVALUATION_EPISODE_LENGTH_S="${EVALUATION_EPISODE_LENGTH_S:-10}"
NUM_ENVS="${NUM_ENVS:-5}"
ALLOW_INTERMEDIATE="${ALLOW_INTERMEDIATE:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Source result directory does not exist: ${SOURCE_ROOT}" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to read the saved run configurations" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
IFS=',' read -r -a EVALUATION_SEEDS <<< "${EVALUATION_SEEDS_CSV}"
evaluated=0
skipped=0
missing=0

for source_dir in "${SOURCE_ROOT}"/*; do
  [[ -d "${source_dir}" && -f "${source_dir}/config.json" ]] || continue

  config_path="${source_dir}/config.json"
  colon_id="$(jq -r '.run_config.colon_id // "unknown"' "${config_path}")"
  task_id="$(jq -r '.run_config.task_id // "unknown"' "${config_path}")"
  seed="$(jq -r '.run_config.seed // "unknown"' "${config_path}")"
  algorithm="$(jq -r '.run_config.algorithm // "PPO"' "${config_path}")"
  reward_variant="$(jq -r '.run_config.reward_variant // "unknown"' "${config_path}")"
  constrained="$(jq -r '.run_config.constrained // false' "${config_path}")"

  # The paper reward ablation consists of these six Colon-1 settings only.
  [[ "${colon_id}" == "c1" && "${seed}" == "0" ]] || continue
  case "${task_id}" in t1|t2|t3|t4) ;; *) continue ;; esac
  case "${reward_variant}:${constrained}" in
    center_only:false|depth_existence:false|deep_area:false|lumen_evidence:false|full_reward:false|full_reward:true) ;;
    *) continue ;;
  esac

  checkpoint=""
  for candidate in "${source_dir}"/checkpoints/*_final_*.zip; do
    if [[ -f "${candidate}" ]]; then
      checkpoint="${candidate}"
      break
    fi
  done

  checkpoint_kind="final"
  if [[ -z "${checkpoint}" && "${ALLOW_INTERMEDIATE}" == "1" ]]; then
    checkpoint="$(
      find "${source_dir}/checkpoints" -maxdepth 1 -type f -name '*_steps.zip' -printf '%f\t%p\n' 2>/dev/null \
        | sort -V \
        | tail -n 1 \
        | cut -f2-
    )"
    checkpoint_kind="intermediate"
  fi

  if [[ -z "${checkpoint}" ]]; then
    echo "MISSING final checkpoint: $(basename "${source_dir}")" >&2
    missing=$((missing + 1))
    continue
  fi

  source_name="$(basename "${source_dir}")"
  constraint_arg="--no-clip_actions"
  if [[ "${constrained}" == "true" ]]; then
    constraint_arg="--clip_actions"
  fi

  for evaluation_seed in "${EVALUATION_SEEDS[@]}"; do
    evaluation_dir="${OUTPUT_ROOT}/${source_name}/eval-seed-${evaluation_seed}"
    if [[ -f "${evaluation_dir}/evaluation_metrics.json" && "${FORCE}" != "1" ]]; then
      echo "SKIP completed evaluation: ${source_name} evaluation_seed=${evaluation_seed}"
      skipped=$((skipped + 1))
      continue
    fi

    command=(
      python scripts/arc_env_rl.py
      --test
      --algo "${algorithm}"
      --seed "${evaluation_seed}"
      --training_seed "${seed}"
      --colon_id "${colon_id}"
      --task_id "${task_id}"
      --num_envs "${NUM_ENVS}"
      --reward_variant "${reward_variant}"
      "${constraint_arg}"
      --model_path "${checkpoint}"
      --test_episodes "${TEST_EPISODES}"
      --evaluation_episode_length_s "${EVALUATION_EPISODE_LENGTH_S}"
      --disable_video
      --result_dir "${evaluation_dir}"
    )

    echo "EVALUATE ${source_name} checkpoint=${checkpoint_kind} evaluation_seed=${evaluation_seed} episodes=${TEST_EPISODES}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '  %q' "${command[@]}"
      printf '\n'
    else
      ARC_CAMERA_HEIGHT=84 ARC_CAMERA_WIDTH=84 "${command[@]}"
    fi
    evaluated=$((evaluated + 1))
  done
done

echo "Evaluation selection: launched=${evaluated} skipped=${skipped} missing_final=${missing}"

if [[ "${DRY_RUN}" != "1" ]]; then
  python3 scripts/summarize_ablation_evaluation_repeats.py \
    --results-dir "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/evaluation_success_summary.csv"
fi

if (( missing > 0 )); then
  echo "Some policies lack final checkpoints. Set ALLOW_INTERMEDIATE=1 only if intermediate results will be labeled explicitly." >&2
fi
