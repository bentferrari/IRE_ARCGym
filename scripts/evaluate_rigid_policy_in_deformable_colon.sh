#!/usr/bin/env bash
set -euo pipefail

# Evaluate rigid-colon-trained policies in the deformable Colon-1 environment.
SOURCE_ROOT="${SOURCE_ROOT:-/home/guanglin/data_arcgym1/rigid_colon_constrained_ppo_results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/guanglin/data_arcgym1/rigid_to_deformable_evaluations}"
EVALUATION_SEEDS_CSV="${EVALUATION_SEEDS_CSV:-0,1}"
CHECKPOINT_KINDS_CSV="${CHECKPOINT_KINDS_CSV:-best,last}"
TEST_EPISODES="${TEST_EPISODES:-5}"
NUM_ENVS="${NUM_ENVS:-5}"
EVALUATION_EPISODE_LENGTH_S="${EVALUATION_EPISODE_LENGTH_S:-10}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a EVALUATION_SEEDS <<< "${EVALUATION_SEEDS_CSV}"
IFS=',' read -r -a CHECKPOINT_KINDS <<< "${CHECKPOINT_KINDS_CSV}"
mkdir -p "${OUTPUT_ROOT}"
selected=0

for source_dir in "${SOURCE_ROOT}"/*; do
  [[ -d "${source_dir}" && -f "${source_dir}/config.json" ]] || continue
  config_path="${source_dir}/config.json"
  [[ "$(jq -r '.run_config.rigid_colon // .env_config.rigid_colon // false' "${config_path}")" == "true" ]] || continue

  checkpoints=()
  checkpoint_labels=()
  for checkpoint_kind in "${CHECKPOINT_KINDS[@]}"; do
    candidate="${source_dir}/checkpoints/ppo_arc_${checkpoint_kind}.zip"
    if [[ -f "${candidate}" ]]; then
      checkpoints+=("${candidate}")
      checkpoint_labels+=("${checkpoint_kind}")
    fi
  done
  if (( ${#checkpoints[@]} == 0 )); then
    for candidate in "${source_dir}"/checkpoints/*_final_*.zip; do
      if [[ -f "${candidate}" ]]; then
        checkpoints+=("${candidate}")
        checkpoint_labels+=("final")
        break
      fi
    done
  fi
  (( ${#checkpoints[@]} > 0 )) || continue

  colon_id="$(jq -r '.run_config.colon_id' "${config_path}")"
  task_id="$(jq -r '.run_config.task_id' "${config_path}")"
  training_seed="$(jq -r '.run_config.training_seed // .run_config.seed' "${config_path}")"
  algorithm="$(jq -r '.run_config.algorithm // "PPO"' "${config_path}")"
  reward_variant="$(jq -r '.run_config.reward_variant // "full_reward"' "${config_path}")"
  constrained="$(jq -r '.run_config.constrained // true' "${config_path}")"
  constraint_arg="--no-clip_actions"
  if [[ "${constrained}" == "true" ]]; then constraint_arg="--clip_actions"; fi
  source_name="$(basename "${source_dir}")"

  for checkpoint_index in "${!checkpoints[@]}"; do
    checkpoint="${checkpoints[checkpoint_index]}"
    checkpoint_kind="${checkpoint_labels[checkpoint_index]}"
    policy_name="${source_name}_${checkpoint_kind}"
    for evaluation_seed in "${EVALUATION_SEEDS[@]}"; do
    evaluation_dir="${OUTPUT_ROOT}/${policy_name}/eval-seed-${evaluation_seed}"
    if [[ -f "${evaluation_dir}/evaluation_metrics.json" && "${FORCE}" != "1" ]]; then
      echo "SKIP completed: ${policy_name} evaluation_seed=${evaluation_seed}"
      continue
    fi
    command=(
      python scripts/arc_env_rl.py
      --test
      --algo "${algorithm}"
      --seed "${evaluation_seed}"
      --training_seed "${training_seed}"
      --colon_id "${colon_id}"
      --task_id "${task_id}"
      --num_envs "${NUM_ENVS}"
      --reward_variant "${reward_variant}"
      "${constraint_arg}"
      --no-rigid_colon
      --model_path "${checkpoint}"
      --test_episodes "${TEST_EPISODES}"
      --evaluation_episode_length_s "${EVALUATION_EPISODE_LENGTH_S}"
      --disable_video
      --result_dir "${evaluation_dir}"
    )
    echo "EVALUATE rigid-trained ${checkpoint_kind} policy in deformable colon: ${source_name} evaluation_seed=${evaluation_seed}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '  %q' "${command[@]}"
      printf '\n'
    else
      ARC_CAMERA_HEIGHT=84 ARC_CAMERA_WIDTH=84 "${command[@]}"
    fi
    selected=$((selected + 1))
    done
  done
done

if (( selected == 0 )); then
  echo "No unevaluated rigid-policy checkpoints found under ${SOURCE_ROOT}" >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  python3 scripts/summarize_ablation_evaluation_repeats.py \
    --results-dir "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/evaluation_success_summary.csv"
fi
