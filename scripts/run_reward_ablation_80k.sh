#!/usr/bin/env bash
set -euo pipefail

# Colon-1 reward study stopped by completed episodes: 300 for T1/T3/T4 and
# 30 for T2. TOTAL_TIMESTEPS is only a safety ceiling.
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
TASKS_CSV="${TASKS_CSV:-t1,t2,t3,t4}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/guanglin/data_arcgym1/reward_ablation_results_episode_stopped}"
NUM_ENVS="${NUM_ENVS:-5}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
TRAIN_EPISODES="${TRAIN_EPISODES:-300}"
TASK2_TRAIN_EPISODES="${TASK2_TRAIN_EPISODES:-30}"
PPO_N_STEPS="${PPO_N_STEPS:-1600}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-1000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
SETTINGS=(
  "center_only:unconstrained"
  "depth_existence:unconstrained"
  "deep_area:unconstrained"
  "lumen_evidence:unconstrained"
  "full_reward:unconstrained"
  "full_reward:constrained"
)

ROLLOUT_TRANSITIONS=$((NUM_ENVS * PPO_N_STEPS))
if (( TOTAL_TIMESTEPS % ROLLOUT_TRANSITIONS != 0 )); then
  echo "TOTAL_TIMESTEPS must be divisible by NUM_ENVS * PPO_N_STEPS (${ROLLOUT_TRANSITIONS})" >&2
  exit 2
fi
if (( ROLLOUT_TRANSITIONS % PPO_BATCH_SIZE != 0 )); then
  echo "NUM_ENVS * PPO_N_STEPS must be divisible by PPO_BATCH_SIZE" >&2
  exit 2
fi

mkdir -p "${RESULTS_ROOT}"

for seed in "${SEEDS[@]}"; do
  for task_id in "${TASKS[@]}"; do
    stop_after_episodes="${TRAIN_EPISODES}"
    if [[ "${task_id}" == "t2" ]]; then
      stop_after_episodes="${TASK2_TRAIN_EPISODES}"
    fi
    for setting in "${SETTINGS[@]}"; do
      reward_variant="${setting%%:*}"
      constraint_label="${setting##*:}"
      if [[ "${constraint_label}" == "constrained" ]]; then
        constraint_arg="--clip_actions"
      else
        constraint_arg="--no-clip_actions"
      fi

      run_name="ppo_${reward_variant}_${constraint_label}_colon-c1_task-${task_id}_seed-${seed}_${RUN_TAG}"
      result_dir="${RESULTS_ROOT}/${run_name}"
      echo "Running seed=${seed} task=${task_id} reward=${reward_variant} ${constraint_label} stop_after_episodes=${stop_after_episodes}"

      ARC_CAMERA_HEIGHT=84 ARC_CAMERA_WIDTH=84 \
        python scripts/arc_env_rl.py \
          --train \
          --algo PPO \
          --seed "${seed}" \
          --colon_id c1 \
          --task_id "${task_id}" \
          --num_envs "${NUM_ENVS}" \
          --reward_variant "${reward_variant}" \
          "${constraint_arg}" \
          --total_timesteps "${TOTAL_TIMESTEPS}" \
          --stop_after_episodes "${stop_after_episodes}" \
          --ppo_n_steps "${PPO_N_STEPS}" \
          --ppo_batch_size "${PPO_BATCH_SIZE}" \
          --eval_after_train \
          --test_episodes "${EVAL_EPISODES}" \
          --disable_video \
          --checkpoint_save_freq 0 \
          --trajectory_save_interval 1000000 \
          --no-save_trajectory_images \
          --result_dir "${result_dir}" \
          "$@"
    done
  done
done

python3 scripts/aggregate_reward_ablation.py \
  --results_dir "${RESULTS_ROOT}" \
  --output_dir "${RESULTS_ROOT}/analysis_${RUN_TAG}" \
  --expected_colons c1 \
  --expected_tasks "${TASKS_CSV}" \
  --expected_seeds "${SEEDS_CSV}" \
  --strict
