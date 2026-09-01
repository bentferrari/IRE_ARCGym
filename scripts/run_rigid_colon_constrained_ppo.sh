#!/usr/bin/env bash
set -euo pipefail

# Matched rigid-colon comparison using constrained PPO and the full reward.
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
TASKS_CSV="${TASKS_CSV:-t2,t4}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/guanglin/data_arcgym1/rigid_colon_constrained_ppo_results}"
NUM_ENVS="${NUM_ENVS:-5}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
PPO_N_STEPS="${PPO_N_STEPS:-1600}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-1000}"
TRAIN_EPISODES="${TRAIN_EPISODES:-300}"
TASK2_TRAIN_EPISODES="${TASK2_TRAIN_EPISODES:-30}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"

mkdir -p "${RESULTS_ROOT}"

for seed in "${SEEDS[@]}"; do
  for task_id in "${TASKS[@]}"; do
    stop_after_episodes="${TRAIN_EPISODES}"
    if [[ "${task_id}" == "t2" ]]; then
      stop_after_episodes="${TASK2_TRAIN_EPISODES}"
    fi

    result_dir="${RESULTS_ROOT}/ppo_full_reward_constrained_rigid_colon-c1_task-${task_id}_seed-${seed}_${RUN_TAG}"
    echo "TRAIN rigid Colon 1 task=${task_id} seed=${seed} stop_after_episodes=${stop_after_episodes}"

    ARC_CAMERA_HEIGHT=84 ARC_CAMERA_WIDTH=84 \
      python scripts/arc_env_rl.py \
        --train \
        --algo PPO \
        --seed "${seed}" \
        --colon_id c1 \
        --task_id "${task_id}" \
        --num_envs "${NUM_ENVS}" \
        --reward_variant full_reward \
        --clip_actions \
        --rigid_colon \
        --total_timesteps "${TOTAL_TIMESTEPS}" \
        --stop_after_episodes "${stop_after_episodes}" \
        --ppo_n_steps "${PPO_N_STEPS}" \
        --ppo_batch_size "${PPO_BATCH_SIZE}" \
        --disable_video \
        --checkpoint_save_freq 0 \
        --save_best_and_last_only \
        --trajectory_save_interval 1000000 \
        --no-save_trajectory_images \
        --result_dir "${result_dir}" \
        "$@"
  done
done
