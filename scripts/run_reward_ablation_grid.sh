#!/usr/bin/env bash
set -euo pipefail

# Reviewer-response grid: 6 reward/policy variants x 2 colons x 4 tasks x 3 seeds.
# Runs are deliberately sequential because each child launches Isaac Sim on one GPU.
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
COLONS_CSV="${COLONS_CSV:-c1,c2}"
TASKS_CSV="${TASKS_CSV:-t1,t2,t3,t4}"
NUM_ENVS="${NUM_ENVS:-5}"
ALGO="${ALGO:-PPO}"
TRAIN_EPISODES="${TRAIN_EPISODES:-300}"
TASK2_TRAIN_EPISODES="${TASK2_TRAIN_EPISODES:-30}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1e7}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a COLONS <<< "${COLONS_CSV}"
IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"

for seed in "${SEEDS[@]}"; do
  for colon_id in "${COLONS[@]}"; do
    for task_id in "${TASKS[@]}"; do
      task_train_episodes="${TRAIN_EPISODES}"
      if [[ "${task_id}" == "t2" ]]; then
        task_train_episodes="${TASK2_TRAIN_EPISODES}"
      fi
      echo "Running reward ablation: seed=${seed} colon=${colon_id} task=${task_id}"
      SEED="${seed}" \
      COLON_ID="${colon_id}" \
      TASK_ID="${task_id}" \
      NUM_ENVS="${NUM_ENVS}" \
      ALGO="${ALGO}" \
      TEST_EPISODES="${EVAL_EPISODES}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      ABLATION_EPISODES="${task_train_episodes}" \
        ./scripts/run_reward_ablation.sh \
          --disable_video \
          --checkpoint_save_freq 0 \
          --trajectory_save_interval 1000000 \
          --no-save_trajectory_images \
          "$@"
    done
  done
done

python3 scripts/aggregate_reward_ablation.py \
  --expected_colons "${COLONS_CSV}" \
  --expected_tasks "${TASKS_CSV}" \
  --expected_seeds "${SEEDS_CSV}" \
  --strict
