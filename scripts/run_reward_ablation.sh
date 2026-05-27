#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-0}"
COLON_ID="${COLON_ID:-c1}"
TASK_ID="${TASK_ID:-t2}"
NUM_ENVS="${NUM_ENVS:-5}"
ALGO="${ALGO:-PPO}"
TEST_EPISODES="${TEST_EPISODES:-5}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1e7}"
ABLATION_EPISODES="${ABLATION_EPISODES:-300}"

# Additional args are forwarded to arc_env_rl.py, e.g.:
#   ./scripts/run_reward_ablation.sh --reward_beta 0.7
EXTRA_ARGS=("$@")

case "${TASK_ID}" in
  t1|t2|t3|t4)
    ABLATION_TASK_ARG="--ablation_${TASK_ID}"
    ;;
  *)
    echo "TASK_ID must be one of t1, t2, t3, or t4 for reward ablation runs" >&2
    exit 1
    ;;
esac

# Colon/task selection uses both:
#   saved_states/${COLON_ID}${TASK_ID}_start.csv
#   saved_states/${COLON_ID}${TASK_ID}_end.csv
COMMON_ARGS=(
  --num_envs "${NUM_ENVS}"
  --algo "${ALGO}"
  --seed "${SEED}"
  --colon_id "${COLON_ID}"
  --test_episodes "${TEST_EPISODES}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --ablation_episodes "${ABLATION_EPISODES}"
  --eval_after_train
)

python scripts/arc_env_rl.py "${COMMON_ARGS[@]}" "${ABLATION_TASK_ARG}" "${EXTRA_ARGS[@]}"
