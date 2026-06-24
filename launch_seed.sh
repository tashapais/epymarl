#!/bin/bash
# Launch one MAPPO protoss_5_vs_5 training run to 10M steps as its own W&B run.
# Usage: ./launch_seed.sh <seed> <tag> [extra sacred args...]
#   e.g. ./launch_seed.sh 1 shared_seed1
#        ./launch_seed.sh 1 individual_seed1 common_reward=False reward_scalarisation=sum
# NOTE: save_model=True is essential -- without it no checkpoints are written and
# the representation metrics cannot be computed afterwards.
set -u
SEED="$1"
TAG="$2"
shift 2
EXTRA="$*"
cd /home/tasha/epymarl
export SC2PATH="$HOME/StarCraftII"
LOG="_run_logs/train_${TAG}.log"
# Ensure no stale resume env leaks in (would merge into the existing run c0tpvzrc).
env -u WANDB_RUN_ID -u WANDB_RESUME \
  python src/main.py --config=mappo --env-config=sc2v2 with \
  env_args.map_name=protoss_5_vs_5 \
  batch_size_run=32 batch_size=32 buffer_size=32 \
  t_max=10050000 seed="${SEED}" \
  save_model=True save_model_interval=2000000 \
  use_wandb=True wandb_team=tashapais wandb_project=epymarl-smacv2-mappo wandb_mode=online \
  ${EXTRA} \
  > "${LOG}" 2>&1 &
echo $! > "_run_logs/train_${TAG}.pid"
echo "Launched ${TAG} (seed ${SEED}) extra='${EXTRA}' pid $(cat _run_logs/train_${TAG}.pid) -> ${LOG}"
