#!/bin/bash
# Launch one MAPPO protoss_5_vs_5 training run to 10M steps as its own W&B run.
# Usage: ./launch_seed.sh <seed> <tag>   (tag e.g. shared_seed1)
set -u
SEED="$1"
TAG="$2"
cd /home/tasha/epymarl
export SC2PATH="$HOME/StarCraftII"
LOG="_run_logs/train_${TAG}.log"
# Ensure no stale resume env leaks in (would merge into the existing run c0tpvzrc).
env -u WANDB_RUN_ID -u WANDB_RESUME \
  python src/main.py --config=mappo --env-config=sc2v2 with \
  env_args.map_name=protoss_5_vs_5 \
  batch_size_run=32 batch_size=32 buffer_size=32 \
  t_max=10050000 seed="${SEED}" \
  use_wandb=True wandb_team=tashapais wandb_project=epymarl-smacv2-mappo wandb_mode=online \
  > "${LOG}" 2>&1 &
echo $! > "_run_logs/train_${TAG}.pid"
echo "Launched ${TAG} (seed ${SEED}) pid $(cat _run_logs/train_${TAG}.pid) -> ${LOG}"
