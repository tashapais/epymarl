"""Upload a trained EPyMARL checkpoint to the HuggingFace Hub.

Uploads the checkpoint files (agent.th, critic.th, optimiser states), the merged
training config, and a generated model card to a model repo on the Hub.
"""
import argparse
import json
import os
from huggingface_hub import HfApi, whoami


CARD = """---
license: apache-2.0
library_name: epymarl
tags:
- multi-agent-reinforcement-learning
- marl
- mappo
- smacv2
- starcraft
---

# MAPPO on SMACv2 {map_name}

Multi-Agent PPO (MAPPO) agents trained with [EPyMARL](https://github.com/uoe-agents/epymarl)
on the SMACv2 `{map_name}` scenario (StarCraft II).

- **Algorithm:** MAPPO (shared parameters, RNN policy, centralised value function)
- **Environment:** SMACv2 `{map_name}`
- **Checkpoint step:** {step} environment timesteps
- **Greedy test win rate at upload:** {win_rate}

## Files
- `agent.th` — actor network weights
- `critic.th` — centralised critic weights
- `*_opt.th` — optimiser states
- `config.json` — full training configuration

## Usage
Load into EPyMARL by pointing `checkpoint_path` at a directory containing a
`{step}/` subfolder with these files:

```sh
python src/main.py --config=mappo --env-config=sc2v2 \\
    with env_args.map_name={map_name} checkpoint_path=<dir> evaluate=True render=False
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-step-dir", required=True, help="results/models/<token>/<step> dir with .th files")
    ap.add_argument("--repo", default=None, help="repo id; default <user>/epymarl-mappo-smacv2-<map>")
    ap.add_argument("--map", default="protoss_5_vs_5")
    ap.add_argument("--win-rate", default="unknown")
    ap.add_argument("--config-json", default=None, help="optional path to a config.json to include")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    step = os.path.basename(os.path.normpath(args.checkpoint_step_dir))
    user = whoami()["name"]
    repo_id = args.repo or f"{user}/epymarl-mappo-smacv2-{args.map.replace('_', '-')}"

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

    # Stage card + config alongside checkpoint files via a temp upload.
    files = sorted(os.listdir(args.checkpoint_step_dir))
    print(f"Checkpoint files: {files}")

    card = CARD.format(map_name=args.map, step=step, win_rate=args.win_rate)
    card_path = os.path.join(args.checkpoint_step_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card)

    if args.config_json and os.path.exists(args.config_json):
        with open(args.config_json) as f:
            cfg = json.load(f)
        with open(os.path.join(args.checkpoint_step_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    api.upload_folder(
        folder_path=args.checkpoint_step_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload MAPPO SMACv2 {args.map} checkpoint @ step {step}",
    )
    print(f"Uploaded checkpoint to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
