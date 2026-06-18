"""Record a video of trained EPyMARL agents in SMACv2 and (optionally) upload to W&B.

Builds the exact MAPPO/sc2v2 config used for training, loads a checkpoint into the
multi-agent controller, runs greedy episodes in a single render-capable SMACv2 env,
captures rgb_array frames headlessly (offscreen pygame), and writes an mp4.

Usage:
  python record_video.py --checkpoint results/models/<unique_token> [--load_step N] \
      --map protoss_5_vs_5 --episodes 6 --out results/videos/protoss_5v5.mp4 \
      [--wandb-run <run_id>]
"""
import os

# Headless rendering: offscreen SDL before pygame is imported anywhere.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

import argparse
import random as _random
import sys
from types import SimpleNamespace as SN

# pysc2's rendering code (colors.py) calls random.shuffle(seq, randfunc), whose
# second positional arg was removed in Python 3.10+. Restore the old behaviour so
# the SMAC renderer can be imported under Python 3.11.
_orig_shuffle = _random.shuffle


def _shuffle_compat(x, rand=None):
    if rand is None:
        return _orig_shuffle(x)
    for i in reversed(range(1, len(x))):
        j = int(rand() * (i + 1))
        x[i], x[j] = x[j], x[i]


_random.shuffle = _shuffle_compat

import numpy as np
import torch as th
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils.logging import Logger, get_logger  # noqa: E402
from controllers import REGISTRY as mac_REGISTRY  # noqa: E402
from components.episode_buffer import EpisodeBatch  # noqa: E402
from components.transforms import OneHot  # noqa: E402
from envs import REGISTRY as env_REGISTRY, register_smacv2  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")


def load_config():
    def _load(sub, name):
        with open(os.path.join(SRC, "config", sub, f"{name}.yaml")) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def merge(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = merge(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    with open(os.path.join(SRC, "config", "default.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = merge(cfg, _load("envs", "sc2v2"))
    cfg = merge(cfg, _load("algs", "mappo"))
    return cfg


def find_checkpoint_step(ckpt_dir, load_step):
    steps = [int(d) for d in os.listdir(ckpt_dir) if d.isdigit() and os.path.isdir(os.path.join(ckpt_dir, d))]
    if not steps:
        raise FileNotFoundError(f"No checkpoint timestep dirs under {ckpt_dir}")
    if load_step == 0:
        chosen = max(steps)
    else:
        chosen = min(steps, key=lambda x: abs(x - load_step))
    return os.path.join(ckpt_dir, str(chosen)), chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="", help="results/models/<unique_token> dir (empty = random policy, for pipeline testing)")
    ap.add_argument("--load_step", type=int, default=0)
    ap.add_argument("--map", default="protoss_5_vs_5")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--out", default="results/videos/protoss_5v5.mp4")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--only-wins", action="store_true", help="only keep frames from episodes the agents won")
    ap.add_argument("--replay-dir", default="", help="if set, save a .SC2Replay into this dir")
    ap.add_argument("--stop-on-win", action="store_true", help="save replay right after the first win and stop")
    args_cli = ap.parse_args()

    cfg = load_config()
    cfg["env_args"]["map_name"] = args_cli.map
    cfg["batch_size_run"] = 1
    cfg["runner"] = "episode"
    cfg["seed"] = args_cli.seed
    cfg["env_args"]["seed"] = args_cli.seed
    cfg["common_reward"] = True
    cfg["reward_scalarisation"] = "sum"
    cfg["use_cuda"] = th.cuda.is_available()
    if args_cli.replay_dir:
        os.makedirs(os.path.abspath(args_cli.replay_dir), exist_ok=True)
        cfg["env_args"]["replay_dir"] = os.path.abspath(args_cli.replay_dir)
        cfg["env_args"]["replay_prefix"] = "mappo_protoss5v5"

    args = SN(**cfg)
    args.device = "cuda" if args.use_cuda else "cpu"

    np.random.seed(args_cli.seed)
    th.manual_seed(args_cli.seed)

    logger = Logger(get_logger())

    register_smacv2()
    env = env_REGISTRY["sc2v2"](
        **args.env_args, common_reward=args.common_reward, reward_scalarisation=args.reward_scalarisation
    )
    env_info = env.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,)},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    mac = mac_REGISTRY[args.mac](scheme, groups, args)
    if args_cli.checkpoint:
        model_path, step = find_checkpoint_step(args_cli.checkpoint, args_cli.load_step)
        logger.console_logger.info(f"Loading agent from {model_path} (step {step})")
        mac.load_models(model_path)
    else:
        logger.console_logger.info("No checkpoint given -> random policy (pipeline test only)")
    if args.use_cuda:
        mac.cuda()

    def new_batch():
        return EpisodeBatch(
            scheme, groups, 1, env_info["episode_limit"] + 1, preprocess=preprocess, device=args.device
        )

    all_frames = []
    n_wins = 0
    for ep in range(args_cli.episodes):
        env.reset()
        mac.init_hidden(batch_size=1)
        batch = new_batch()
        terminated = False
        t = 0
        ep_frames = []
        won = False
        while not terminated:
            pre = {
                "state": [env.get_state()],
                "avail_actions": [env.get_avail_actions()],
                "obs": [env.get_obs()],
            }
            batch.update(pre, ts=t)
            actions = mac.select_actions(batch, t_ep=t, t_env=0, test_mode=True)
            _, reward, terminated, truncated, env_info_step = env.step(actions[0])
            terminated = terminated or truncated
            frame = env.env.render(mode="rgb_array")  # SMACv2Wrapper -> StarCraftCapabilityEnvWrapper
            ep_frames.append(np.asarray(frame, dtype=np.uint8))
            batch.update({"actions": actions, "terminated": [(terminated,)], "reward": [(reward,)]}, ts=t)
            t += 1
            if env_info_step.get("battle_won", False):
                won = True
        n_wins += int(won)
        logger.console_logger.info(f"Episode {ep+1}/{args_cli.episodes}: len={t} won={won}")
        if (not args_cli.only_wins) or won:
            all_frames.extend(ep_frames)
        if args_cli.replay_dir and won and args_cli.stop_on_win:
            logger.console_logger.info("Won -> saving replay of this game and stopping early")
            env.save_replay()
            break
    else:
        if args_cli.replay_dir:
            logger.console_logger.info("Saving replay of final game session")
            env.save_replay()

    env.close()
    logger.console_logger.info(f"Won {n_wins}/{args_cli.episodes} episodes; {len(all_frames)} frames captured")

    if not all_frames:
        logger.console_logger.info("No frames to write (no winning episodes?). Exiting.")
        return

    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    imageio.mimsave(args_cli.out, all_frames, fps=args_cli.fps, macro_block_size=1)
    logger.console_logger.info(f"Saved video: {args_cli.out} ({len(all_frames)} frames)")

    if args_cli.__dict__.get("wandb_run"):
        pass  # handled by separate upload step


if __name__ == "__main__":
    main()
