"""Representation-geometry metrics for a trained EPyMARL MAPPO checkpoint on SMACv2.

Loads a checkpoint, runs greedy eval episodes in a single render-free SMACv2 env,
and records per-(episode, timestep, agent): the shared encoder's GRU hidden state
(128-d embedding), the (masked) action distribution, the available-action mask, the
agent's unit type, and an alive flag. From these it computes the three paper metrics:

  * EffRank/n  -- effective rank (Roy & Vetterli) of the per-agent time-averaged
                 embedding centroids, normalised by agent count (paper teaser
                 definition, <= 1 scale). Also reports the global (all
                 timestep x agent) effective rank for cross-reference.
  * D_act      -- mean pairwise KL between agents' action distributions,
                 (1/C(n,2)) sum_{i!=j} KL(pi_i || pi_j), averaged over timesteps.
                 MASK-AWARE: KL is taken over each pair's renormalised
                 available-action support (union, epsilon-smoothed), so it is not
                 dominated by which actions are masked.
  * Probe      -- logistic regression decoding unit type from the embedding, with
                 leave-one-agent-slot-out CV (chance = 1/#unit_types).

Writes a JSON summary. Usage:
  python metrics_repr.py --checkpoint "results/models/<token>" --load_step N \
      --map protoss_5_vs_5 --episodes 32 --out results/metrics/<tag>.json --tag <tag>
"""
import os

os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

import argparse
import json
import sys
from itertools import combinations
from types import SimpleNamespace as SN

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
            d[k] = merge(d.get(k, {}), v) if isinstance(v, dict) else v
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
    chosen = max(steps) if load_step == 0 else min(steps, key=lambda x: abs(x - load_step))
    return os.path.join(ckpt_dir, str(chosen)), chosen


def effective_rank(M):
    """Roy & Vetterli effective rank of matrix M (rows = samples). exp(entropy of
    normalised singular value spectrum). No centering (matches the paper eqn)."""
    s = np.linalg.svd(np.asarray(M, dtype=np.float64), compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def unit_types_from_env(env, n_agents):
    """Best-effort per-agent SC2 unit-type id from the underlying StarCraft2Env."""
    base = env.env  # StarCraftCapabilityEnvWrapper
    sc = getattr(base, "env", base)  # underlying StarCraft2Env
    types = []
    for i in range(n_agents):
        t = -1
        try:
            u = sc.get_unit_by_id(i)
            t = int(u.unit_type)
        except Exception:
            try:
                t = int(sc.agents[i].unit_type)
            except Exception:
                t = -1
        types.append(t)
    return types


def mask_aware_dact(dists, avails):
    """(1/#pairs) sum_{i!=j} KL(p_i||p_j), mask-aware. KL is taken over each pair's
    INTERSECTION of available actions (renormalised), so it reflects policy
    preference over commonly-available actions, not which actions are masked."""
    n = len(dists)
    if n < 2:
        return None
    eps = 1e-12
    total = 0.0
    cnt = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sup = (avails[i] > 0) & (avails[j] > 0)  # commonly-available actions
            if sup.sum() < 2:
                continue
            p = dists[i][sup] + eps
            p = p / p.sum()
            q = dists[j][sup] + eps
            q = q / q.sum()
            total += float((p * np.log(p / q)).sum())
            cnt += 1
    return total / cnt if cnt else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--load_step", type=int, default=0)
    ap.add_argument("--map", default="protoss_5_vs_5")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default="results/metrics/metrics.json")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--wandb-run", default="")
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
    n_agents = args.n_agents

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,)},
    }
    groups = {"agents": n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    mac = mac_REGISTRY[args.mac](scheme, groups, args)
    model_path, step = find_checkpoint_step(args_cli.checkpoint, args_cli.load_step)
    logger.console_logger.info(f"Loading agent from {model_path} (step {step})")
    mac.load_models(model_path)
    if args.use_cuda:
        mac.cuda()

    def new_batch():
        return EpisodeBatch(scheme, groups, 1, env_info["episode_limit"] + 1, preprocess=preprocess, device=args.device)

    H, D, AV, UT, SLOT, ALIVE = [], [], [], [], [], []  # per-row stores
    per_t = []  # list of (dists, avails) for alive agents, for D_act per timestep
    n_wins = 0
    for ep in range(args_cli.episodes):
        env.reset()
        mac.init_hidden(batch_size=1)
        batch = new_batch()
        terminated = False
        t = 0
        won = False
        while not terminated:
            avail = np.asarray(env.get_avail_actions())  # (n_agents, n_actions)
            pre = {"state": [env.get_state()], "avail_actions": [avail], "obs": [env.get_obs()]}
            batch.update(pre, ts=t)
            probs = mac.forward(batch, t=t, test_mode=True)  # (1, n_agents, n_actions) masked softmax
            hid = mac.hidden_states.detach().cpu().numpy().reshape(n_agents, -1)  # (n_agents, hidden)
            probs_np = probs.detach().cpu().numpy().reshape(n_agents, -1)
            utypes = unit_types_from_env(env, n_agents)

            # greedy actions from masked probs
            actions = th.tensor(np.argmax(probs_np, axis=1).reshape(1, n_agents, 1), device=args.device)
            _, reward, terminated, truncated, info = env.step(actions[0])
            terminated = terminated or truncated

            alive_dists, alive_avail = [], []
            for a in range(n_agents):
                alive = int(avail[a].sum() > 1)  # >1 avail action => not dead (dead = only no-op)
                H.append(hid[a]); D.append(probs_np[a]); AV.append(avail[a])
                UT.append(utypes[a]); SLOT.append(a); ALIVE.append(alive)
                if alive:
                    alive_dists.append(probs_np[a]); alive_avail.append(avail[a])
            if len(alive_dists) >= 2:
                per_t.append((np.array(alive_dists), np.array(alive_avail)))

            batch.update({"actions": actions, "terminated": [(terminated,)], "reward": [(reward,)]}, ts=t)
            t += 1
            if info.get("battle_won", False):
                won = True
        n_wins += int(won)
        logger.console_logger.info(f"Episode {ep+1}/{args_cli.episodes}: len={t} won={won}")
    env.close()

    H = np.array(H); D = np.array(D); AV = np.array(AV)
    UT = np.array(UT); SLOT = np.array(SLOT); ALIVE = np.array(ALIVE)
    alive_mask = ALIVE == 1

    # ---- EffRank/n: per-agent-slot time-averaged centroids over alive rows ----
    centroids = []
    for a in range(n_agents):
        m = alive_mask & (SLOT == a)
        if m.sum() > 0:
            centroids.append(H[m].mean(axis=0))
    centroids = np.array(centroids)
    effrank_centroid = effective_rank(centroids)
    effrank_n = effrank_centroid / n_agents
    # global effrank over all alive embeddings (~ embedding-dim scale), for reference
    effrank_global = effective_rank(H[alive_mask])

    # ---- D_act: mean over timesteps of mask-aware pairwise KL ----
    dact_vals = [mask_aware_dact(d, a) for d, a in per_t]
    dact_vals = [v for v in dact_vals if v is not None]
    dact_mean = float(np.mean(dact_vals)) if dact_vals else None

    # ---- Probe: unit-type decode from embedding, leave-one-slot-out CV ----
    probe_acc, probe_chance, n_types = None, None, None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        Xm = alive_mask & (UT >= 0)
        X, y, slot = H[Xm], UT[Xm], SLOT[Xm]
        classes = sorted(set(y.tolist()))
        n_types = len(classes)
        probe_chance = 1.0 / n_types
        if n_types >= 2:
            accs = []
            for held in range(n_agents):
                tr, te = slot != held, slot == held
                if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
                    continue
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(sc.transform(X[tr]), y[tr])
                accs.append(float((clf.predict(sc.transform(X[te])) == y[te]).mean()))
            probe_acc = float(np.mean(accs)) if accs else None
    except Exception as e:
        logger.console_logger.info(f"probe skipped: {e}")

    out = {
        "tag": args_cli.tag,
        "map": args_cli.map,
        "checkpoint": model_path,
        "step": step,
        "wandb_run": args_cli.wandb_run,
        "episodes": args_cli.episodes,
        "win_rate": n_wins / args_cli.episodes,
        "n_agents": n_agents,
        "n_actions": int(args.n_actions),
        "n_unit_types": n_types,
        "n_rows_total": int(len(H)),
        "n_rows_alive": int(alive_mask.sum()),
        "effrank_n_centroid": effrank_n,
        "effrank_centroid_raw": effrank_centroid,
        "effrank_global_allrows": effrank_global,
        "d_act_maskaware": dact_mean,
        "probe_acc": probe_acc,
        "probe_chance": probe_chance,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    logger.console_logger.info("METRICS:\n" + json.dumps(out, indent=2))
    print("WROTE", args_cli.out)


if __name__ == "__main__":
    main()
