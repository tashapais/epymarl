#!/usr/bin/env python3
"""Generate the teaser figure: per-agent SMACv2 embeddings projected onto the 2D
role-discriminating subspace, coloured by unit type, for two conditions side by side
(e.g. unit type observed vs masked). Shows that role-separable geometry tracks whether
unit type is observed, not the reward attribution.

Pipeline
--------
1. Dump embeddings from each checkpoint with metrics_repr.py:
     python metrics_repr.py --checkpoint <observed_ckpt> --load_step 4000000 \
         --map protoss_5_vs_5 --episodes 16 --out obs.json --dump-embeddings obs.npz
     python metrics_repr.py --checkpoint <masked_ckpt>   --load_step 4000000 \
         --map protoss_5_vs_5 --episodes 16 --obs-mask --out mask.json --dump-embeddings mask.npz
   (each .npz holds H=embeddings, UT=unit-type ids, SLOT=agent slot, ALIVE flags)
2. Render one panel per condition (left to right), each "<npz>:<title>:<probe>":
     python make_teaser.py \
         --panel "indiv.npz:(a) Individual reward:0.64" \
         --panel "shared.npz:(b) Shared (team) reward:0.65" \
         --panel "mask.npz:(c) Individual, type masked:0.41" \
         --out teaser.png

The probe accuracies are read off the corresponding metrics_repr JSONs (probe_acc).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

TYPE_NAMES = ["Stalker", "Zealot", "Colossus"]
COLORS = ["#2E6FB7", "#E1812C", "#3A923A"]


def panel(ax, npz, title, probe):
    d = np.load(npz)
    H, UT, SLOT, ALIVE = d["H"], d["UT"], d["SLOT"], d["ALIVE"]
    m = (ALIVE == 1) & (UT >= 0)
    H, UT, SLOT = H[m], UT[m], SLOT[m]
    uids = sorted(set(UT.tolist()))
    role = np.array([uids.index(u) for u in UT])
    # Leave-agent-slots-out role-discriminating projection (the probe's view; role
    # structure is not in the top PCA directions, so a supervised projection is needed
    # to see it in 2D). Fit the discriminant on all-but-two slots and plot only the
    # held-out slots, so the layout reflects role structure that *generalises* across
    # agents rather than in-sample overfit (which would separate every condition).
    held = set(np.argsort(np.bincount(SLOT))[-2:].tolist())
    te = np.array([s in held for s in SLOT]); tr = ~te
    Z = LDA(n_components=2).fit(H[tr], role[tr]).transform(H)[te]
    role = role[te]
    # balance classes by subsampling to the smallest so all roles are visible (alive
    # timesteps are unit-type-imbalanced: fragile units contribute fewer points).
    rng = np.random.default_rng(0)
    k = min(np.bincount(role))
    keep = np.concatenate([rng.choice(np.where(role == r)[0], k, replace=False)
                           for r in range(len(uids))])
    Z, role = Z[keep], role[keep]
    lo, hi = np.percentile(Z, [2, 98], axis=0)
    pad = 0.25 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    for r in range(len(uids)):
        sel = role == r
        ax.scatter(Z[sel, 0], Z[sel, 1], s=8, alpha=0.30, color=COLORS[r % 3],
                   edgecolors="none", rasterized=True,
                   label=TYPE_NAMES[r] if r < len(TYPE_NAMES) else f"Type {r}")
    for r in range(len(uids)):
        sel = role == r
        if sel.sum():
            ax.scatter(*Z[sel].mean(0), s=240, color=COLORS[r % 3],
                       edgecolors="black", linewidths=2.0, zorder=5)
    t = title if probe is None else f"{title}\nunit-type probe = {probe:.2f}  (chance 0.33)"
    ax.set_title(t)
    ax.set_xlabel("role-discriminant 1"); ax.set_ylabel("role-discriminant 2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelleft=False, labelbottom=False)


def main():
    ap = argparse.ArgumentParser()
    # one or more panels, each "<npz>:<title>:<probe>" (probe optional), left to right
    ap.add_argument("--panel", action="append", required=True,
                    help='repeatable "npz:title:probe", e.g. "obs.npz:(a) Individual reward:0.64"')
    ap.add_argument("--out", default="teaser.png")
    a = ap.parse_args()
    specs = []
    for p in a.panel:
        parts = p.split(":")
        npz, title = parts[0], parts[1] if len(parts) > 1 else "(panel)"
        probe = float(parts[2]) if len(parts) > 2 and parts[2] else None
        specs.append((npz, title, probe))
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 200})
    n = len(specs)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.4), squeeze=False)
    axes = axes[0]
    for ax, (npz, title, probe) in zip(axes, specs):
        panel(ax, npz, title, probe)
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.9, markerscale=2.2,
                   handletextpad=0.2, borderpad=0.3, fontsize=10)
    fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
