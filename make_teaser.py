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
2. Render:
     python make_teaser.py --left obs.npz  --left-title "(a) Unit type observed" --left-probe 0.76 \
                           --right mask.npz --right-title "(b) Unit type masked"  --right-probe 0.41 \
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
    H, UT, ALIVE = d["H"], d["UT"], d["ALIVE"]
    m = (ALIVE == 1) & (UT >= 0)
    H, UT = H[m], UT[m]
    uids = sorted(set(UT.tolist()))
    role = np.array([uids.index(u) for u in UT])
    # role-discriminating projection (the probe's view; role structure is not in the
    # top PCA directions, so a supervised projection is needed to see it in 2D).
    Z = LDA(n_components=2).fit(H, role).transform(H)
    lo, hi = np.percentile(Z, [1, 99], axis=0)
    pad = 0.15 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    for r in range(len(uids)):
        sel = role == r
        ax.scatter(Z[sel, 0], Z[sel, 1], s=8, alpha=0.30, color=COLORS[r % 3],
                   edgecolors="none", rasterized=True,
                   label=TYPE_NAMES[r] if r < len(TYPE_NAMES) else f"Type {r}")
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
    ap.add_argument("--left", required=True); ap.add_argument("--right", required=True)
    ap.add_argument("--left-title", default="(a)"); ap.add_argument("--right-title", default="(b)")
    ap.add_argument("--left-probe", type=float, default=None)
    ap.add_argument("--right-probe", type=float, default=None)
    ap.add_argument("--out", default="teaser.png")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 200})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    panel(axes[0], a.left, a.left_title, a.left_probe)
    panel(axes[1], a.right, a.right_title, a.right_probe)
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.9, markerscale=2.2,
                   handletextpad=0.2, borderpad=0.3, fontsize=10)
    fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
