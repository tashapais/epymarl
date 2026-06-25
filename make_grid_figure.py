#!/usr/bin/env python3
"""Grouped bar chart of the SMACv2 protoss_5_vs_5 2x2 grid (4M steps, 3 seeds/cell).

Two diagnostics side by side, each grouped by whether unit type is observed vs
masked (rows of the grid), with individual vs shared reward bars (columns):

  - Unit-type probe accuracy (geometric / role decodability), chance = 1/3.
    Masking the observation roughly halves the above-chance signal, while reward
    attribution leaves it unchanged: the observation, not the reward, pins the
    role geometry.
  - D_act, the mean pairwise action-distribution KL (behavioral). Leans
    individual > shared: the reward signal surfaces behaviorally.

Numbers are the reported mean +/- std over 3 seeds (32 greedy eval episodes each);
the underlying runs are the four W&B groups / HuggingFace 4M checkpoints listed in
runs.md (shared vs individual x observed vs obs_mask_unit_type).

Usage:
    python make_grid_figure.py --out results/grid_metrics.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (mean, std) over 3 seeds; 4M-step checkpoints, 32 greedy eval episodes each.
PROBE = {
    "observed": {"individual": (0.73, 0.05), "shared": (0.75, 0.05)},
    "masked":   {"individual": (0.49, 0.04), "shared": (0.49, 0.14)},
}
DACT = {
    "observed": {"individual": (1.23, 0.06), "shared": (1.07, 0.20)},
    "masked":   {"individual": (1.13, 0.08), "shared": (0.98, 0.18)},
}
CHANCE = 1.0 / 3.0

C_IND = "#003E72"   # camblue (poster colour) -- individual reward
C_SHR = "#E1812C"   # orange -- shared reward
GROUPS = ["observed", "masked"]
XLABELS = ["Unit type\nobserved", "Unit type\nmasked"]


def grouped(ax, data, title, ylim, ref=None, ref_label=None):
    x = np.arange(len(GROUPS))
    w = 0.36
    ind = [data[g]["individual"] for g in GROUPS]
    shr = [data[g]["shared"] for g in GROUPS]
    ax.bar(x - w / 2, [m for m, _ in ind], w, yerr=[s for _, s in ind],
           capsize=6, color=C_IND, label="Individual reward",
           error_kw=dict(lw=2, ecolor="0.25"))
    ax.bar(x + w / 2, [m for m, _ in shr], w, yerr=[s for _, s in shr],
           capsize=6, color=C_SHR, label="Shared (team) reward",
           error_kw=dict(lw=2, ecolor="0.25"))
    for xi, (m, _) in zip(x - w / 2, ind):
        ax.text(xi, m, f"{m:.2f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=C_IND)
    for xi, (m, _) in zip(x + w / 2, shr):
        ax.text(xi, m, f"{m:.2f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=C_SHR)
    if ref is not None:
        ax.axhline(ref, ls="--", lw=1.8, color="0.45")
        ax.text(1.0 + w, ref, ref_label, ha="right", va="bottom",
                fontsize=10.5, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=12.5)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.margins(x=0.12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/grid_metrics.png")
    args = ap.parse_args()

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    grouped(ax1, PROBE, "Role decodability (geometric)", ylim=(0, 0.92),
            ref=CHANCE, ref_label="chance = 1/3")
    ax1.set_ylabel("Unit-type probe accuracy", fontsize=12.5)

    grouped(ax2, DACT, "Action divergence (behavioral)", ylim=(0, 1.55))
    ax2.set_ylabel(r"$D_{\mathrm{act}}$  (mean pairwise KL)", fontsize=12.5)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=12.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
