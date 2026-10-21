"""Loss and error curves for a training run (works while the run is still going).

Usage: python plot_run.py R3 [--out figs/curves_R3.png]
Four panels, one measure each: training loss, field error per channel, surface error, and the
CL / CD errors against their targets. Log scales, since errors fall by orders of magnitude.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
# reference categorical palette, fixed order: blue, orange, aqua, yellow
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e5a", "#e4e3dd"


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_xlabel("epoch", color=INK_2)
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, which="both")
    plain = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 1.5, 2, 3, 5, 7)))
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax._end_labels = []


def line(ax, x, y, color, label):
    ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round")
    ax._end_labels.append((label, x.iloc[-1], y.iloc[-1]))


def place_end_labels(ax, gap_pt=11):
    """Direct labels at line ends, in ink (the line carries the colour), nudged apart vertically."""
    fig = ax.figure
    to_pt = 72 / fig.dpi
    items = sorted(((ax.transData.transform((x, y))[1] * to_pt, label, x, y) for label, x, y in ax._end_labels))
    placed = []
    for ypt, label, x, y in items:
        ypt = max(ypt, placed[-1] + gap_pt) if placed else ypt
        placed.append(ypt)
        y0 = ax.transData.transform((x, y))[1] * to_pt
        ax.annotate(label, (x, y), xytext=(6, ypt - y0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)


def target(ax, value, label):
    ax.axhline(value, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(label, (0, value), xycoords=("axes fraction", "data"), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=INK_2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out")
    args = ap.parse_args()
    m = pd.read_csv(os.path.join(ROOT, "runs", args.run, "metrics.csv"))
    pct = lambda c: 100 * m[c]
    e = m["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=SURFACE)
    (a1, a2), (a3, a4) = axes

    style(a1, "Training loss", "loss (relative L2, fields + surface)")
    a1.plot(e, m["train_loss"], color=SERIES[0], linewidth=2)

    style(a2, "Validation field error", "relative L2 error (%)")
    for c, (col, name) in enumerate([("val_u/U", "u"), ("val_v/U", "v"), ("val_p/U2", "p"), ("val_nut/nu", "ν_t")]):
        line(a2, e, pct(col), SERIES[c], name)
    target(a2, 5, "target 5% (u, v, p)")

    style(a3, "Validation surface error", "relative L2 error (%)")
    line(a3, e, pct("val_surf_p"), SERIES[0], "pressure")
    line(a3, e, pct("val_surf_tau"), SERIES[1], "wall shear")

    style(a4, "Validation force error (median over cases)", "relative error (%)")
    line(a4, e, pct("val_cl_median_rel"), SERIES[0], "CL")
    line(a4, e, pct("val_cd_median_rel"), SERIES[1], "CD")
    target(a4, 2, "CL target 2%")
    target(a4, 5, "CD target 5%")

    for ax in axes.flat:
        ax.set_xlim(0, max(e.max() * 1.08, 10))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ax in axes.flat:
        place_end_labels(ax)
    last = m.iloc[-1]
    fig.suptitle(f"Run {args.run}: epoch {int(last.epoch)}, CL {100 * last.val_cl_median_rel:.2f}%, "
                 f"CD {100 * last.val_cd_median_rel:.1f}%, CL ranking {last.val_cl_spearman:.4f}",
                 x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.out or os.path.join(ROOT, "figs", f"curves_{args.run}.png")
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
