"""Best, median and worst test cases by CD error: CFD against the model, field and surface.

Usage: python checks/case_comparison.py v1
  figs/<run>_cases.png         rows best / median / worst; columns CFD Cp field, model Cp field,
                               surface Cp, drag build-up along the chord
  figs/<run>_cases_errors.png  |model - CFD| on the grid for u, v, Cp and nu_t, one colour scale per channel
  figs/<run>_cases.csv         the numbers behind both
"""
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluate import load_model  # noqa: E402
from fno_data import load_dimensionless, load_surface, split_indices  # noqa: E402

SERIES = ["#2a78d6", "#eb6834"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e5a", "#e4e3dd"


def drag_buildup(sv, lines, cell_n, length, aoa_deg, pts):
    """Drag accumulated from the leading edge (x = 0) back to each x, over both surfaces.

    sv: (n, 3) dimensionless surface values p / U^2, tau_x / U^2, tau_y / U^2 in point order. Uses the same
    segment rule as forces.py, so the curve ends at the case's CD.
    """
    a, b = lines[:, 0], lines[:, 1]
    d = np.array([np.cos(np.radians(aoa_deg)), np.sin(np.radians(aoa_deg))])
    p_cell = 0.5 * (sv[a, 0] + sv[b, 0])
    tau_cell = 0.5 * (sv[a, 1:] + sv[b, 1:])
    contrib = (p_cell * (cell_n @ d) + tau_cell @ d) * length / 0.5  # q = U^2 / 2 with U = 1
    xm = 0.5 * (pts[a, 0] + pts[b, 0])
    order = np.argsort(xm)
    return xm[order], np.cumsum(contrib[order])


def style_line(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_xlabel("x / c", color=INK_2)


def describe(name):
    p = [float(v) for v in name.split("_")[4:]]
    if len(p) == 3:
        return f"NACA 4-digit: camber {p[0]:.1f}%, at {10 * p[1]:.0f}% chord, thickness {p[2]:.1f}%"
    return f"NACA 5-digit: design CL {0.15 * p[0]:.2f}, camber at {5 * p[1]:.0f}% chord{', reflex' if p[2] else ''}, thickness {p[3]:.1f}%"


def main(run):
    dev = "cuda"
    model, norm, cfg, _ = load_model(run, "best", dev)
    res = pd.read_csv(os.path.join(ROOT, "results", f"{'R3' if run == 'v1' else run}_test.csv"))
    order = res.cd_rel.sort_values().index
    picks = [("Best", order[0]), ("Median", order[len(order) // 2]), ("Worst", order[-1])]
    _, _, te = split_indices(cfg["data"], cfg["task"], seed=0)
    te = np.sort(te)
    idx = np.array([te[k] for _, k in picks])
    # the loaders return cases sorted by index; put them back in best, median, worst order
    pos = np.searchsorted(np.sort(idx), idx)
    x, y, _ = load_dimensionless(cfg["data"], idx)
    x, y = x[pos], y[pos]
    surf = load_surface(cfg["data"], idx)
    surf = {k: (v[pos] if isinstance(v, np.ndarray) else [v[i] for i in pos]) for k, v in surf.items()}
    with h5py.File(cfg["data"], "r") as h5:
        box = [h5.attrs[k] for k in ["x0", "x1", "y0", "y1"]]
    t = lambda k: torch.from_numpy(surf[k]).to(dev)
    with torch.no_grad():
        f_n, s_n = model(norm.x(torch.from_numpy(x).to(dev)), t("pts"), t("nrm"), t("order"), t("pos"), t("n_pts"))
        fields, spred = norm.y_inverse(f_n).cpu().numpy(), norm.s_inverse(s_n).cpu().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(22, 11.5), facecolor=SURFACE, layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1, 0.85, 0.85]))
    rows = []
    for r, (label, k) in enumerate(picks):
        c = res.loc[k]
        fluid = x[r, 1].astype(bool)
        cp_cfd, cp_mod = np.where(fluid, 2 * y[r, 2], np.nan), np.where(fluid, 2 * fields[r, 2], np.nan)
        lo = min(np.nanmin(cp_cfd), np.nanmin(cp_mod), -0.2)
        norm_c = mcolors.TwoSlopeNorm(vmin=lo, vcenter=0, vmax=1)
        for col, (cp, who) in enumerate([(cp_cfd, "CFD"), (cp_mod, "v1 model")]):
            ax = axes[r, col]
            im = ax.imshow(cp, origin="lower", extent=box, cmap="RdBu_r", norm=norm_c, interpolation="bilinear")
            n = int(surf["n_pts"][r])
            loop = surf["pts"][r, surf["order"][r, :n]]
            ax.fill(loop[:, 0], loop[:, 1], color="#6b6a65")
            ax.set_xlim(-0.3, 1.4)
            ax.set_ylim(-0.4, 0.4)
            ax.set_aspect("equal")
            ax.set_title(f"{label} case, {who}: pressure coefficient", loc="left", fontsize=11, color=INK)
            ax.tick_params(colors=INK_2, labelsize=8)
            for s in ax.spines.values():
                s.set_color(GRID)
        fig.colorbar(im, ax=axes[r, :2], shrink=0.75, label="Cp field", pad=0.02)
        ax = axes[r, 2]
        n = int(surf["n_pts"][r])
        loop = surf["order"][r, :n]
        xs = surf["pts"][r, loop, 0]
        ax.plot(xs, 2 * surf["s"][r, loop, 0], color=SERIES[0], linewidth=2, label="CFD")
        ax.plot(xs, 2 * spred[r, loop, 0], color=SERIES[1], linewidth=1.5, linestyle=(0, (5, 2)), label="v1 model")
        ax.invert_yaxis()
        style_line(ax)
        ax.set_ylabel("surface Cp", color=INK_2)
        ax.set_title("Surface pressure coefficient", loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, labelcolor=INK, loc="upper right", fontsize=9)

        # drag build-up: where along the chord the drag (and the model's drag error) comes from
        ax = axes[r, 3]
        args = (surf["lines"][r], surf["cell_normals"][r].astype(np.float64), surf["cell_length"][r].astype(np.float64),
                float(surf["aoa"][r]), surf["pts"][r, :n].astype(np.float64))
        xc, cd_c = drag_buildup(surf["s"][r, :n].astype(np.float64), *args)
        xm, cd_m = drag_buildup(spred[r, :n].astype(np.float64), *args)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.plot(xc, cd_c, color=SERIES[0], linewidth=2, label=f"CFD, CD {cd_c[-1]:.4f}")
        ax.plot(xm, cd_m, color=SERIES[1], linewidth=1.5, linestyle=(0, (5, 2)), label=f"v1 model, CD {cd_m[-1]:.4f}")
        style_line(ax)
        ax.set_ylabel("CD accumulated from the nose", color=INK_2)
        ax.set_title(f"Drag build-up: error {100 * c.cd_rel:.1f}%", loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, labelcolor=INK, loc="lower right", fontsize=9)
        rows.append(dict(case=label, shape=describe(c["name"]), re=c.re, aoa=c.aoa, cl_cfd=c.cl_true, cl_model=c.cl_pred,
                         cl_err=c.cl_rel, cd_cfd=c.cd_true, cd_model=c.cd_pred, cd_err=c.cd_rel,
                         p_field_err=c.err_p, surf_p_err=c.err_surf_p))
    out = os.path.join(ROOT, "figs", f"{run}_cases.png")
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    plot_errors(run, picks, x, y, fields, surf, box)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "figs", f"{run}_cases.csv"), index=False)
    print(out)
    print(pd.DataFrame(rows).to_string(index=False))


def plot_errors(run, picks, x, y, fields, surf, box):
    """|model - CFD| on the grid, per channel; each column shares one colour scale across the three cases."""
    chans = [(0, "u / U\u221e", 1), (1, "v / U\u221e", 1), (2, "Cp", 2), (3, "\u03bd_t / \u03bd", 1)]
    errs = {ch: [np.where(x[r, 1] > 0, np.abs(fields[r, ch] - y[r, ch]) * scale, np.nan) for r in range(3)]
            for ch, _, scale in chans}
    fig, axes = plt.subplots(3, 4, figsize=(22, 9.5), facecolor=SURFACE, layout="constrained")
    for col, (ch, name, _) in enumerate(chans):
        vmax = np.nanpercentile(np.concatenate([e.ravel() for e in errs[ch]]), 99.5)
        for r, (label, _) in enumerate(picks):
            ax = axes[r, col]
            im = ax.imshow(errs[ch][r], origin="lower", extent=box, cmap="magma_r", vmin=0, vmax=vmax,
                           interpolation="nearest")
            n = int(surf["n_pts"][r])
            loop = surf["pts"][r, surf["order"][r, :n]]
            ax.fill(loop[:, 0], loop[:, 1], color="#6b6a65")
            ax.set_xlim(-0.3, 1.4)
            ax.set_ylim(-0.4, 0.4)
            ax.set_aspect("equal")
            mean = np.nanmean(errs[ch][r])
            ax.set_title(f"{label} case: error in {name} (mean {mean:.3g})", loc="left", fontsize=11, color=INK)
            ax.tick_params(colors=INK_2, labelsize=8)
            for s in ax.spines.values():
                s.set_color(GRID)
        cb = fig.colorbar(im, ax=axes[:, col], shrink=0.6, label=f"|model \u2212 CFD|, {name}", location="bottom", pad=0.02)
        cb.locator = matplotlib.ticker.MaxNLocator(4)
        cb.update_ticks()
    out = os.path.join(ROOT, "figs", f"{run}_cases_errors.png")
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
