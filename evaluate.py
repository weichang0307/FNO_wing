"""Score a trained run on a held-out split (default: the 200 full-split test cases).

Usage: python evaluate.py R3 [--split test --ckpt best]
Writes results/<run>_<split>.csv (one row per case) and results/<run>_<split>_summary.json, prints the
target metrics, and draws figs/<run>_<split>_scatter.png and figs/<run>_<split>_cp.png.
"""
import argparse
import json
import os
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from fno_data import Normalizer, load_dimensionless, load_surface, split_indices
from forces import force_coefficients
from model import SurfaceFNO
from train import masked_rel_l2, surface_rel_l2

ROOT = os.path.dirname(os.path.abspath(__file__))
SERIES = ["#2a78d6", "#eb6834"]  # reference palette slots 1-2
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e5a", "#e4e3dd"


def run_dir(run):
    """A frozen model version (models/<name>) or a training run (runs/<name>)."""
    frozen = os.path.join(ROOT, "models", run)
    return frozen if os.path.isdir(frozen) else os.path.join(ROOT, "runs", run)


def load_model(run, which, dev):
    ckpt = torch.load(os.path.join(run_dir(run), f"{which}.pt"), map_location=dev, weights_only=False)
    cfg = ckpt["cfg"]
    model = SurfaceFNO(cfg, cfg["box"]).to(dev)
    model.load_state_dict({k: v for k, v in ckpt["model"].items() if k != "_metadata"})
    model.eval()
    with open(os.path.join(run_dir(run), "norm_stats.json")) as f:
        norm = Normalizer(json.load(f)).to(dev)
    return model, norm, cfg, ckpt["epoch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    args = ap.parse_args()
    dev = "cuda"
    model, norm, cfg, epoch = load_model(args.run, args.ckpt, dev)
    _, va, te = split_indices(cfg["data"], cfg["task"], seed=0)
    idx = np.sort(te if args.split == "test" else va)
    x, y, u_inf = load_dimensionless(cfg["data"], idx)
    surf = load_surface(cfg["data"], idx)
    with h5py.File(cfg["data"], "r") as h5:
        names = [n.decode() for n in h5["names"][idx]]
        re, aoa = h5["re"][idx], h5["aoa"][idx]
        cl_ref, cd_ref = h5["cl"][idx], h5["cd"][idx]

    t = lambda k: torch.from_numpy(surf[k]).to(dev)
    X, Y = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
    f_err, s_err, s_pred = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), 16):
            b = slice(i, i + 16)
            f_n, s_n = model(norm.x(X[b]), t("pts")[b], t("nrm")[b], t("order")[b], t("pos")[b], t("n_pts")[b])
            f_err.append(masked_rel_l2(norm.y_inverse(f_n), Y[b], X[b, 1]).cpu().numpy())
            s = norm.s_inverse(s_n)
            s_err.append(surface_rel_l2(s, t("s")[b], t("smask")[b]).cpu().numpy())
            s_pred.append(s.cpu().numpy())
        # timing: one case at a time, as a designer would call it
        one = (norm.x(X[:1]), t("pts")[:1], t("nrm")[:1], t("order")[:1], t("pos")[:1], t("n_pts")[:1])
        for _ in range(5):
            model(*one)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            model(*one)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 50 * 1000
    f_err, s_err, s_pred = np.concatenate(f_err), np.concatenate(s_err), np.concatenate(s_pred)

    rows = []
    for k in range(len(idx)):
        n = int(surf["n_pts"][k])
        sk = s_pred[k, :n].astype(np.float64)
        cl, cd, parts = force_coefficients(sk[:, 0], sk[:, 1:], surf["lines"][k], surf["cell_normals"][k],
                                           surf["cell_length"][k], 1.0, float(surf["aoa"][k]))
        rows.append(dict(name=names[k], re=re[k], aoa=aoa[k], cl_true=cl_ref[k], cl_pred=cl, cd_true=cd_ref[k], cd_pred=cd,
                         err_u=f_err[k, 0], err_v=f_err[k, 1], err_p=f_err[k, 2], err_nut=f_err[k, 3],
                         err_surf_p=s_err[k, 0], err_surf_tau=s_err[k, 1]))
    df = pd.DataFrame(rows)
    df["cl_rel"] = (df.cl_pred - df.cl_true).abs() / df.cl_true.abs()
    df["cd_rel"] = (df.cd_pred - df.cd_true).abs() / df.cd_true.abs()
    df["cl_abs"] = (df.cl_pred - df.cl_true).abs()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    tag = f"{args.run}_{args.split}"
    df.to_csv(os.path.join(ROOT, "results", f"{tag}.csv"), index=False)

    q = lambda s, p: float(np.quantile(s, p))
    summary = dict(
        run=args.run, checkpoint=f"{args.ckpt}.pt (epoch {epoch})", split=args.split, cases=len(df),
        field_error_mean={c: float(df[f"err_{c}"].mean()) for c in ["u", "v", "p", "nut"]},
        surface_error_mean=dict(p=float(df.err_surf_p.mean()), tau=float(df.err_surf_tau.mean())),
        cl_rel=dict(median=float(df.cl_rel.median()), p90=q(df.cl_rel, 0.9), max=float(df.cl_rel.max())),
        cd_rel=dict(median=float(df.cd_rel.median()), p90=q(df.cd_rel, 0.9), max=float(df.cd_rel.max())),
        cl_abs_mean=float(df.cl_abs.mean()), cl_spearman=float(spearmanr(df.cl_pred, df.cl_true).statistic),
        cd_spearman=float(spearmanr(df.cd_pred, df.cd_true).statistic), inference_ms=ms)
    with open(os.path.join(ROOT, "results", f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ok = lambda b: "PASS" if b else "MISS"
    fe = summary["field_error_mean"]
    print(f"{args.run} {summary['checkpoint']} on {len(df)} {args.split} cases\n")
    print(f"  field error u / v / p     {100*fe['u']:.2f} / {100*fe['v']:.2f} / {100*fe['p']:.2f}%   target < 5%     "
          f"{ok(max(fe['u'], fe['v'], fe['p']) < 0.05)}")
    print(f"  field error nu_t          {100*fe['nut']:.2f}%")
    print(f"  surface p / wall shear    {100*summary['surface_error_mean']['p']:.2f} / {100*summary['surface_error_mean']['tau']:.2f}%")
    for c, tgt in [("cl", 0.02), ("cd", 0.05)]:
        r = summary[f"{c}_rel"]
        print(f"  {c.upper()} error median         {100*r['median']:.2f}%   (90th pct {100*r['p90']:.2f}%, max {100*r['max']:.1f}%)   "
              f"target < {100*tgt:.0f}%   {ok(r['median'] < tgt)}")
    print(f"  CL mean absolute error    {summary['cl_abs_mean']:.4f}")
    print(f"  CL ranking (Spearman)     {summary['cl_spearman']:.5f}   target >= 0.98  {ok(summary['cl_spearman'] >= 0.98)}")
    print(f"  CD ranking (Spearman)     {summary['cd_spearman']:.5f}")
    print(f"  inference, one case       {ms:.1f} ms   target < 20 ms  {ok(ms < 20)}")
    worst = df.nlargest(5, "cd_rel")[["name", "re", "aoa", "cd_true", "cd_pred", "cd_rel"]]
    print("\n  largest CD errors:\n" + worst.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    plot_scatter(df, tag)
    plot_cp(df, surf, s_pred, tag)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=8)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2)


def plot_scatter(df, tag):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=SURFACE)
    for ax, c, name in [(axes[0], "cl", "CL"), (axes[1], "cd", "CD")]:
        style(ax, f"{name}: predicted vs CFD", f"CFD {name}", f"predicted {name}")
        lo, hi = min(df[f"{c}_true"].min(), df[f"{c}_pred"].min()), max(df[f"{c}_true"].max(), df[f"{c}_pred"].max())
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.scatter(df[f"{c}_true"], df[f"{c}_pred"], s=18, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
        ax.annotate("perfect prediction", (hi, hi), xytext=(-60, 6), textcoords="offset points", ha="right",
                    fontsize=8, color=INK_2)
    ax = axes[2]
    style(ax, "Error distribution over test cases", "relative error (%)", "cases")
    bins = np.logspace(np.log10(0.01), np.log10(max(100 * df.cd_rel.max(), 10)), 35)
    ax.hist(100 * df.cl_rel, bins=bins, color=SERIES[0], alpha=0.85, label="CL", edgecolor=SURFACE)
    ax.hist(100 * df.cd_rel, bins=bins, color=SERIES[1], alpha=0.65, label="CD", edgecolor=SURFACE)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    for v, lab, dy in [(2, "CL target 2%", -12), (5, "CD target 5%", -24)]:
        ax.axvline(v, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(lab, (v, ax.get_ylim()[1]), xytext=(3, dy), textcoords="offset points", fontsize=8, color=INK_2)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figs", f"{tag}_scatter.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)


def plot_cp(df, surf, s_pred, tag):
    order = df.cd_rel.sort_values().index
    picks = [("best", order[0]), ("median", order[len(order) // 2]), ("worst", order[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=SURFACE)
    for ax, (label, k) in zip(axes, picks):
        r = df.loc[k]
        n = int(surf["n_pts"][k])
        loop = surf["order"][k, :n]  # walk the surface in order so the curve is continuous
        x = surf["pts"][k, loop, 0]
        cp_t, cp_p = 2 * surf["s"][k, loop, 0], 2 * s_pred[k, loop, 0]  # Cp = p / (U^2 / 2)
        style(ax, f"{label} CD case: Re {r.re / 1e6:.1f} M, AoA {r.aoa:.1f}°", "x / c", "Cp")
        ax.plot(x, cp_t, color=SERIES[0], linewidth=2, label="CFD")
        ax.plot(x, cp_p, color=SERIES[1], linewidth=1.5, linestyle=(0, (5, 2)), label="model")
        ax.invert_yaxis()
        ax.legend(frameon=False, labelcolor=INK, loc="upper right")
        ax.annotate(f"CL {r.cl_true:.3f} → {r.cl_pred:.3f} ({100 * r.cl_rel:.2f}%)\n"
                    f"CD {r.cd_true:.4f} → {r.cd_pred:.4f} ({100 * r.cd_rel:.1f}%)",
                    (0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figs", f"{tag}_cp.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
