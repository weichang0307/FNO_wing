"""Phase 2 step 2: look at AirfRANS before preprocessing.

Usage: python airfrans_viz.py [--root data/AirfRANS/Dataset] [--out figs]
Produces:
  dataset_spread.png  Re/AoA coverage per split, CL/CD histograms, CL vs AoA, drag polar
  case_<name>.png     u-magnitude, pressure, nut fields + surface Cp for 5 sample cases
  cpflow_<name>.png   Cp map with velocity arrows (whole airfoil + leading-edge close-up)
  coefficients.csv    CL, CD per case (cached; delete to recompute)
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import airfrans as af

NU = 1.56e-5  # kinematic viscosity used by AirfRANS, chord = 1 m


def parse_name(name):
    # airFoil2D_SST_<Uinf>_<AoA>_<NACA params...>
    parts = name.split("_")
    u_inf, aoa = float(parts[2]), float(parts[3])
    params = [float(p) for p in parts[4:]]
    family = "NACA 4" if len(params) == 3 else "NACA 5"
    return dict(name=name, u_inf=u_inf, aoa=aoa, re=u_inf / NU, family=family)


def coefficients(args):
    root, name = args
    sim = af.Simulation(root=root, name=name)
    (cd, _, _), (cl, _, _) = sim.force_coefficient()
    return name, cl, cd


def build_table(root, out):
    csv = os.path.join(out, "coefficients.csv")
    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    split_of = {}
    for split in ["full_train", "full_test", "scarce_train",
                  "reynolds_train", "reynolds_test", "aoa_train", "aoa_test"]:
        for n in manifest.get(split, []):
            split_of.setdefault(n, {})[split] = True
    names = sorted(split_of)
    df = pd.DataFrame([parse_name(n) for n in names])
    for split in ["full_test", "reynolds_test", "aoa_test", "scarce_train"]:
        df[split] = [split_of[n].get(split, False) for n in names]

    if os.path.exists(csv):
        coef = pd.read_csv(csv)
    else:
        with ProcessPoolExecutor(max_workers=16) as ex:
            rows = list(ex.map(coefficients, [(root, n) for n in names], chunksize=8))
        coef = pd.DataFrame(rows, columns=["name", "cl", "cd"])
        coef.to_csv(csv, index=False)
    return df.merge(coef, on="name")


def plot_spread(df, out):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    a = ax[0, 0]
    tr = df[~df.full_test]
    te = df[df.full_test]
    a.scatter(tr.re / 1e6, tr.aoa, s=8, alpha=0.6, label=f"full train ({len(tr)})")
    a.scatter(te.re / 1e6, te.aoa, s=8, alpha=0.8, label=f"full test ({len(te)})")
    a.set(xlabel="Re (millions)", ylabel="AoA (deg)", title="Operating conditions, full split")
    a.legend()

    a = ax[0, 1]
    a.scatter(df.re / 1e6, df.aoa, s=8, c=np.where(df.reynolds_test, "C3", "C7"), alpha=0.6)
    a.set(xlabel="Re (millions)", ylabel="AoA (deg)",
          title="Reynolds-extrapolation split (red = test)")

    a = ax[0, 2]
    a.scatter(df.re / 1e6, df.aoa, s=8, c=np.where(df.aoa_test, "C3", "C7"), alpha=0.6)
    a.set(xlabel="Re (millions)", ylabel="AoA (deg)", title="AoA-extrapolation split (red = test)")

    a = ax[1, 0]
    a.hist(df.cl, bins=40, color="C0")
    a.set(xlabel="CL", ylabel="cases", title=f"CL distribution (n={len(df)})")

    a = ax[1, 1]
    a.hist(df.cd, bins=40, color="C1")
    a.set(xlabel="CD", ylabel="cases", title="CD distribution")

    a = ax[1, 2]
    sc = a.scatter(df.cd, df.cl, c=df.aoa, s=8, cmap="viridis")
    fig.colorbar(sc, ax=a, label="AoA (deg)")
    a.set(xlabel="CD", ylabel="CL", title="Drag polar, all cases")

    fig.tight_layout()
    fig.savefig(os.path.join(out, "dataset_spread.png"), dpi=130)
    plt.close(fig)


def pick_cases(df):
    # low / mid / high AoA, low and high Re, plus one NACA 5-digit
    picks = []
    for aoa, re in [(-4, 3e6), (2, 2.5e6), (8, 4e6), (14, 5.5e6)]:
        d = (df.aoa - aoa) ** 2 / 25 + ((df.re - re) / 2e6) ** 2
        picks.append(df.loc[d.idxmin(), "name"])
    five = df[df.family == "NACA 5"]
    picks.append(five.loc[(five.aoa - 6).abs().idxmin(), "name"])
    return list(dict.fromkeys(picks))


def load_case(root, name):
    sim = af.Simulation(root=root, name=name)
    pos = sim.position
    tri = mtri.Triangulation(pos[:, 0], pos[:, 1])
    # drop triangles spanning the airfoil (Delaunay fills the solid)
    centroid = pos[tri.triangles].mean(axis=1)
    sdf_c = sim.sdf[:, 0][tri.triangles].mean(axis=1)
    edge = np.max(np.linalg.norm(pos[tri.triangles] - centroid[:, None, :], axis=2), axis=1)
    tri.set_mask((sdf_c < 1e-4) | (edge > 0.2))

    # surface points are unordered: walk the upper surface leading to trailing edge, then the
    # lower surface back (normals point into the airfoil, so upper-surface normals have y < 0)
    ap = sim.airfoil_position
    up = sim.airfoil_normals[:, 1] < 0
    upper, lower = ap[up], ap[~up]
    outline = np.vstack([upper[np.argsort(upper[:, 0])], lower[np.argsort(-lower[:, 0])]])
    return sim, tri, outline


def case_tag(row):
    return f"{row['family'].replace(' ', '')}_aoa{row['aoa']:+.1f}_re{row['re']/1e6:.1f}M"


def plot_case(root, row, out):
    sim, tri, outline = load_case(root, row["name"])
    pos = sim.position

    q = 0.5 * row["u_inf"] ** 2
    umag = np.linalg.norm(sim.velocity, axis=1)
    fields = [(umag / row["u_inf"], "|u| / U_inf", "viridis"),
              (sim.pressure[:, 0] / q, "Cp", "RdBu_r"),
              (sim.nu_t[:, 0] / NU, "nu_t / nu", "magma")]

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    for i, (val, label, cmap) in enumerate(fields):
        a = fig.add_subplot(gs[0, i])
        lo, hi = np.percentile(val, [1, 99])
        if label == "Cp":
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
        cs = a.tricontourf(tri, np.clip(val, lo, hi), levels=64, cmap=cmap)
        a.fill(outline[:, 0], outline[:, 1], color="0.3")
        a.set(xlim=(-0.3, 1.6), ylim=(-0.5, 0.5), aspect="equal", title=label)
        fig.colorbar(cs, ax=a, shrink=0.7)

    a = fig.add_subplot(gs[1, :2])
    sx = sim.airfoil_position[:, 0]
    cp = np.asarray(sim.airfoil.point_data["p"]) / q  # OpenFOAM kinematic pressure, p / rho
    upper = sim.airfoil_normals[:, 1] < 0  # AirfRANS surface normals point into the airfoil
    a.scatter(sx[upper], cp[upper], s=4, label="upper surface")
    a.scatter(sx[~upper], cp[~upper], s=4, label="lower surface")
    a.invert_yaxis()
    a.set(xlabel="x / c", ylabel="Cp", title="Surface pressure coefficient")
    a.legend()

    a = fig.add_subplot(gs[1, 2])
    a.axis("off")
    a.text(0, 0.9, f"{row['name']}", fontsize=9, family="monospace", wrap=True)
    a.text(0, 0.6, f"{row['family']}\nRe = {row['re']/1e6:.2f} M\nAoA = {row['aoa']:.2f} deg\n"
           f"CL = {row['cl']:.4f}\nCD = {row['cd']:.5f}\nmesh points = {len(pos):,}", fontsize=13, va="top")

    fig.tight_layout()
    fig.savefig(os.path.join(out, f"case_{case_tag(row)}.png"), dpi=110)
    plt.close(fig)


def plot_cp_flow(root, row, out):
    """Cp map with velocity arrows: arrow direction = flow direction, length = local speed."""
    sim, tri, outline = load_case(root, row["name"])
    u_inf = row["u_inf"]
    cp = sim.pressure[:, 0] / (0.5 * u_inf ** 2)
    ux = mtri.LinearTriInterpolator(tri, sim.velocity[:, 0] / u_inf)
    uy = mtri.LinearTriInterpolator(tri, sim.velocity[:, 1] / u_inf)

    # full view and a leading-edge close-up; arrow spacing scales with the window
    views = [("Whole airfoil", (-0.3, 1.6), (-0.45, 0.45), 0.05),
             ("Leading edge close-up", (-0.06, 0.24), (-0.07, 0.08), 0.0085)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 13), gridspec_kw={"height_ratios": [0.9, 1]})
    for a, (title, xl, yl, dx) in zip(axes, views):
        inside = ((tri.x >= xl[0] - dx) & (tri.x <= xl[1] + dx) & (tri.y >= yl[0] - dx) & (tri.y <= yl[1] + dx))
        # Cp tops out at +1 (stagnation) but suction can reach -6: centre the colours on 0
        # with separate scales either side, so the compression side stays visible
        lo = min(np.percentile(cp[inside], 0.2), -0.2)
        norm = mcolors.TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=1.0)
        levels = np.concatenate([np.linspace(lo, 0, 40, endpoint=False), np.linspace(0, 1, 25)])
        cs = a.tricontourf(tri, np.clip(cp, lo, 1.0), levels=levels, cmap="RdBu_r", norm=norm)
        a.fill(outline[:, 0], outline[:, 1], color="0.3", zorder=5)

        gx, gy = np.meshgrid(np.arange(xl[0] + dx / 2, xl[1], dx), np.arange(yl[0] + dx / 2, yl[1], dx))
        vx, vy = ux(gx, gy), uy(gx, gy)  # masked inside the airfoil
        # length in plot units = (|u| / U_inf) * 0.8 * spacing, so a freestream arrow fills 80% of a cell
        qv = a.quiver(gx, gy, vx, vy, angles="xy", scale_units="xy", scale=1 / (0.8 * dx),
                      width=0.0018, headwidth=3.5, color="k", zorder=4)
        a.quiverkey(qv, 0.9, 1.03, 1.0, "U\u221e (freestream)", labelpos="W", coordinates="axes")
        a.set(xlim=xl, ylim=yl, aspect="equal", title=title, xlabel="x / c", ylabel="y / c")
        fig.colorbar(cs, ax=a, shrink=0.8, label="Cp", ticks=np.r_[np.linspace(lo, 0, 5), 0.5, 1.0].round(2))

    fig.suptitle(f"{row['family']}, Re = {row['re']/1e6:.2f} M, AoA = {row['aoa']:.1f} deg, "
                 f"CL = {row['cl']:.3f}, CD = {row['cd']:.4f}\n"
                 "colour: pressure coefficient (blue = suction, red = compression); "
                 "arrows: flow direction, length proportional to local speed", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"cpflow_{case_tag(row)}.png"), dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "AirfRANS", "Dataset"))
    p.add_argument("--out", default="figs")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = build_table(args.root, args.out)
    print(df[["re", "aoa", "cl", "cd"]].describe().round(4).to_string())
    print(df.family.value_counts().to_string())
    plot_spread(df, args.out)
    for name in pick_cases(df):
        row = df[df.name == name].iloc[0]
        plot_case(args.root, row, args.out)
        plot_cp_flow(args.root, row, args.out)
        print("plotted", name)


if __name__ == "__main__":
    main()
