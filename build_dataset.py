"""Phase 2: grid every AirfRANS case into one HDF5 file and run the Gate 1 force check.

For each case:
  1. sample the CFD mesh onto a regular H x W grid (pyvista probe on the true mesh cells)
  2. build input channels  sdf, mask, U_inf_x, U_inf_y   and targets  u, v, p, nu_t (physical units)
  3. Gate 1: interpolate the gridded targets back onto the CFD mesh, recompute CL and CD with
     airfrans exactly as a model prediction will be scored, and compare with the original fields

Usage: python build_dataset.py [--H 128 --W 256 --workers 14]
Writes data/airfrans_<H>x<W>.h5 and results/gate1_<H>x<W>.csv
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import airfrans as af
import h5py
import numpy as np
import pandas as pd
import pyvista as pv
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from forces import force_coefficients

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(ROOT, "data", "AirfRANS", "Dataset")
BOX = dict(x0=-0.5, x1=1.5, y0=-0.5, y1=0.5)  # chord lengths; leading edge at 0, trailing edge at 1
SPLITS = ["full_train", "full_test", "scarce_train", "reynolds_train", "reynolds_test", "aoa_train", "aoa_test"]


def grid_axes(H, W):
    """Cell-centre coordinates of the training grid."""
    dx = (BOX["x1"] - BOX["x0"]) / W
    dy = (BOX["y1"] - BOX["y0"]) / H
    xs = BOX["x0"] + (np.arange(W) + 0.5) * dx
    ys = BOX["y0"] + (np.arange(H) + 0.5) * dy
    return xs, ys, dx, dy


def process_case(args):
    name, H, W = args
    sim = af.Simulation(root=DATASET, name=name)
    xs, ys, dx, dy = grid_axes(H, W)

    # 1. sample the mesh onto the grid; ImageData points run x fastest, then y
    image = pv.ImageData(dimensions=(W, H, 1), spacing=(dx, dy, 1.0), origin=(xs[0], ys[0], 0.5))
    probe = image.sample(sim.internal)
    mask = np.asarray(probe.point_data["vtkValidPointMask"]).reshape(H, W).astype(bool)
    u = np.asarray(probe.point_data["U"])[:, 0].reshape(H, W)
    v = np.asarray(probe.point_data["U"])[:, 1].reshape(H, W)
    p = np.asarray(probe.point_data["p"]).reshape(H, W)
    nut = np.asarray(probe.point_data["nut"]).reshape(H, W)

    # 2. inputs; signed distance to the true surface, negative inside the airfoil
    gx, gy = np.meshgrid(xs, ys)
    dist, _ = cKDTree(sim.airfoil_position).query(np.c_[gx.ravel(), gy.ravel()])
    sdf = np.where(mask, 1.0, -1.0) * dist.reshape(H, W)
    u_inf = float(sim.inlet_velocity)
    aoa = float(sim.angle_of_attack)
    inputs = np.stack([sdf, mask.astype(float), np.full((H, W), u_inf * np.cos(aoa)),
                       np.full((H, W), u_inf * np.sin(aoa))])
    targets = np.stack([np.where(mask, u, 0), np.where(mask, v, 0), np.where(mask, p, 0), np.where(mask, nut, 0)])

    # 3. Gate 1: grid -> mesh round trip, scored with airfrans' own force integration
    (cd_t, cdp_t, cdv_t), (cl_t, clp_t, clv_t) = sim.force_coefficient(reference=True)
    # inside the airfoil: velocity 0 (no-slip), pressure from the nearest fluid cell, so
    # bilinear interpolation next to the wall is not dragged towards 0
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    filled = dict(u=np.where(mask, u, 0.0), v=np.where(mask, v, 0.0), p=p[iy, ix])
    pos = sim.position
    in_box = ((pos[:, 0] > xs[0]) & (pos[:, 0] < xs[-1]) & (pos[:, 1] > ys[0]) & (pos[:, 1] < ys[-1]))
    back = {k: RegularGridInterpolator((ys, xs), f)(pos[in_box][:, ::-1]) for k, f in filled.items()}
    sim.velocity[in_box, 0], sim.velocity[in_box, 1] = back["u"], back["v"]
    sim.velocity[sim.surface] = 0.0  # the wall boundary condition is known exactly
    sim.pressure[in_box, 0] = back["p"]
    (cd_g, cdp_g, cdv_g), (cl_g, clp_g, clv_g) = sim.force_coefficient()

    # 4. surface path: pressure and wall shear at the true surface points, the targets of the
    #    surface head; forces integrated from them must reproduce airfrans
    surf_p = np.asarray(sim.airfoil.point_data["p"], dtype=np.float64)
    surf_wss = sim.wallshearstress(over_airfoil=True, reference=True)
    lines = np.asarray(sim.airfoil.lines).reshape(-1, 3)[:, 1:]
    # airfrans converts point data to cells before integrating, which replaces the stored segment
    # normals with the mean of the two point normals; do the same so forces match it exactly
    cell_n = 0.5 * (sim.airfoil_normals[lines[:, 0]] + sim.airfoil_normals[lines[:, 1]])
    cell_len = np.asarray(sim.airfoil.cell_data["Length"])
    cl_s, cd_s, parts_s = force_coefficients(surf_p, surf_wss, lines, cell_n, cell_len, u_inf, np.degrees(aoa))

    gate = dict(name=name, cl=cl_t, cd=cd_t, clp=clp_t, cdp=cdp_t, cdv=cdv_t,
                cl_grid=cl_g, cd_grid=cd_g, clp_grid=clp_g, cdp_grid=cdp_g, cdv_grid=cdv_g,
                cl_surface=cl_s, cd_surface=cd_s)
    surface = dict(points=sim.airfoil_position, normals=sim.airfoil_normals, p=surf_p, wss=surf_wss,
                   lines=lines, cell_normals=cell_n, cell_length=cell_len)
    return (inputs.astype(np.float32), targets.astype(np.float32), u_inf, np.degrees(aoa),
            u_inf / float(sim.NU), surface, gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases, for testing")
    args = ap.parse_args()

    with open(os.path.join(DATASET, "manifest.json")) as f:
        manifest = json.load(f)
    names = sorted(set(manifest["full_train"]) | set(manifest["full_test"]))
    if args.limit:
        names = names[:args.limit]
    N, H, W = len(names), args.H, args.W
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "data", f"airfrans_{H}x{W}{'_test' if args.limit else ''}.h5")

    t0 = time.time()
    gates, surf = [], []
    with h5py.File(out, "w") as h5, ProcessPoolExecutor(max_workers=args.workers) as ex:
        xs, ys, dx, dy = grid_axes(H, W)
        h5.attrs.update(dict(H=H, W=W, dx=dx, dy=dy, nu=1.56e-5, **BOX,
                             inputs="sdf, mask, U_inf_x, U_inf_y",
                             targets="u, v, p (kinematic, m2/s2), nu_t; physical units, 0 inside airfoil"))
        d_in = h5.create_dataset("inputs", (N, 4, H, W), "f4", chunks=(1, 4, H, W))
        d_tg = h5.create_dataset("targets", (N, 4, H, W), "f4", chunks=(1, 4, H, W))
        meta = {k: np.zeros(N) for k in ["u_inf", "aoa", "re"]}
        for i, (inp, tgt, u_inf, aoa, re, sf, gate) in enumerate(
                ex.map(process_case, [(n, H, W) for n in names], chunksize=4)):
            d_in[i], d_tg[i] = inp, tgt
            meta["u_inf"][i], meta["aoa"][i], meta["re"][i] = u_inf, aoa, re
            surf.append(sf)
            gates.append(gate)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{N} cases, {time.time() - t0:.0f} s", flush=True)

        for k, v in meta.items():
            h5[k] = v
        h5["names"] = np.array(names, dtype="S")
        gdf = pd.DataFrame(gates)
        h5["cl"], h5["cd"] = gdf.cl.values, gdf.cd.values
        # per-point arrays concatenated over cases, sliced by offsets; normals point into the airfoil
        h5["surface/offsets"] = np.r_[0, np.cumsum([len(s["points"]) for s in surf])]
        h5["surface/cell_offsets"] = np.r_[0, np.cumsum([len(s["lines"]) for s in surf])]
        for k in ["points", "normals", "p", "wss", "cell_normals", "cell_length"]:
            h5[f"surface/{k}"] = np.concatenate([s[k] for s in surf]).astype(np.float32)
        h5["surface/lines"] = np.concatenate([s["lines"] for s in surf]).astype(np.int32)  # local indices
        h5["surface"].attrs["p"] = "kinematic pressure at surface points, m2/s2"
        h5["surface"].attrs["wss"] = "kinematic wall shear stress at surface points, m2/s2"
        index = {n: i for i, n in enumerate(names)}
        for s in SPLITS:
            h5[f"splits/{s}"] = np.array(sorted(index[n] for n in manifest[s] if n in index), dtype=int)

    gate_csv = os.path.join(ROOT, "results", f"gate1_{H}x{W}{'_test' if args.limit else ''}.csv")
    gdf.to_csv(gate_csv, index=False)
    print(f"wrote {out} ({os.path.getsize(out) / 2**30:.2f} GB) and {gate_csv} in {time.time() - t0:.0f} s")
    report_gate1(gdf)


def report_gate1(g):
    rel = lambda a, b: (g[a] - g[b]).abs() / g[b].abs()
    rows = {
        "CL": rel("cl_grid", "cl"), "CD": rel("cd_grid", "cd"),
        "CL pressure part": rel("clp_grid", "clp"), "CD pressure part": rel("cdp_grid", "cdp"),
        "CD friction part": rel("cdv_grid", "cdv"),
    }
    print("\nGate 1: relative error of forces recomputed from the gridded ground truth")
    for k, r in rows.items():
        print(f"  {k:18s} median {100 * r.median():6.2f}%   90th pct {100 * r.quantile(0.9):6.2f}%   max {100 * r.max():7.2f}%")
    cl_ok, cd_ok = rows["CL"].median() < 0.01, rows["CD"].median() < 0.02
    print(f"Gate 1 {'PASSED' if cl_ok and cd_ok else 'FAILED'}: CL median under 1% {cl_ok}, CD median under 2% {cd_ok}")
    # friction needs the wall velocity gradient (first cell about 2 micrometres off the wall), which no
    # Cartesian grid resolves; the pressure parts show what the grid itself can carry
    print(f"  pressure-only: CL pressure part median under 1% {rows['CL pressure part'].median() < 0.01}")
    if "cl_surface" in g:
        cl_s, cd_s = rel("cl_surface", "cl"), rel("cd_surface", "cd")
        print(f"Surface path (forces from surface p and wall shear): CL max error {100 * cl_s.max():.2e}%, "
              f"CD max error {100 * cd_s.max():.2e}% -> {'PASSED' if max(cl_s.max(), cd_s.max()) < 1e-4 else 'FAILED'}")


if __name__ == "__main__":
    main()
