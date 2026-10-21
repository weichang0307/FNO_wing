"""Run a trained surface-FNO on any NACA 4- or 5-digit airfoil, without CFD.

Builds the same inputs the training data had: the airfoil from airfrans' NACA generator (chord 1,
leading edge at the origin), signed distance and fluid mask on the 128 x 256 grid, freestream
velocity channels, and the surface loop with inward normals for the surface head.

Runs on an NVIDIA GPU (CUDA), an Apple GPU (MPS) or the CPU; needs only PyTorch, neuraloperator,
numpy, scipy and matplotlib, plus models/<run>/best.pt and data/surface_template_sides.npz.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from matplotlib.path import Path
from scipy.spatial import cKDTree

from naca import naca_generator  # airfrans' generator, copied so airfrans (and VTK) is not needed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fno_data import Normalizer  # noqa: E402
from forces import force_coefficients  # noqa: E402
from model import SurfaceFNO  # noqa: E402

NU = 1.56e-5  # air at 298 K, as in AirfRANS: Re = U_inf * chord / NU with chord = 1
# training ranges (from the 1,000 AirfRANS cases)
RANGES = dict(naca4=dict(camber=(0, 7), position=(0, 7), thickness=(5, 20)),
              naca5=dict(design_cl=(0, 4), position=(3, 8), thickness=(5, 20)),
              aoa=(-5, 15), re=(2e6, 6e6))


TEMPLATE = os.path.join(ROOT, "data", "surface_template_sides.npz")


def model_dir(run):
    """A frozen model version (models/<name>) or a training run (runs/<name>)."""
    frozen = os.path.join(ROOT, "models", run)
    return frozen if os.path.isdir(frozen) else os.path.join(ROOT, "runs", run)


def load_model(run, which, dev):
    path = os.path.join(model_dir(run), f"{which}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"Model weights not found: {path}\n"
                         f"Copy {which}.pt there from the training machine (it is not in git).")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = SurfaceFNO(cfg, cfg["box"])
    model.load_state_dict({k: v for k, v in ckpt["model"].items() if k != "_metadata"})
    with open(os.path.join(model_dir(run), "norm_stats.json")) as f:
        norm = Normalizer(json.load(f))
    return model.to(dev).eval(), norm.to(dev), cfg, ckpt["epoch"]


def sync(dev):
    """Wait for queued GPU work, so timings measure the computation itself."""
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def pick_device(pref="auto"):
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # checked against the CPU in Predictor before it is used
    return "cpu"


def loop_fractions(pts):
    """Normalized arc length of each point along a closed loop, starting at the trailing edge."""
    seg = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
    s = np.r_[0, np.cumsum(seg)[:-1]]
    return s / seg.sum()


def sides(P):
    """Split a clockwise loop that starts at the trailing edge into lower and upper surfaces at the
    leading edge (the point farthest from the trailing edge)."""
    le = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
    return P[:le + 1], np.vstack([P[le:], P[:1]])


def side_fractions(S):
    seg = np.linalg.norm(np.diff(S, axis=0), axis=1)
    return np.r_[0, np.cumsum(seg)] / seg.sum()


def surface_template(n=1000, cases=300):
    """Average point spacing of the AirfRANS CFD surfaces, separately for the lower surface (trailing
    edge to leading edge) and the upper surface (leading edge back to the trailing edge), as arc-length
    fractions of each side. The mesh packs ~40% of its points into the first 5% of chord, and the
    surface head was trained on that spacing."""
    if os.path.exists(TEMPLATE):
        t = np.load(TEMPLATE)
        return t["lower"], t["upper"]
    h5 = os.path.join(ROOT, "data", "airfrans_128x256.h5")
    if not os.path.exists(h5):
        raise SystemExit(f"{TEMPLATE} is missing and cannot be rebuilt without {h5}; copy it from the "
                         "training machine (it is small and should be in git).")
    from fno_data import load_surface, split_indices
    tr, _, _ = split_indices(h5, "full")
    surf = load_surface(h5, np.sort(tr)[:cases])
    low, up, share = [], [], []
    for k in range(len(surf["n_pts"])):
        m = int(surf["n_pts"][k])
        P = surf["pts"][k, surf["order"][k, :m]].astype(np.float64)
        area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        if area > 0:
            P = P[::-1]  # make it clockwise
        P = np.roll(P, -int(np.argmax(P[:, 0])), 0)  # start at the trailing edge
        lo, hi = sides(P)
        share.append(len(lo) / m)
        low.append(np.interp(np.linspace(0, len(lo) - 1, 500), np.arange(len(lo)), side_fractions(lo)))
        up.append(np.interp(np.linspace(0, len(hi) - 1, 500), np.arange(len(hi)), side_fractions(hi)))
    nl = int(round(np.mean(share) * n))
    lower = np.interp(np.linspace(0, 499, nl + 1), np.arange(500), np.mean(low, 0))
    upper = np.interp(np.linspace(0, 499, n - nl + 1), np.arange(500), np.mean(up, 0))
    np.savez(TEMPLATE, lower=lower, upper=upper)
    return lower, upper


def airfoil(params, n=1000):
    """Closed surface loop like the AirfRANS CFD surfaces: clockwise from the trailing edge, with the
    CFD mesh's point spacing (dense at the leading edge). Returns points, inward point normals,
    segments (i, i + 1), segment normals as airfrans integrates them, and segment lengths."""
    dense = naca_generator(np.asarray(params, dtype=float), nb_samples=8000, verbose=False)[:-1][::-1]
    dense = np.roll(dense, -int(np.argmax(dense[:, 0])), 0)  # clockwise, starting at the trailing edge
    t_low, t_up = surface_template(n)
    side_pts = []
    for S, t in zip(sides(dense), (t_low, t_up)):
        f = side_fractions(S)
        side_pts.append(np.stack([np.interp(t, f, S[:, 0]), np.interp(t, f, S[:, 1])], 1)[:-1])
    pts = np.vstack(side_pts)  # trailing edge, lower surface, leading edge, upper surface
    n = len(pts)
    seg = np.roll(pts, -1, 0) - pts                 # segment i joins point i to i + 1
    length = np.linalg.norm(seg, axis=1)
    seg_n = np.stack([seg[:, 1], -seg[:, 0]], 1) / length[:, None]  # inward for a clockwise loop
    pt_n = seg_n + np.roll(seg_n, 1, 0)             # average of the two segments meeting at each point
    pt_n /= np.linalg.norm(pt_n, axis=1, keepdims=True)
    lines = np.stack([np.arange(n), (np.arange(n) + 1) % n], 1)
    cell_n = 0.5 * (pt_n[lines[:, 0]] + pt_n[lines[:, 1]])  # as airfrans integrates
    return pts, pt_n, lines, cell_n, length


class Predictor:
    def __init__(self, run="v1", ckpt="best", dev="auto"):
        self.run, self.ckpt = run, ckpt
        self.dev = pick_device(dev)
        self.model, self.norm, self.cfg, self.epoch = load_model(run, ckpt, self.dev)
        H, W = 128, 256
        x0, x1, y0, y1 = self.cfg["box"]  # the training grid: cell centres over the box, in chord lengths
        self.xs = x0 + (np.arange(W) + 0.5) * (x1 - x0) / W
        self.ys = y0 + (np.arange(H) + 0.5) * (y1 - y0) / H
        gx, gy = np.meshgrid(self.xs, self.ys)
        self.grid_pts = np.c_[gx.ravel(), gy.ravel()]
        self.H, self.W = H, W
        self._geom_cache = {}
        if self.dev == "mps" and dev == "auto":
            self._check_mps()

    def _check_mps(self):
        """PyTorch's Apple GPU support for FFTs and complex numbers is incomplete; use it only if it
        gives the same answer as the CPU on a test airfoil."""
        try:
            gpu, _ = self.predict([2, 4, 12], [4.0], 3e6)
            cpu_model, cpu_norm, _, _ = load_model(self.run, self.ckpt, "cpu")
            self.model, self.norm, self.dev = cpu_model, cpu_norm, "cpu"
            ref, _ = self.predict([2, 4, 12], [4.0], 3e6)
            ok = abs(gpu[0]["cl"] - ref[0]["cl"]) < 1e-3 and abs(gpu[0]["cd"] - ref[0]["cd"]) < 1e-4
            if ok:
                self.model, self.norm, _, _ = load_model(self.run, self.ckpt, "mps")
                self.dev = "mps"
            else:
                print("Apple GPU results differ from the CPU; using the CPU.", flush=True)
        except Exception as e:  # an operation the Apple GPU does not support
            print(f"Apple GPU not usable for this model ({type(e).__name__}); using the CPU.", flush=True)
            self.model, self.norm, _, _ = load_model(self.run, self.ckpt, "cpu")
            self.dev = "cpu"

    def geometry(self, params):
        key = tuple(round(float(v), 4) for v in params)
        if key not in self._geom_cache:
            pts, pt_n, lines, cell_n, length = airfoil(params)
            inside = Path(pts).contains_points(self.grid_pts).reshape(self.H, self.W)
            dist, _ = cKDTree(pts).query(self.grid_pts)
            sdf = np.where(inside, -1.0, 1.0) * dist.reshape(self.H, self.W)
            if len(self._geom_cache) > 64:
                self._geom_cache.clear()
            self._geom_cache[key] = (pts, pt_n, lines, cell_n, length, sdf, ~inside)
        return self._geom_cache[key]

    @torch.no_grad()
    def predict(self, params, aoa_list, re):
        """One airfoil at one Reynolds number, any number of angles in one batch."""
        pts, pt_n, lines, cell_n, length, sdf, fluid = self.geometry(params)
        u_inf = re * NU
        B, n = len(aoa_list), len(pts)
        a = np.radians(np.asarray(aoa_list, dtype=np.float64))
        x = np.zeros((B, 4, self.H, self.W), np.float32)
        x[:, 0], x[:, 1] = sdf, fluid
        x[:, 2] = (u_inf * np.cos(a))[:, None, None]
        x[:, 3] = (u_inf * np.sin(a))[:, None, None]
        d = self.dev
        T = lambda v, dt=torch.float32: torch.as_tensor(np.broadcast_to(v, (B,) + v.shape).copy(), dtype=dt, device=d)
        order = np.arange(n)
        sync(d)
        t0 = time.time()
        f_n, s_n = self.model(self.norm.x(torch.from_numpy(x).to(d)), T(pts), T(pt_n),
                              T(order, torch.int64), T(order, torch.int64),
                              torch.full((B,), n, dtype=torch.int64, device=d))
        fields, surf = self.norm.y_inverse(f_n), self.norm.s_inverse(s_n)
        sync(d)
        ms = (time.time() - t0) * 1000
        fields, surf = fields.cpu().numpy(), surf.cpu().double().numpy()  # MPS has no float64
        out = []
        for k, aoa in enumerate(aoa_list):
            cl, cd, parts = force_coefficients(surf[k, :, 0], surf[k, :, 1:], lines, cell_n, length, 1.0, float(aoa))
            out.append(dict(cl=cl, cd=cd, parts=parts, fields=fields[k], surface=surf[k]))
        return out, dict(points=pts, normals=pt_n, fluid=fluid, ms=ms, u_inf=u_inf)


def in_range(family, params, aoa, re):
    r = RANGES[family]
    vals = dict(zip(r.keys(), [params[0], params[1], params[-1]]))
    off = [k for k, (lo, hi) in r.items() if not lo - 1e-9 <= vals[k] <= hi + 1e-9]
    if not RANGES["aoa"][0] <= aoa <= RANGES["aoa"][1]:
        off.append("aoa")
    if not RANGES["re"][0] <= re <= RANGES["re"][1]:
        off.append("re")
    return off
