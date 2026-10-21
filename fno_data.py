"""Loading the gridded AirfRANS HDF5 into model space, and the normalization statistics.

Model space:
  x = [sdf, mask, U_inf_x, U_inf_y]              standardized (mask left as 0/1)
  y = [u/U_inf, v/U_inf, p/U_inf^2, nu_t/nu]     dimensionless, then standardized
  s = [p/U_inf^2, tau_x/U_inf^2, tau_y/U_inf^2]  at the true surface points, then standardized
Statistics come from training cases only; targets use fluid cells only.
"""
import json

import h5py
import numpy as np
import torch

NU = 1.56e-5
X_NAMES = ["sdf", "mask", "u_inf_x", "u_inf_y"]
Y_NAMES = ["u/U", "v/U", "p/U2", "nut/nu"]
S_NAMES = ["p/U2", "tau_x/U2", "tau_y/U2"]


def grid_box(h5_path):
    with h5py.File(h5_path, "r") as h5:
        return tuple(float(h5.attrs[k]) for k in ["x0", "x1", "y0", "y1"])


def loop_order(lines, n):
    """Walk the closed surface loop: order[k] = k-th point along the loop, pos[i] = place of point i."""
    adj = [[] for _ in range(n)]
    for a, b in lines:
        adj[a].append(b)
        adj[b].append(a)
    order, prev, cur = [0], -1, 0
    while True:
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        prev, cur = cur, nxt
        if cur == 0:
            break
        order.append(cur)
    assert len(order) == n, "surface is not a single closed loop"
    order = np.array(order)
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    return order, pos


def load_surface(h5_path, idx):
    """Surface points, normals and dimensionless targets, padded to the largest case.

    Returns dict of numpy arrays: pts, nrm (N, S, 2), s (N, S, 3), smask (N, S), the loop
    ordering order, pos (N, S) and n_pts (N,) for convolution along the surface, per-case
    lists lines, cell_normals, cell_length for force integration; plus aoa (degrees).
    """
    idx = np.sort(idx)
    with h5py.File(h5_path, "r") as h5:
        g = h5["surface"]
        off, coff = g["offsets"][:], g["cell_offsets"][:]
        u_inf, aoa = h5["u_inf"][:], h5["aoa"][:]
        cols = {k: g[k][:] for k in ["points", "normals", "p", "wss", "lines", "cell_normals", "cell_length"]}
    S = int(max(off[i + 1] - off[i] for i in idx))
    N = len(idx)
    out = dict(pts=np.zeros((N, S, 2), np.float32), nrm=np.zeros((N, S, 2), np.float32),
               s=np.zeros((N, S, 3), np.float32), smask=np.zeros((N, S), np.float32),
               order=np.zeros((N, S), np.int64), pos=np.zeros((N, S), np.int64), n_pts=np.zeros(N, np.int64),
               lines=[], cell_normals=[], cell_length=[], aoa=aoa[idx].astype(np.float32))
    for j, i in enumerate(idx):
        a, b = off[i], off[i + 1]
        n, U2 = b - a, float(u_inf[i]) ** 2
        out["pts"][j, :n] = cols["points"][a:b]
        out["nrm"][j, :n] = cols["normals"][a:b]
        out["s"][j, :n, 0] = cols["p"][a:b] / U2
        out["s"][j, :n, 1:] = cols["wss"][a:b] / U2
        out["smask"][j, :n] = 1
        ca, cb = coff[i], coff[i + 1]
        out["order"][j, :n], out["pos"][j, :n] = loop_order(cols["lines"][ca:cb], n)
        out["n_pts"][j] = n
        out["lines"].append(cols["lines"][ca:cb])
        out["cell_normals"].append(cols["cell_normals"][ca:cb])
        out["cell_length"].append(cols["cell_length"][ca:cb])
    return out


def split_indices(h5_path, task="full", val_frac=0.1, seed=0):
    """Train / validation / test case indices for an AirfRANS task (full, scarce, reynolds, aoa)."""
    with h5py.File(h5_path, "r") as h5:
        train = h5[f"splits/{task}_train"][:]
        test = h5[f"splits/{'full' if task == 'scarce' else task}_test"][:]
    rng = np.random.default_rng(seed)
    train = rng.permutation(train)
    n_val = int(round(val_frac * len(train)))
    return np.sort(train[n_val:]), np.sort(train[:n_val]), test


def load_dimensionless(h5_path, idx):
    """Inputs (raw) and dimensionless targets for the given cases, as float32 numpy arrays."""
    idx = np.sort(idx)
    with h5py.File(h5_path, "r") as h5:
        x = h5["inputs"][idx]
        y = h5["targets"][idx]
        u_inf = h5["u_inf"][idx].astype(np.float32)
    U = u_inf[:, None, None]
    y[:, 0] /= U
    y[:, 1] /= U
    y[:, 2] /= U ** 2
    y[:, 3] /= NU
    return x, y, u_inf


def compute_stats(x, y, surf=None):
    mask = x[:, 1].astype(bool)
    stats = {"x_mean": [], "x_std": [], "y_mean": [], "y_std": []}
    for c in range(4):
        if X_NAMES[c] == "mask":
            m, s = 0.0, 1.0
        else:
            m, s = float(x[:, c].mean()), float(x[:, c].std())
        stats["x_mean"].append(m)
        stats["x_std"].append(s)
    for c in range(4):
        v = y[:, c][mask]
        stats["y_mean"].append(float(v.mean()))
        stats["y_std"].append(float(v.std()))
    stats["x_names"], stats["y_names"] = X_NAMES, Y_NAMES
    if surf is not None:
        valid = surf["smask"].astype(bool)
        stats["s_mean"] = [float(surf["s"][..., c][valid].mean()) for c in range(3)]
        stats["s_std"] = [float(surf["s"][..., c][valid].std()) for c in range(3)]
        stats["s_names"] = S_NAMES
    return stats


def save_stats(stats, path):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path):
    with open(path) as f:
        return json.load(f)


class Normalizer(torch.nn.Module):
    """Standardizes inputs and targets; travels with the checkpoint."""

    def __init__(self, stats):
        super().__init__()
        t = lambda k: torch.tensor(stats[k], dtype=torch.float32).view(1, 4, 1, 1)
        for k in ["x_mean", "x_std", "y_mean", "y_std"]:
            self.register_buffer(k, t(k))
        for k in ["s_mean", "s_std"]:
            self.register_buffer(k, torch.tensor(stats.get(k, [0.0, 0.0, 0.0] if k == "s_mean" else [1.0] * 3),
                                                 dtype=torch.float32).view(1, 1, 3))

    def x(self, x):
        return (x - self.x_mean) / self.x_std

    def y(self, y):
        return (y - self.y_mean) / self.y_std

    def y_inverse(self, y_n):
        return y_n * self.y_std + self.y_mean

    def s_inverse(self, s_n):
        return s_n * self.s_std + self.s_mean


if __name__ == "__main__":
    # write the full-split statistics (checklist item); training recomputes them per task
    import os
    root = os.path.dirname(os.path.abspath(__file__))
    h5 = os.path.join(root, "data", "airfrans_128x256.h5")
    tr, va, te = split_indices(h5, "full")
    x, y, _ = load_dimensionless(h5, tr)
    stats = compute_stats(x, y, load_surface(h5, tr))
    out = os.path.join(root, "data", "norm_stats_full_128x256.json")
    save_stats(stats, out)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  -> {out}")
    for k in ["x", "y", "s"]:
        for n, m, s in zip(stats[f"{k}_names"], stats[f"{k}_mean"], stats[f"{k}_std"]):
            print(f"  {k} {n:8s} mean {m:12.5g}  std {s:12.5g}")
