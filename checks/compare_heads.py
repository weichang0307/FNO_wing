"""Why do surface errors and force errors disagree? Compare two trained runs on the validation cases.

For each run: surface error, then CL and CD errors split into pressure and friction parts, the
signed bias of each part (systematic error survives integration, scatter mostly cancels), and how
much of the force error comes from the leading-edge region.

Usage: python checks/compare_heads.py T30 T30c
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fno_data import Normalizer, load_dimensionless, load_surface, split_indices  # noqa: E402
from forces import force_coefficients  # noqa: E402
from model import SurfaceFNO  # noqa: E402

LE = 0.05  # leading-edge region: first 5% of chord


class PointwiseFNO(SurfaceFNO):
    """The first surface head (run T30): a 4-layer pointwise MLP, before the loop convolution."""

    def __init__(self, cfg, box):
        super().__init__(dict(cfg, head_conv_layers=0), box)
        n_in, h = cfg["width"] + 6, cfg.get("head_width", 128)
        del self.head_in, self.head_conv, self.head_out
        self.head = torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.GELU(), torch.nn.Linear(h, h),
                                        torch.nn.GELU(), torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, 3))

    def forward(self, x, s_pts, s_nrm, *loop):
        h = self.hidden(x)
        x0, x1, y0, y1 = self.box
        g = torch.stack([2 * (s_pts[..., 0] - x0) / (x1 - x0) - 1, 2 * (s_pts[..., 1] - y0) / (y1 - y0) - 1], -1)[:, :, None]
        feat = torch.nn.functional.grid_sample(h, g, align_corners=False).squeeze(-1).transpose(1, 2)
        free = x[:, 2:4, 0, 0][:, None].expand(-1, s_pts.shape[1], -1)
        return self.fno.projection(h), self.head(torch.cat([feat, s_pts, s_nrm, free], -1))


def run_predictions(run, data_h5, va_idx, surf, x_va, dev):
    ckpt = torch.load(os.path.join(ROOT, "runs", run, "last.pt"), map_location=dev, weights_only=False)
    cfg = ckpt["cfg"]
    cls = PointwiseFNO if any(k.startswith("head.") for k in ckpt["model"]) else SurfaceFNO
    model = cls(cfg, cfg["box"]).to(dev)
    state = {k: v for k, v in ckpt["model"].items() if k != "_metadata"}  # neuraloperator bookkeeping
    model.load_state_dict(state)
    model.eval()
    with open(os.path.join(ROOT, "runs", run, "norm_stats.json")) as f:
        norm = Normalizer(json.load(f)).to(dev)
    t = lambda k: torch.from_numpy(surf[k]).to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(va_idx), 8):
            b = slice(i, i + 8)
            _, s_n = model(norm.x(torch.from_numpy(x_va[b]).to(dev)), t("pts")[b], t("nrm")[b],
                           t("order")[b], t("pos")[b], t("n_pts")[b])
            out.append(norm.s_inverse(s_n).cpu().numpy())
    return np.concatenate(out), ckpt["epoch"]


def forces(s, surf, k, n, keep=None):
    p, wss = s[k, :n, 0].astype(np.float64), s[k, :n, 1:].astype(np.float64)
    if keep is not None:  # zero the error outside a region by using the true values there
        p = np.where(keep, p, surf["s"][k, :n, 0])
        wss = np.where(keep[:, None], wss, surf["s"][k, :n, 1:])
    return force_coefficients(p, wss, surf["lines"][k], surf["cell_normals"][k], surf["cell_length"][k],
                              1.0, float(surf["aoa"][k]))


def main(runs):
    dev = "cuda"
    data = os.path.join(ROOT, "data", "airfrans_128x256.h5")
    _, va_idx, _ = split_indices(data, "full", seed=0)
    va_idx = np.sort(va_idx)
    x_va, _, _ = load_dimensionless(data, va_idx)
    surf = load_surface(data, va_idx)
    N = len(va_idx)

    rows = {}
    for run in runs:
        pred, epoch = run_predictions(run, data, va_idx, surf, x_va, dev)
        r = {k: [] for k in ["p_err", "tau_err", "cl", "cd", "clp", "clv", "cdp", "cdv", "le_share_cl", "le_share_cd",
                             "p_bias", "tau_bias"]}
        for k in range(N):
            n = int(surf["n_pts"][k])
            tru, prd = surf["s"][k, :n], pred[k, :n]
            r["p_err"].append(np.linalg.norm(prd[:, 0] - tru[:, 0]) / np.linalg.norm(tru[:, 0]))
            r["tau_err"].append(np.linalg.norm(prd[:, 1:] - tru[:, 1:]) / np.linalg.norm(tru[:, 1:]))
            # signed bias: mean error relative to the mean magnitude (0 = errors scatter around the truth)
            r["p_bias"].append(np.mean(prd[:, 0] - tru[:, 0]) / np.mean(np.abs(tru[:, 0])))
            r["tau_bias"].append(np.mean(np.linalg.norm(prd[:, 1:], axis=1) - np.linalg.norm(tru[:, 1:], axis=1))
                                 / np.mean(np.linalg.norm(tru[:, 1:], axis=1)))
            cl_t, cd_t, pt = forces(surf["s"], surf, k, n)
            cl_p, cd_p, pp = forces(pred, surf, k, n)
            r["cl"].append((cl_p - cl_t) / abs(cl_t))
            r["cd"].append((cd_p - cd_t) / abs(cd_t))
            for part in ["clp", "clv", "cdp", "cdv"]:  # error of each part, relative to the total coefficient
                total = cl_t if part.startswith("cl") else cd_t
                r[part].append((pp[part] - pt[part]) / abs(total))
            le = surf["pts"][k, :n, 0] < LE
            cl_le, cd_le, _ = forces(pred, surf, k, n, keep=le)
            r["le_share_cl"].append(abs(cl_le - cl_t) / max(abs(cl_p - cl_t), 1e-12))
            r["le_share_cd"].append(abs(cd_le - cd_t) / max(abs(cd_p - cd_t), 1e-12))
        rows[run] = (epoch, {k: np.array(v) for k, v in r.items()})

    pct = lambda v: f"{100 * v:7.2f}%"
    print(f"{N} validation cases\n")
    print(f"{'metric':52s}" + "".join(f"{run + ' (ep ' + str(e) + ')':>18s}" for run, (e, _) in rows.items()))
    lines = [
        ("surface pressure error, mean", lambda r: pct(r["p_err"].mean())),
        ("wall shear error, mean", lambda r: pct(r["tau_err"].mean())),
        ("surface pressure bias, mean signed", lambda r: pct(r["p_bias"].mean())),
        ("wall shear magnitude bias, mean signed", lambda r: pct(r["tau_bias"].mean())),
        ("CL error, median abs", lambda r: pct(np.median(np.abs(r["cl"])))),
        ("  from pressure, median abs (share of CL)", lambda r: pct(np.median(np.abs(r["clp"])))),
        ("  from friction, median abs (share of CL)", lambda r: pct(np.median(np.abs(r["clv"])))),
        ("  signed mean (systematic part)", lambda r: pct(r["cl"].mean())),
        ("  share of CL error from leading edge (x < 0.05)", lambda r: pct(np.median(r["le_share_cl"]))),
        ("CD error, median abs", lambda r: pct(np.median(np.abs(r["cd"])))),
        ("  from pressure, median abs (share of CD)", lambda r: pct(np.median(np.abs(r["cdp"])))),
        ("  from friction, median abs (share of CD)", lambda r: pct(np.median(np.abs(r["cdv"])))),
        ("  signed mean (systematic part)", lambda r: pct(r["cd"].mean())),
        ("  pressure and friction errors cancel (share of cases)",
         lambda r: pct(np.mean(np.sign(r["cdp"]) != np.sign(r["cdv"])))),
        ("  share of CD error from leading edge (x < 0.05)", lambda r: pct(np.median(r["le_share_cd"]))),
    ]
    for name, f in lines:
        print(f"{name:52s}" + "".join(f"{f(r):>18s}" for _, r in rows.values()))
    a, b = [r for _, r in rows.values()][:2]
    print(f"\ncorrelation of per-case wall-shear error with per-case CD error: "
          + "  ".join(f"{run} {np.corrcoef(r['tau_err'], np.abs(r['cd']))[0, 1]:.2f}" for run, (_, r) in rows.items()))


if __name__ == "__main__":
    main(sys.argv[1:] or ["T30", "T30c"])
