"""Train the FNO with a surface head on gridded AirfRANS (Phase 3).

Loss = field relative L2 (u, v, p, nu_t over fluid cells)
     + surface-weight x surface relative L2 (pressure, and the wall shear vector, at surface points)
Validation also scores CL and CD integrated from the predicted surface pressure and wall shear.

Examples:
  python train.py --run-id R0 --n-train 16 --epochs 400 --eval-every 50   # smoke test: must overfit
  python train.py --run-id R3                                              # FNO v1, full split
Writes runs/<run-id>/: config.json, norm_stats.json, best.pt, last.pt, metrics.csv, summary.json,
and TensorBoard logs in runs/<run-id>/tb (view with: tensorboard --logdir runs)
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter

from fno_data import (Normalizer, Y_NAMES, compute_stats, grid_box, load_dimensionless, load_surface,
                      save_stats, split_indices)
from forces import force_coefficients
from model import SurfaceFNO

ROOT = os.path.dirname(os.path.abspath(__file__))


def masked_rel_l2(pred, true, mask):
    """Relative L2 per case and channel over fluid cells, in dimensionless units. Shape (B, 4)."""
    m = mask[:, None]
    num = ((pred - true) * m).flatten(2).norm(dim=2)
    den = (true * m).flatten(2).norm(dim=2).clamp_min(1e-8)
    return num / den


def surface_rel_l2(pred, true, smask):
    """Relative L2 per case for surface pressure and for the wall shear vector. Shape (B, 2)."""
    m = smask[..., None]
    err = (pred - true) * m
    tru = true * m
    p = err[..., 0].norm(dim=1) / tru[..., 0].norm(dim=1).clamp_min(1e-8)
    tau = err[..., 1:].flatten(1).norm(dim=1) / tru[..., 1:].flatten(1).norm(dim=1).clamp_min(1e-8)
    return torch.stack([p, tau], dim=1)


def log_tb(tb, epoch, v, score):
    """Validation metrics in percent, grouped so related curves share a TensorBoard section."""
    for name, e in zip(["u", "v", "p", "nu_t"], v["field"]):
        tb.add_scalar(f"val_fields/{name}_%", 100 * e, epoch)
    tb.add_scalar("val_surface/pressure_%", 100 * v["surf"][0], epoch)
    tb.add_scalar("val_surface/wall_shear_%", 100 * v["surf"][1], epoch)
    tb.add_scalar("val_forces/CL_median_%", 100 * v["cl_med"], epoch)
    tb.add_scalar("val_forces/CD_median_%", 100 * v["cd_med"], epoch)
    tb.add_scalar("val_forces/CL_mean_abs", v["cl_abs"], epoch)
    tb.add_scalar("val_forces/CL_spearman", v["spearman"], epoch)
    tb.add_scalar("loss/val_score", score, epoch)


class Data:
    """One split held on the GPU."""

    def __init__(self, h5, idx, dev):
        x, y, _ = load_dimensionless(h5, idx)
        self.surf = load_surface(h5, idx)
        self.x, self.y = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
        for k in ["pts", "nrm", "s", "smask", "order", "pos", "n_pts"]:
            setattr(self, k, torch.from_numpy(self.surf[k]).to(dev))
        self.x_np, self.y_np = x, y

    def __len__(self):
        return len(self.x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "airfrans_128x256.h5"))
    ap.add_argument("--task", default="full", choices=["full", "scarce", "reynolds", "aoa"])
    ap.add_argument("--n-train", type=int, default=0, help="use only the first N training cases (smoke test)")
    ap.add_argument("--arch", default="fno", choices=["fno", "unet"], help="backbone; unet is the R1 baseline")
    ap.add_argument("--modes", type=int, default=32)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--padding", type=float, default=0.125)
    ap.add_argument("--head-width", type=int, default=128)
    ap.add_argument("--head-conv-layers", type=int, default=4, help="0 = no convolution along the surface (pointwise head)")
    ap.add_argument("--head-kernel", type=int, default=9)
    ap.add_argument("--surface-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    cfg = vars(ap.parse_args())

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    run_dir = os.path.join(ROOT, "runs", cfg["run_id"])
    os.makedirs(run_dir, exist_ok=True)

    tr_idx, va_idx, _ = split_indices(cfg["data"], cfg["task"], seed=0)
    if cfg["n_train"]:
        tr_idx = tr_idx[:cfg["n_train"]]
    tr, va = Data(cfg["data"], tr_idx, dev), Data(cfg["data"], va_idx, dev)
    with __import__("h5py").File(cfg["data"], "r") as h5:
        va_cl, va_cd = h5["cl"][np.sort(va_idx)], h5["cd"][np.sort(va_idx)]
    stats = compute_stats(tr.x_np, tr.y_np, tr.surf)
    save_stats(stats, os.path.join(run_dir, "norm_stats.json"))
    cfg.update(n_train_cases=len(tr_idx), n_val_cases=len(va_idx), box=grid_box(cfg["data"]))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    norm = Normalizer(stats).to(dev)
    model = SurfaceFNO(cfg, cfg["box"]).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    floor = 1e-5 / cfg["lr"]

    def lr_factor(epoch):  # linear warm-up, then cosine decay to 1e-5
        if epoch < cfg["warmup"]:
            return (epoch + 1) / cfg["warmup"]
        t = (epoch - cfg["warmup"]) / max(1, cfg["epochs"] - cfg["warmup"])
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    def predict(d, b):
        """Dimensionless fields and surface values for cases b of split d."""
        f_n, s_n = model(norm.x(d.x[b]), d.pts[b], d.nrm[b], d.order[b], d.pos[b], d.n_pts[b])
        return norm.y_inverse(f_n), norm.s_inverse(s_n)

    def evaluate(d, ref_cl=None, ref_cd=None):
        model.eval()
        f_err, s_err, cls, cds = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(d), 16):
                b = torch.arange(i, min(i + 16, len(d)), device=dev)
                f, s = predict(d, b)
                f_err.append(masked_rel_l2(f, d.y[b], d.x[b, 1]))
                s_err.append(surface_rel_l2(s, d.s[b], d.smask[b]))
                for j, k in enumerate(b.tolist()):
                    n = int(d.smask[k].sum())
                    sk = s[j, :n].double().cpu().numpy()
                    cl, cd, _ = force_coefficients(sk[:, 0], sk[:, 1:], d.surf["lines"][k], d.surf["cell_normals"][k],
                                                   d.surf["cell_length"][k], 1.0, float(d.surf["aoa"][k]))
                    cls.append(cl)
                    cds.append(cd)
        model.train()
        out = dict(field=torch.cat(f_err).mean(0).cpu().numpy(), surf=torch.cat(s_err).mean(0).cpu().numpy())
        if ref_cl is not None:
            cls, cds = np.array(cls), np.array(cds)
            out.update(cl_med=float(np.median(np.abs(cls - ref_cl) / np.abs(ref_cl))),
                       cl_abs=float(np.mean(np.abs(cls - ref_cl))),
                       cd_med=float(np.median(np.abs(cds - ref_cd) / np.abs(ref_cd))),
                       spearman=float(spearmanr(cls, ref_cl).statistic))
        return out

    print(f"run {cfg['run_id']}: {len(tr)} train / {len(va)} val cases, {n_params / 1e6:.2f} M parameters")
    log = open(os.path.join(run_dir, "metrics.csv"), "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "lr", "train_loss", "epoch_s"] + [f"val_{n}" for n in Y_NAMES] +
                    ["val_surf_p", "val_surf_tau", "val_cl_median_rel", "val_cl_mean_abs", "val_cd_median_rel",
                     "val_cl_spearman", "val_score"])
    tb = SummaryWriter(os.path.join(run_dir, "tb"))
    # every epoch, for the live dashboard (metrics.csv only has evaluation epochs)
    epoch_log = open(os.path.join(run_dir, "train_log.csv"), "w")
    epoch_log.write("epoch,train_loss,lr,epoch_s\n")
    best, t_start = float("inf"), time.time()
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        perm = torch.randperm(len(tr), device=dev)
        total = 0.0
        for i in range(0, len(perm), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            f, s = predict(tr, b)
            loss = (masked_rel_l2(f, tr.y[b], tr.x[b, 1]).mean() +
                    cfg["surface_weight"] * surface_rel_l2(s, tr.s[b], tr.smask[b]).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        train_loss, epoch_s = total / len(tr), time.time() - t0
        epoch_log.write(f"{epoch + 1},{train_loss},{sched.get_last_lr()[0]},{epoch_s}\n")
        epoch_log.flush()
        tb.add_scalar("loss/train", train_loss, epoch + 1)
        tb.add_scalar("time/lr", sched.get_last_lr()[0], epoch + 1)
        tb.add_scalar("time/epoch_s", epoch_s, epoch + 1)

        if (epoch + 1) % cfg["eval_every"] == 0 or epoch == cfg["epochs"] - 1:
            v = evaluate(va, va_cl, va_cd)
            # model selection: field error plus the two force errors the targets are set on
            score = float(v["field"][:3].mean() + v["cl_med"] + v["cd_med"])
            writer.writerow([epoch + 1, sched.get_last_lr()[0], train_loss, epoch_s] + list(v["field"]) +
                            list(v["surf"]) + [v["cl_med"], v["cl_abs"], v["cd_med"], v["spearman"], score])
            log.flush()
            log_tb(tb, epoch + 1, v, score)
            ckpt = dict(model=model.state_dict(), norm=norm.state_dict(), cfg=cfg, epoch=epoch + 1)
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))
            if score < best:
                best = score
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
            print(f"epoch {epoch + 1:4d}  train {train_loss:.4f}  val fields " +
                  " ".join(f"{n} {100 * e:.1f}%" for n, e in zip(Y_NAMES, v["field"])) +
                  f" | surface p {100 * v['surf'][0]:.1f}% tau {100 * v['surf'][1]:.1f}%"
                  f" | CL {100 * v['cl_med']:.2f}% CD {100 * v['cd_med']:.2f}% rho {v['spearman']:.4f}"
                  f"  ({epoch_s:.1f} s/epoch)", flush=True)

    tb.close()
    t = evaluate(tr)
    summary = dict(best_val_score=best, minutes=(time.time() - t_start) / 60, params=n_params,
                   final_train_field=t["field"].tolist(), final_train_surface=t["surf"].tolist())
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done in {summary['minutes']:.1f} min; final train error fields " +
          " ".join(f"{n} {100 * e:.1f}%" for n, e in zip(Y_NAMES, t["field"])) +
          f" | surface p {100 * t['surf'][0]:.1f}% tau {100 * t['surf'][1]:.1f}%")


if __name__ == "__main__":
    main()
