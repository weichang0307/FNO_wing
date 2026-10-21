"""Import runs logged before TensorBoard was added (metrics.csv only) into TensorBoard.

Writes runs/<run-id>/tb_import with the same tags train.py logs. Training loss only exists at
evaluation epochs in metrics.csv, so those curves are coarser than in natively logged runs.

Usage:
  python csv_to_tensorboard.py            # every run that has metrics.csv but no native tb/ logs
  python csv_to_tensorboard.py R3 --follow  # keep re-importing a run that is still training
"""
import argparse
import os
import shutil
import time

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
TAGS = {  # metrics.csv column -> (TensorBoard tag, scale)
    "train_loss": ("loss/train", 1), "val_score": ("loss/val_score", 1), "lr": ("time/lr", 1),
    "epoch_s": ("time/epoch_s", 1),
    "val_u/U": ("val_fields/u_%", 100), "val_v/U": ("val_fields/v_%", 100),
    "val_p/U2": ("val_fields/p_%", 100), "val_nut/nu": ("val_fields/nu_t_%", 100),
    "val_surf_p": ("val_surface/pressure_%", 100), "val_surf_tau": ("val_surface/wall_shear_%", 100),
    "val_cl_median_rel": ("val_forces/CL_median_%", 100), "val_cd_median_rel": ("val_forces/CD_median_%", 100),
    "val_cl_mean_abs": ("val_forces/CL_mean_abs", 1), "val_cl_spearman": ("val_forces/CL_spearman", 1),
}


def import_run(run):
    csv = os.path.join(RUNS, run, "metrics.csv")
    m = pd.read_csv(csv)
    out = os.path.join(RUNS, run, "tb_import")
    shutil.rmtree(out, ignore_errors=True)  # rewrite from scratch so re-imports never duplicate points
    tb = SummaryWriter(out)
    for _, row in m.iterrows():
        for col, (tag, scale) in TAGS.items():
            if col in m.columns and pd.notna(row[col]):
                tb.add_scalar(tag, scale * float(row[col]), int(row["epoch"]))
    tb.close()
    return len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--follow", action="store_true", help="re-import every minute until the run finishes")
    args = ap.parse_args()
    runs = args.runs or sorted(r for r in os.listdir(RUNS) if os.path.exists(os.path.join(RUNS, r, "metrics.csv"))
                               and not os.path.isdir(os.path.join(RUNS, r, "tb")))
    while True:
        for run in runs:
            if os.path.getsize(os.path.join(RUNS, run, "metrics.csv")):
                print(f"{run}: {import_run(run)} evaluation points", flush=True)
        if not args.follow or all(os.path.exists(os.path.join(RUNS, r, "summary.json")) for r in runs):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
