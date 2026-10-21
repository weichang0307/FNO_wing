"""Phase 1 check: train an FNO on neuraloperator's small Darcy-flow dataset end to end.

Trains at 16x16 and tests at 16x16 and 32x32 (zero-shot super-resolution).
Usage: python checks/darcy_check.py
"""
from pathlib import Path

import torch
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.losses import H1Loss, LpLoss
from neuralop.models import FNO

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(0)
dev = "cuda"

data_root = ROOT / "data" / "darcy"
data_root.mkdir(parents=True, exist_ok=True)
train_loader, test_loaders, processor = load_darcy_flow_small(
    n_train=1000, batch_size=32, n_tests=[100, 50], test_resolutions=[16, 32],
    test_batch_sizes=[32, 32], data_root=data_root)
processor = processor.to(dev)

# settings from neuraloperator's own Darcy example; its 15-epoch run reaches about 0.105 test L2
model = FNO(n_modes=(8, 8), hidden_channels=24, in_channels=1, out_channels=1,
            projection_channel_ratio=2).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
l2 = LpLoss(d=2, p=2)  # relative L2, summed over the batch
h1 = H1Loss(d=2)  # relative error on values and gradients; trains smoother fields than L2


def evaluate(loader):
    # the processor only decodes outputs in eval mode; scoring in train mode compares
    # normalized fields, which inflates relative error
    model.eval()
    processor.eval()
    err, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = processor.preprocess(batch)
            out, batch = processor.postprocess(model(batch["x"]), batch)
            err += l2(out, batch["y"]).item()
            n += batch["y"].shape[0]
    return err / n


for epoch in range(40):
    model.train()
    processor.train()
    total, n = 0.0, 0
    for batch in train_loader:
        batch = processor.preprocess(batch)
        opt.zero_grad()
        loss = h1(model(batch["x"]), batch["y"])
        loss.backward()
        opt.step()
        total += loss.item()
        n += batch["y"].shape[0]
    sched.step()
    if epoch % 10 == 9 or epoch == 0:
        print(f"epoch {epoch + 1:2d}  train H1 {total / n:.4f}  test16 {evaluate(test_loaders[16]):.4f}")

e16, e32 = evaluate(test_loaders[16]), evaluate(test_loaders[32])
print(f"final relative L2 error: 16x16 {e16:.4f}, 32x32 (never trained on) {e32:.4f}")
assert e16 < 0.10, "Darcy check failed: expected below the library reference (about 0.105)"
print("DARCY CHECK PASSED")
