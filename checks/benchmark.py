"""Speed and memory of FNO configurations on this GPU, with random tensors of the real shape.

Usage: python checks/benchmark.py
Reports GPU load from other processes, since a shared GPU makes every number pessimistic.
"""
import subprocess
import time

import torch
from neuralop.models import FNO

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def build_model(cfg):
    return FNO(n_modes=(cfg["modes"], cfg["modes"]), hidden_channels=cfg["width"], in_channels=4,
               out_channels=4, n_layers=cfg["layers"], positional_embedding="grid",
               domain_padding=cfg["padding"] or None)


def gpu_load():
    q = lambda f: subprocess.run(["nvidia-smi", f"--query-{f}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip()
    util, used = q("gpu=utilization.gpu,memory.used").split(", ")
    others = len([l for l in q("compute-apps=pid").splitlines() if l.strip()])
    return f"GPU {util}% busy, {int(used) / 1024:.1f} GB used by {others} other process(es)"


def bench(H, W, modes, width, padding, B=16, iters=20):
    cfg = dict(modes=modes, width=width, layers=4, padding=padding)
    m = build_model(cfg).cuda()
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x, y = torch.randn(B, 4, H, W, device="cuda"), torch.randn(B, 4, H, W, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(iters + 3):
            if i == 3:
                torch.cuda.synchronize()
                t = time.time()
            opt.zero_grad()
            ((m(x) - y) ** 2).mean().backward()
            opt.step()
        torch.cuda.synchronize()
        step = (time.time() - t) / iters
        mem = torch.cuda.max_memory_allocated() / 2**30
        m.eval()
        with torch.no_grad():
            for _ in range(5):
                m(x[:1])
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(50):
                m(x[:1])
            torch.cuda.synchronize()
        infer = (time.time() - t) / 50 * 1000
        result = f"step {step * 1000:5.0f} ms, epoch (45 steps) {step * 45:5.1f} s, peak {mem:4.1f} GB, infer {infer:5.1f} ms"
    except torch.OutOfMemoryError:
        result = "out of memory"
    p = sum(q.numel() for q in m.parameters()) / 1e6
    print(f"{H}x{W} modes {modes} width {width} padding {padding}: {p:.1f} M params, {result}", flush=True)
    del m, opt, x, y
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(gpu_load())
    bench(128, 256, 32, 64, 0.0)
    bench(128, 256, 32, 64, 0.125)
    bench(256, 512, 32, 64, 0.125, B=8, iters=10)
    print(gpu_load())
