"""FNO with a surface head.

The FNO maps the input grid to the four flow fields as before. The surface head reads the FNO's
last hidden field (64 channels, before the projection) at the true airfoil surface points by
bilinear sampling, adds each point's position and normal and the freestream inputs, mixes
neighbouring points with dilated 1D convolutions around the closed surface loop, and predicts
pressure and wall shear there. Forces come from the surface outputs, because the grid cannot
resolve the wall velocity gradient that sets skin friction.
"""
import torch
import torch.nn.functional as F
from neuralop.models import FNO

S_NAMES = ["p/U2", "tau_x/U2", "tau_y/U2"]


class LoopConv(torch.nn.Module):
    """Residual 1D convolution along each case's closed surface loop.

    Cases have different point counts, so instead of a padded Conv1d the k-th neighbour of every
    point is gathered by its place on the loop, wrapping around at the trailing edge.
    """

    def __init__(self, ch, kernel, dilation):
        super().__init__()
        r = kernel // 2
        self.register_buffer("offsets", torch.arange(-r, r + 1) * dilation, persistent=False)
        self.mix = torch.nn.Linear(kernel * ch, ch)
        self.norm = torch.nn.LayerNorm(ch)

    def forward(self, h, order, pos, n_pts):
        B, S, C = h.shape
        place = (pos[..., None] + self.offsets) % n_pts[:, None, None]       # (B, S, K) places on the loop
        nbr = torch.gather(order, 1, place.flatten(1)).view(B, S, -1)          # (B, S, K) point indices
        g = torch.gather(self.norm(h), 1, nbr.flatten(1)[..., None].expand(-1, -1, C))  # (B, S*K, C)
        return h + F.gelu(self.mix(g.view(B, S, len(self.offsets) * C)))


class UNet(torch.nn.Module):
    """Convolutional baseline backbone: 4-level U-Net returning a 64-channel hidden field at full
    resolution, so the projection and surface head are the same as the FNO's."""

    def __init__(self, in_ch, width=64, levels=4):
        super().__init__()
        block = lambda a, b: torch.nn.Sequential(
            torch.nn.Conv2d(a, b, 3, padding=1), torch.nn.GroupNorm(8, b), torch.nn.GELU(),
            torch.nn.Conv2d(b, b, 3, padding=1), torch.nn.GroupNorm(8, b), torch.nn.GELU())
        chs = [width * 2 ** i for i in range(levels)]  # 64, 128, 256, 512
        self.down = torch.nn.ModuleList(block(a, b) for a, b in zip([in_ch] + chs[:-1], chs))
        self.up = torch.nn.ModuleList(block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(levels - 1)))

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for blk, skip in zip(self.up, reversed(skips[:-1])):
            x = blk(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        return x


class SurfaceFNO(torch.nn.Module):
    def __init__(self, cfg, box):
        super().__init__()
        self.arch = cfg.get("arch", "fno")
        if self.arch == "fno":
            self.fno = FNO(n_modes=(cfg["modes"], cfg["modes"]), hidden_channels=cfg["width"], in_channels=4,
                           out_channels=4, n_layers=cfg["layers"], positional_embedding="grid",
                           domain_padding=cfg["padding"] or None)
        else:  # U-Net baseline: same inputs plus the same x, y coordinate channels the FNO appends
            self.unet = UNet(4 + 2, cfg["width"])
            self.unet_proj = torch.nn.Conv2d(cfg["width"], 4, 1)
        self.box = box  # x0, x1, y0, y1 of the grid, in chord lengths
        hidden = cfg.get("head_width", 128)
        n_in = cfg["width"] + 2 + 2 + 2  # sampled features, position, normal, freestream
        self.head_in = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.GELU(),
                                           torch.nn.Linear(hidden, hidden))
        # dilations 1, 2, 4, 8 with 9-point kernels: each point sees about 60 neighbours either side
        self.head_conv = torch.nn.ModuleList(
            LoopConv(hidden, cfg.get("head_kernel", 9), 2 ** i) for i in range(cfg.get("head_conv_layers", 4)))
        self.head_out = torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(hidden, 3))

    def hidden(self, x):
        """The backbone forward pass up to (not including) the projection."""
        if self.arch != "fno":
            B, _, H, W = x.shape
            ys = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
            xs = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
            return self.unet(torch.cat([x, ys, xs], 1))
        f = self.fno
        if f.positional_embedding is not None:
            x = f.positional_embedding(x)
        x = f.lifting(x)
        if f.domain_padding is not None:
            x = f.domain_padding.pad(x)
        for i in range(f.n_layers):
            x = f.fno_blocks(x, i)
        if f.domain_padding is not None:
            x = f.domain_padding.unpad(x)
        return x

    def forward(self, x, s_pts, s_nrm, order, pos, n_pts):
        """x (B, 4, H, W) normalized inputs; s_pts, s_nrm (B, S, 2); order, pos (B, S) and n_pts (B,)
        describe each surface loop. Returns fields (B, 4, H, W), surface (B, S, 3)."""
        h = self.hidden(x)
        fields = self.fno.projection(h) if self.arch == "fno" else self.unet_proj(h)
        x0, x1, y0, y1 = self.box
        g = torch.stack([2 * (s_pts[..., 0] - x0) / (x1 - x0) - 1,
                         2 * (s_pts[..., 1] - y0) / (y1 - y0) - 1], dim=-1)[:, :, None]  # (B, S, 1, 2)
        feat = F.grid_sample(h, g, align_corners=False).squeeze(-1).transpose(1, 2)  # (B, S, width)
        free = x[:, 2:4, 0, 0][:, None].expand(-1, s_pts.shape[1], -1)  # normalized U_inf x, y
        s = self.head_in(torch.cat([feat, s_pts, s_nrm, free], dim=-1))
        for conv in self.head_conv:
            s = conv(s, order, pos, n_pts)
        return fields, self.head_out(s)
