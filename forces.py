"""Lift and drag from surface pressure and wall shear stress, matching airfrans' force integration.

Works on numpy arrays or torch tensors for one case:
  p        (Ns,)    kinematic pressure at surface points (p / rho, m2/s2)
  wss      (Ns, 2)  kinematic wall shear stress at surface points (m2/s2)
  lines    (Nc, 2)  surface segments as pairs of point indices
  normals  (Nc, 2)  segment normals, pointing into the airfoil (AirfRANS convention)
  length   (Nc,)    segment lengths (chord = 1)
Dimensionless inputs (p / U_inf^2, wss / U_inf^2) work too with u_inf = 1.
"""
import math


def force_coefficients(p, wss, lines, normals, length, u_inf, aoa_deg):
    """Returns (cl, cd, parts) with parts = dict(clp, clv, cdp, cdv)."""
    a, b = lines[:, 0], lines[:, 1]
    p_cell = 0.5 * (p[a] + p[b])  # airfrans averages point values onto each segment
    wss_cell = 0.5 * (wss[a] + wss[b])
    f_p = (p_cell[:, None] * normals * length[:, None]).sum(0)
    f_v = (wss_cell * length[:, None]).sum(0)
    aoa = math.radians(aoa_deg)
    drag_dir = (math.cos(aoa), math.sin(aoa))
    lift_dir = (-math.sin(aoa), math.cos(aoa))
    q = 0.5 * u_inf ** 2
    along = lambda f, d: (f[0] * d[0] + f[1] * d[1]) / q
    parts = dict(cdp=along(f_p, drag_dir), cdv=along(f_v, drag_dir),
                 clp=along(f_p, lift_dir), clv=along(f_v, lift_dir))
    return parts["clp"] + parts["clv"], parts["cdp"] + parts["cdv"], parts
