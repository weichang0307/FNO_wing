"""Does the simulator's own geometry give the same answers as the dataset's CFD surfaces?

For every test case: rebuild the airfoil from its NACA parameters with the simulator's predictor,
predict, and compare CL and CD with the CFD values and with evaluate.py's predictions (which used
the CFD mesh's surface points). Usage: python checks/validate_generated_geometry.py R3
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "simulator"))
from predictor import Predictor  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "R3"
ref = pd.read_csv(os.path.join(ROOT, "results", f"{run}_test.csv"))
pred = Predictor(run)
rows = []
for r in ref.itertuples():
    parts = r.name.split("_")
    params = [float(v) for v in parts[4:]]
    out, info = pred.predict(params, [r.aoa], r.re)
    rows.append(dict(cl_gen=out[0]["cl"], cd_gen=out[0]["cd"], ms=info["ms"]))
g = pd.concat([ref, pd.DataFrame(rows)], axis=1)
rel = lambda a, b: (g[a] - g[b]).abs() / g[b].abs()
print(f"{len(g)} test cases, model {run}\n")
print(f"{'':38s}{'median':>9s}{'90th pct':>10s}")
for label, a, b in [("CL error, CFD mesh surface (evaluate.py)", "cl_pred", "cl_true"),
                    ("CL error, generated geometry", "cl_gen", "cl_true"),
                    ("CD error, CFD mesh surface (evaluate.py)", "cd_pred", "cd_true"),
                    ("CD error, generated geometry", "cd_gen", "cd_true"),
                    ("CL generated vs mesh prediction", "cl_gen", "cl_pred"),
                    ("CD generated vs mesh prediction", "cd_gen", "cd_pred")]:
    e = rel(a, b)
    print(f"{label:38s}{100 * e.median():8.2f}%{100 * e.quantile(0.9):9.2f}%")
print(f"\nmedian inference {g.ms.median():.1f} ms")
