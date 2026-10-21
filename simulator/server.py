"""Local server for the interactive airfoil simulator.

Serves simulator/index.html and two endpoints backed by the trained model on the GPU:
  POST /api/predict {family, params, aoa, re}  -> forces, surface Cp and the flow field
  POST /api/polar   {family, params, re}       -> CL and CD from -5 to 15 degrees in one batch
Usage: python simulator/server.py [--run R3 --port 8770]
"""
import argparse
import base64
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from predictor import RANGES, Predictor, in_range

HERE = os.path.dirname(os.path.abspath(__file__))
NU = 1.56e-5
lock = threading.Lock()  # one GPU call at a time
pred = None


def b64(a, dtype=np.float32):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def parse(body):
    req = json.loads(body)
    family = req["family"]
    p = [float(v) for v in req["params"]]
    # NACA 4: camber %, position (tenths of chord), thickness %; NACA 5: L, P, Q (0/1), thickness %
    params = p[:3] if family == "naca4" else [p[0], p[1], round(p[2]), p[3]]
    return family, params, float(req.get("aoa", 0)), float(req["re"])


def predict(family, params, aoa, re):
    with lock:
        out, info = pred.predict(params, [aoa], re)
    o, f = out[0], out[0]["fields"]
    pts = info["points"]
    le = int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))
    return dict(
        cl=o["cl"], cd=o["cd"], parts=o["parts"], ms=info["ms"], u_inf=info["u_inf"],
        off_range=in_range(family, params, aoa, re), params=params,
        surface=dict(x=pts[:, 0].round(5).tolist(), y=pts[:, 1].round(5).tolist(),
                     cp=(2 * o["surface"][:, 0]).round(4).tolist(), le=le),
        field=dict(H=pred.H, W=pred.W, box=list(pred.cfg["box"]),
                   cp=b64(2 * f[2]), ux=b64(f[0]), uy=b64(f[1]), nut=b64(f[3]),
                   fluid=b64(info["fluid"], np.uint8)))


def polar(family, params, re, lo=-5.0, hi=15.0):
    aoa = list(np.arange(lo, hi + 0.01, 1.0))  # follows the page's angle slider, 1 degree apart
    with lock:
        out, info = pred.predict(params, aoa, re)
    return dict(aoa=aoa, cl=[o["cl"] for o in out], cd=[o["cd"] for o in out], ms=info["ms"])


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/info"):
            return self.send_json(dict(run=pred.run, epoch=pred.epoch, ranges=RANGES))
        return super().do_GET()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            family, params, aoa, re = parse(body)
            if self.path == "/api/predict":
                return self.send_json(predict(family, params, aoa, re))
            if self.path == "/api/polar":
                req = json.loads(body)
                return self.send_json(polar(family, params, re, float(req.get("aoa_min", -5)), float(req.get("aoa_max", 15))))
            self.send_error(404)
        except Exception as e:  # report bad input to the page instead of dropping the connection
            self.send_json(dict(error=str(e)), 400)

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main():
    global pred
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="R3")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    pred = Predictor(args.run)
    pred.predict([2, 4, 12], [0.0], 3e6)  # warm up the GPU and build the spacing template
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        if e.errno == 98:
            raise SystemExit(f"Port {args.port} is already in use, probably by a simulator that is already "
                             f"running: open http://localhost:{args.port}, or start this one with --port {args.port + 1}.")
        raise
    print(f"simulator: model {args.run} (epoch {pred.epoch}) at http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
