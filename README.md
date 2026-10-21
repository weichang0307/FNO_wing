# FNO_wing

Fourier Neural Operator surrogate for 2D airfoil CFD, trained on AirfRANS. Model v1 (`models/v1`)
predicts the flow field, surface pressure, skin friction, lift and drag of NACA 4- and 5-digit
airfoils in milliseconds.

## Run the simulator on a Mac

The interactive simulator runs on an Apple GPU or the CPU; it does not need the training data,
CUDA or airfrans. On a CPU one prediction takes about 0.1 s, so the sliders stay live.

1. Clone this repository and go into it.

2. Check that the model weights came with the clone: `models/v1/best.pt` should be 74 MB. They
   are in the repository, so nothing needs copying.

3. Create a Python environment and install the requirements. It needs Python 3.10 or newer; check
   with `python3 --version`, and if it is older, install a current one with `brew install python`.

   ```bash
   python3 -m venv .venv && source .venv/bin/activate && pip install -r simulator/requirements.txt
   ```

4. Start the simulator:

   ```bash
   python simulator/server.py --run v1 --port 8770
   ```

   It prints the device it chose (`mps` for the Apple GPU, otherwise `cpu`). It uses the Apple GPU
   only if a test prediction matches the CPU's; otherwise it falls back to the CPU and says why.
   Force a device with `--device cpu` or `--device mps`.

5. Open http://localhost:8770 in a browser. Stop the server with Ctrl+C.

Without an NVIDIA GPU the lift and drag curves are computed every 2° instead of every 1°.
