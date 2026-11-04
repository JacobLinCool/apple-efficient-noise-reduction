# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "torch==2.5.1",
#   "coremltools==8.3.0",
#   "einops",
#   "numpy<2.2",
#   "safetensors",
#   "huggingface_hub",
# ]
# ///
"""Would the Neural Engine help RE-USE? Times the convolutional parts on Core ML.

Exports NVIDIA's own encoder and decoder modules (dense blocks with sub-pixel up-sampling;
about 40% of the network's MACs) for one chunk, converts them to Core ML in float16 and
times them on each compute unit; the Mamba blocks stay out of it (ANE has no selective
scan). Run the energy meter alongside to read ANE / GPU power.

    uv run reference/ane_experiment.py [--frames 24] [--rate 16000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

REPO, REVISION = "nvidia/Real-time_RE-USE", "4c0c186272ca853cd18d832fc5c28777b62f32d1"


def make_even(v):
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


class Convs(torch.nn.Module):
    """Encoder, then both decoders of one exit layer, without the Mamba blocks in between."""

    def __init__(self, model, exit_layer):
        super().__init__()
        self.enc = model.dense_encoder
        self.mag = model.mask_decoder_list[exit_layer - 1]
        self.pha = model.phase_decoder_list[exit_layer - 1]

    def forward(self, x):
        h = self.enc(x, 0)
        return self.mag(h), self.pha(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    root = Path(snapshot_download(REPO, revision=REVISION))
    sys.path.insert(0, str(root))
    sys.modules["mamba_ssm"] = types.SimpleNamespace(Mamba=torch.nn.Identity)  # unused here
    from models.streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv import SEMamba_decoder_list

    cfg = json.loads((root / "config.json").read_text())
    model = SEMamba_decoder_list(cfg).eval()
    model.load_state_dict(load_file(root / "model.safetensors"), strict=False)
    net = Convs(model, 8).eval()

    bins = make_even(cfg["stft_cfg"]["n_fft"] * args.rate // 8000) // 2 + 1
    x = torch.randn(1, 2, args.frames, bins + 2)
    traced = torch.jit.trace(net, x)
    with torch.no_grad():
        ref = [t.numpy() for t in net(x)]

    print(f"{args.rate} Hz, {args.frames} frames/call, input {tuple(x.shape)}")
    for name, unit in [("ANE", ct.ComputeUnit.CPU_AND_NE), ("GPU", ct.ComputeUnit.CPU_AND_GPU), ("CPU", ct.ComputeUnit.CPU_ONLY)]:
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=x.shape)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=unit,
            minimum_deployment_target=ct.target.macOS15,
        )
        feed = {"x": x.numpy()}
        out = ml.predict(feed)
        got = list(out.values())
        # Outputs come back unnamed; pair each reference with its closest output.
        err = max(min(float(np.abs(g - r).max() / (np.abs(r).max() + 1e-9)) for g in got) for r in ref)
        for _ in range(5):
            ml.predict(feed)
        times = []
        start = time.monotonic()
        t_end = start + args.seconds
        while time.monotonic() < t_end:
            t0 = time.perf_counter()
            ml.predict(feed)
            times.append(time.perf_counter() - t0)
        times.sort()
        print(
            f"{name}: median {1e3 * times[len(times) // 2]:.2f} ms, p95 {1e3 * times[int(len(times) * 0.95)]:.2f} ms "
            f"over {len(times)} calls, max rel err vs fp32 torch {err:.1e} [window {start:.3f} {time.monotonic():.3f} calls {len(times)}]",
            flush=True,
        )
        time.sleep(3)


if __name__ == "__main__":
    main()
