# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy>=2,<2.3",
#   "scipy",
#   "soundfile",
#   "pandas",
#   "pyarrow",
#   "huggingface_hub",
#   "pesq",
#   "pystoi",
#   "torch>=2.5",
#   "torchmetrics>=1.6",
#   "onnxruntime",
#   "librosa",
#   "requests",
# ]
# ///
"""Speech quality on the VoiceBank-DEMAND test set (824 utterances, 16 kHz).

    uv run reference/quality_vbd.py prepare <dir>       # <dir>/noisy, <dir>/clean
    noise-reduction enhance-batch <dir>/noisy <dir>/reuse ... # any number of systems
    uv run reference/quality_vbd.py score <dir> noisy reuse dfn3 ...

Outputs are scored at 16 kHz against the clean references (PESQ wide-band, ESTOI,
SI-SDR) and without them (DNSMOS P.835 OVRL / SIG / BAK). Systems that ran at another
rate (DFN3 at 48 kHz) are resampled back to 16 kHz first.
"""

from __future__ import annotations

import io
import sys
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

SR = 16000


def prepare(root: Path) -> None:
    path = hf_hub_download("JacobLinCool/VoiceBank-DEMAND-16k", "data/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    for kind in ("noisy", "clean"):
        (root / kind).mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        for kind in ("noisy", "clean"):
            audio, sr = sf.read(io.BytesIO(row[kind]["bytes"]), dtype="float32")
            assert sr == SR
            sf.write(root / kind / f"{row['id']}.wav", audio, SR, subtype="FLOAT")
    print(f"{len(df)} utterances in {root}")


def load(path: Path, length: int) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    x = x[:length]
    return np.pad(x, (0, length - len(x)))


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    s = (est @ ref) / (ref @ ref + 1e-12) * ref
    return 10 * np.log10((s @ s) / ((est - s) @ (est - s) + 1e-12))


def score(root: Path, systems: list[str]) -> None:
    import torch
    from pesq import pesq
    from pystoi import stoi
    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score as dnsmos

    names = sorted(p.stem for p in (root / "clean").glob("*.wav"))
    print("system   PESQ   ESTOI  SI-SDR  DNSMOS OVRL  SIG    BAK")
    for system in systems:
        rows = []
        for name in names:
            clean, _ = sf.read(root / "clean" / f"{name}.wav", dtype="float32")
            est = load(root / system / f"{name}.wav", len(clean))
            mos = dnsmos(torch.from_numpy(est), SR, personalized=False).numpy()  # p808, sig, bak, ovrl
            rows.append(
                (
                    pesq(SR, clean, est, "wb"),
                    stoi(clean, est, SR, extended=True),
                    si_sdr(clean, est),
                    mos[3],
                    mos[1],
                    mos[2],
                )
            )
        m = np.mean(rows, axis=0)
        print(f"{system:8s} {m[0]:.3f}  {m[1]:.3f}  {m[2]:6.2f}  {m[3]:.3f}        {m[4]:.3f}  {m[5]:.3f}", flush=True)


if __name__ == "__main__":
    cmd, root = sys.argv[1], Path(sys.argv[2])
    if cmd == "prepare":
        prepare(root)
    else:
        score(root, sys.argv[3:])
