# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "soundfile", "pillow", "huggingface_hub"]
# ///
"""How well do the RE-USE models restore the band above 8 kHz from a 16 kHz recording?

    uv run reference/bandwidth_extension.py <work dir> [noise-reduction binary]

NVIDIA's test recordings (44.1 kHz) are resampled to 48 kHz ("full band") and to 16 kHz
("narrow band"). Each model enhances the full-band file at 48 kHz, and the narrow-band
file with `--output-rate 48000` (bandwidth extension). The extension is scored against
the same model's full-band output, which is what it would have produced had the high
band been recorded: log-spectral distance (LSD) per band and the energy above 8 kHz.
"""

from __future__ import annotations

import subprocess
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy.signal import correlate, resample_poly, stft

SR = 48000
FILES = [
    ("nvidia/Real-time_RE-USE", "4c0c186272ca853cd18d832fc5c28777b62f32d1", "noisy_audio/mic_test.wav"),
    ("nvidia/NVIDIA-NemotronLabs-RE-USE", "31f82f2efad1095034492f236e8dac9239ba713b", "noisy_audio/mic_test2.wav"),
]
SYSTEMS = {
    "reuse_e12": ["-e", "12", "--offline"],
    "nemotron": ["--engine", "nemotron"],
}


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, _, z = stft(x, SR, nperseg=2048, noverlap=1536)
    return f, np.abs(z) ** 2 + 1e-10


def lsd(a: np.ndarray, b: np.ndarray, rows: np.ndarray) -> float:
    d = 10 * np.log10(a[rows]) - 10 * np.log10(b[rows])
    return float(np.mean(np.sqrt(np.mean(d**2, axis=0))))


def align(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Shift x so its low band lines up with ref (the resampler may add a delay)."""
    n = min(len(ref), len(x), SR * 3)
    lag = int(np.argmax(correlate(x[:n], ref[:n], mode="full", method="fft"))) - (n - 1)
    x = x[lag:] if lag > 0 else np.concatenate([np.zeros(-lag, np.float32), x])
    m = min(len(ref), len(x))
    return x[:m]


def image(x: np.ndarray, path: Path) -> None:
    f, p = spectrum(x)
    keep = f <= 24000
    db = 10 * np.log10(p[keep])[::-1]
    v = np.clip((db + 100) / 80, 0, 1)
    stops = np.array([[12, 14, 22], [38, 44, 92], [116, 56, 120], [214, 96, 66], [250, 200, 90], [255, 248, 220]], float)
    t = v * (len(stops) - 1)
    i = np.clip(t.astype(int), 0, len(stops) - 2)
    w = (t - i)[..., None]
    rgb = (stops[i] * (1 - w) + stops[i + 1] * w).astype(np.uint8)
    Image.fromarray(rgb).resize((900, 300)).save(path)


def main() -> None:
    work = Path(sys.argv[1])
    cli = sys.argv[2] if len(sys.argv) > 2 else ".build/xcode/Build/Products/Release/noise-reduction"
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for repo, rev, name in FILES:
        audio, sr = sf.read(hf_hub_download(repo, name, revision=rev), dtype="float32")
        stem = Path(name).stem
        full, narrow = resample(audio, sr, SR), resample(audio, sr, 16000)
        sf.write(work / f"{stem}_full.wav", full, SR, subtype="FLOAT")
        sf.write(work / f"{stem}_16k.wav", narrow, 16000, subtype="FLOAT")
        image(full, work / f"{stem}_input_full.png")
        image(resample(narrow, 16000, SR), work / f"{stem}_input_16k.png")
        for system, args in SYSTEMS.items():
            ref_path, bwe_path = work / f"{stem}_{system}_full.wav", work / f"{stem}_{system}_bwe.wav"
            subprocess.run([cli, "enhance", str(work / f"{stem}_full.wav"), str(ref_path), *args], check=True)
            subprocess.run(
                [cli, "enhance", str(work / f"{stem}_16k.wav"), str(bwe_path), "--output-rate", str(SR), *args], check=True
            )
            ref, _ = sf.read(ref_path, dtype="float32")
            bwe, _ = sf.read(bwe_path, dtype="float32")
            bwe = align(ref, bwe)
            ref = ref[: len(bwe)]
            plain = align(ref, resample(narrow, 16000, SR))
            f, pr = spectrum(ref)
            _, pb = spectrum(bwe)
            _, pp = spectrum(plain[: len(ref)])
            low, high = (f > 100) & (f < 7500), (f > 8500) & (f < 20000)
            high_db = lambda p: 10 * np.log10(p[high].sum() / p[low].sum())  # noqa: E731
            rows.append((stem, system, lsd(pr, pb, low), lsd(pr, pb, high), lsd(pr, pp, high),
                         high_db(pr), high_db(pb), high_db(pp)))
            image(ref, work / f"{stem}_{system}_full.png")
            image(bwe, work / f"{stem}_{system}_bwe.png")
    print("file       system      LSD 0-7.5k  LSD 8.5-20k (BWE / plain upsample)  energy above 8 kHz vs below, dB (full / BWE / plain)")
    for r in rows:
        print(f"{r[0]:10s} {r[1]:10s}  {r[2]:6.2f}      {r[3]:6.2f} / {r[4]:6.2f}                       {r[5]:6.1f} / {r[6]:6.1f} / {r[7]:6.1f}")


if __name__ == "__main__":
    main()
