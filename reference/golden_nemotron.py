# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.4",
#   "einops",
#   "numpy",
#   "scipy",
#   "soundfile",
#   "safetensors",
#   "huggingface_hub",
#   "pyyaml",
# ]
# ///
"""Golden outputs for the Swift port of NVIDIA NemotronLabs RE-USE, from NVIDIA's own model code.

    uv run reference/golden_nemotron.py   # writes Tests/EfficientNoiseReductionTests/Fixtures/nemotron_*.safetensors

The model files import the CUDA-only ``mamba_ssm``; the pure-PyTorch ``Mamba`` from
``golden.py`` stands in for it. ``whole`` reproduces the per-file loop of NVIDIA's
``inference.py``; ``chunked`` reproduces ``inference_chunk.py`` (overlapping chunks,
Hann cross-fade, end-aligned last chunk) line by line.

Whole-file fixtures also hold the network's compressed magnitude and phase and its
input features, so the network can be checked on its own.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.numpy import save_file
from safetensors.torch import load_file
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden import install_stubs  # noqa: E402  (pure-PyTorch mamba_ssm)

REPO = "nvidia/NVIDIA-NemotronLabs-RE-USE"
REVISION = "31f82f2efad1095034492f236e8dac9239ba713b"
OUT = Path(__file__).resolve().parent.parent / "Tests" / "EfficientNoiseReductionTests" / "Fixtures"
RELU = nn.ReLU()


def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def sizes(cfg, sr):
    s = cfg["stft_cfg"]
    base = s["sampling_rate"]
    return (
        make_even(s["n_fft"] * sr // base),
        make_even(s["hop_size"] * sr // base),
        make_even(s["win_size"] * sr // base),
    )


def enhance_once(model, cfg, wav, sr):
    """The body of NVIDIA's per-file / per-chunk inference: STFT, network, gating, iSTFT."""
    from models.stfts import mag_phase_istft, mag_phase_stft
    from utils.util import pad_or_trim_to_match

    n_fft, hop, win = sizes(cfg, sr)
    compress_factor = cfg["model_cfg"]["compress_factor"]
    noisy_mag, noisy_pha, _ = mag_phase_stft(wav, n_fft=n_fft, hop_size=hop, win_size=win,
                                             compress_factor=compress_factor, center=True, addeps=False)
    amp_g, pha_g, _ = model(noisy_mag, noisy_pha)
    raw_mag, raw_pha = amp_g.clone(), pha_g.clone()
    mag = torch.expm1(RELU(amp_g))
    zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
    amp_g[:, :, (zero_portion > 0.5)[0]] = 0
    audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, compress_factor)
    audio_g = pad_or_trim_to_match(wav.detach(), audio_g, pad_value=1e-8)
    return audio_g, raw_mag, raw_pha, noisy_mag, noisy_pha


def whole(model, cfg, wav, sr):
    """inference.py"""
    noisy_wav = torch.from_numpy(wav)[None]
    audio_g, mag, pha, noisy_mag, noisy_pha = enhance_once(model, cfg, noisy_wav, sr)
    assert audio_g.shape == noisy_wav.shape
    return audio_g[0].numpy(), mag[0].numpy(), pha[0].numpy(), noisy_mag[0].numpy(), noisy_pha[0].numpy()


def chunked(model, cfg, wav, sr, chunk_size_in_seconds, hop_length_portion):
    """inference_chunk.py, single channel."""
    Noisy_wav = torch.from_numpy(wav)[None]
    chunk_size = int(chunk_size_in_seconds * sr)
    hop_length = int(hop_length_portion * chunk_size)
    window = torch.clamp(torch.hann_window(chunk_size), min=1e-7)
    audio_enhanced = torch.zeros_like(Noisy_wav)
    window_sum = torch.zeros_like(Noisy_wav)
    num_chunks = max(1, math.ceil((Noisy_wav.shape[1] - chunk_size) / hop_length) + 1)
    c = 0
    noisy_wav = Noisy_wav[c:c + 1, :]
    for i in range(num_chunks):
        if i < num_chunks - 1:
            noisy_wav_chunk = noisy_wav[:, i * hop_length: i * hop_length + chunk_size]
        else:
            noisy_wav_chunk = noisy_wav[:, -chunk_size:]
        audio_g = enhance_once(model, cfg, noisy_wav_chunk, sr)[0]
        if i < num_chunks - 1:
            audio_enhanced[c:c + 1, i * hop_length:i * hop_length + chunk_size] += audio_g * window[0:audio_g.shape[1]]
            window_sum[c:c + 1, i * hop_length:i * hop_length + chunk_size] += window[0:audio_g.shape[1]]
        else:
            audio_enhanced[c:c + 1, -chunk_size:] += audio_g * window[0:audio_g.shape[1]]
            window_sum[c:c + 1, -chunk_size:] += window[0:audio_g.shape[1]]
    audio_enhanced = audio_enhanced / window_sum
    assert audio_enhanced.shape == Noisy_wav.shape
    return audio_enhanced[0].numpy(), chunk_size, hop_length, num_chunks


def main() -> None:
    root = Path(snapshot_download(REPO, revision=REVISION))
    sys.path.insert(0, str(root))
    install_stubs()
    from models.generator_SEMamba_time_d4 import SEMamba

    cfg = json.loads((root / "config.json").read_text())
    model = SEMamba(cfg).eval()
    model.load_state_dict(load_file(root / "model.safetensors"), strict=True)
    torch.set_num_threads(8)

    audio, sr = sf.read(root / "noisy_audio" / "mic_test2.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f"mic_test2.wav: {sr} Hz, {len(audio) / sr:.2f} s")
    g = math.gcd(sr, 48000)
    a48 = resample_poly(audio, 48000 // g, sr // g).astype(np.float32)
    g16 = math.gcd(sr, 16000)
    a16 = resample_poly(audio, 16000 // g16, sr // g16).astype(np.float32)
    g44 = math.gcd(sr, 44100)
    a44 = resample_poly(audio, 44100 // g44, sr // g44).astype(np.float32)

    OUT.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for name, wav, rate in [
            ("nemotron_whole_16k", a16[8000: 8000 + 3 * 16000 + 77], 16000),
            ("nemotron_whole_44k", a44[44100: 44100 + 44100 + 4410 + 13], 44100),
            ("nemotron_whole_48k", a48[48000: 48000 + 72000 + 33], 48000),
        ]:
            if (OUT / f"{name}.safetensors").exists():
                continue
            wav = np.ascontiguousarray(wav, dtype=np.float32)
            out, mag, pha, nmag, npha = whole(model, cfg, wav, rate)
            save_file({
                "input": wav, "output": out.astype(np.float32),
                "mag": np.ascontiguousarray(mag.T), "pha": np.ascontiguousarray(pha.T),
                "noisy_mag": np.ascontiguousarray(nmag.T), "noisy_pha": np.ascontiguousarray(npha.T),
                "config": np.array([rate, 0, 0], dtype=np.int32),
            }, OUT / f"{name}.safetensors")
            print(f"{name}: {len(wav)} samples, {mag.shape[1]} frames x {mag.shape[0]} bins, rms out {np.sqrt((out ** 2).mean()):.4f}")

        long16 = a16[: 7 * 16000 + 3 * 1600 + 11]
        long48 = a48[24000: 24000 + 2 * 48000 + 33333]
        for name, wav, rate, seconds, portion in [
            ("nemotron_chunk_16k", long16, 16000, 2.0, 0.5),       # several chunks + end-aligned last one
            ("nemotron_chunk_48k", long48, 48000, 1.0, 0.5),
            ("nemotron_chunk_16k_short", a16[: 16000 + 4321], 16000, 2.0, 0.5),  # shorter than one chunk
        ]:
            if (OUT / f"{name}.safetensors").exists():
                continue
            wav = np.ascontiguousarray(wav, dtype=np.float32)
            out, chunk, hop, n = chunked(model, cfg, wav, rate, seconds, portion)
            save_file({
                "input": wav, "output": out.astype(np.float32),
                "config": np.array([rate, chunk, hop], dtype=np.int32),
            }, OUT / f"{name}.safetensors")
            print(f"{name}: {len(wav)} samples, chunk {chunk}, hop {hop}, {n} chunks, rms out {np.sqrt((out ** 2).mean()):.4f}")


if __name__ == "__main__":
    main()
