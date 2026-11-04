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
"""Golden outputs for the Swift port, from NVIDIA's own Real-time RE-USE model code.

NVIDIA's model files import the CUDA-only ``mamba_ssm``. A pure-PyTorch ``Mamba`` that
follows mamba_ssm's reference path (``selective_scan_ref`` for ``forward``, the
``step`` method for one-frame decoding) stands in for it, so the rest of the reference
code runs unmodified on the CPU.

    uv run reference/golden.py            # writes Tests/EfficientNoiseReductionTests/Fixtures/*.safetensors

Each fixture holds the input waveform, the waveform NVIDIA's ``offline_inference.py``
produces (silent-frame gating, iSTFT, epsilon padding), and the network's compressed
magnitude and phase for one (sampling rate, exit layer, look-ahead) setting. The script
also runs NVIDIA's frame-by-frame ``online_inference.py`` path on one setting and checks
it matches the offline path, which is the property the Swift chunked streaming relies on.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.numpy import save_file
from safetensors.torch import load_file
from scipy.signal import resample_poly

REPO = "nvidia/Real-time_RE-USE"
REVISION = "4c0c186272ca853cd18d832fc5c28777b62f32d1"
OUT = Path(__file__).resolve().parent.parent / "Tests" / "EfficientNoiseReductionTests" / "Fixtures"


class Mamba(nn.Module):
    """mamba_ssm.Mamba (v1) with the reference scan; same parameter names."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **_):
        super().__init__()
        self.d_inner, self.d_state, self.d_conv = expand * d_model, d_state, d_conv
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        L = u.shape[1]
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :L]).transpose(1, 2)
        dt, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        h = torch.zeros(u.shape[0], self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            h = torch.exp(delta[:, t, :, None] * A) * h + delta[:, t, :, None] * B[:, t, None, :] * x[:, t, :, None]
            ys.append((h * C[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1) + x * self.D
        return self.out_proj(y * F.silu(z))

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **_):
        return (
            torch.zeros(batch_size, self.d_inner, self.d_conv),
            torch.zeros(batch_size, self.d_inner, self.d_state),
        )

    def step(self, hidden_states, conv_state, ssm_state):
        x, z = self.in_proj(hidden_states.squeeze(1)).chunk(2, dim=-1)
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x
        x = F.silu((conv_state * self.conv1d.weight[:, 0]).sum(-1) + self.conv1d.bias)
        dt, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        ssm_state = ssm_state * torch.exp(dt[..., None] * A) + x[..., None] * (dt[..., None] * B[:, None, :])
        y = (ssm_state * C[:, None, :]).sum(-1) + self.D * x
        return self.out_proj(y * F.silu(z)).unsqueeze(1), conv_state, ssm_state


def install_stubs() -> None:
    mamba_ssm = types.ModuleType("mamba_ssm")
    mamba_ssm.Mamba = Mamba  # type: ignore[attr-defined]
    sys.modules["mamba_ssm"] = mamba_ssm
    # online_inference.py imports these at module level; only its classes are used here.
    for name in ("torchaudio", "torchaudio.transforms", "librosa"):
        sys.modules.setdefault(name, types.ModuleType(name))


def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def stft_sizes(cfg, sr):
    s = cfg["stft_cfg"]
    base = s["sampling_rate"]
    return tuple(make_even(s[k] * sr // base) for k in ("n_fft", "hop_size", "win_size"))


def offline(model, cfg, wav, sr, exit_layer, look_ahead):
    from models.stfts import mag_phase_istft, mag_phase_stft

    n_fft, hop, win = stft_sizes(cfg, sr)
    noisy = torch.from_numpy(wav)[None]
    mag, pha, _ = mag_phase_stft(noisy, n_fft, hop, win, cfg["model_cfg"]["compress_factor"], center=True)
    amp_g, pha_g, _ = model(mag, pha, exit_layer, look_ahead)
    raw_mag, raw_pha = amp_g.clone(), pha_g.clone()
    lin = torch.expm1(torch.relu(amp_g))
    zero_portion = torch.sum(lin == 0, 1) / lin.shape[1]
    amp_g[:, :, (zero_portion > 0.5)[0]] = 0
    audio = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, cfg["model_cfg"]["compress_factor"])
    out = torch.full_like(noisy, 1e-8)
    n = min(audio.shape[1], noisy.shape[1])
    out[:, :n] = audio[:, :n]
    return out[0].numpy(), raw_mag[0].numpy(), raw_pha[0].numpy(), mag[0].numpy(), pha[0].numpy()


def online_matches_offline(model, cfg, wav, sr, exit_layer, look_ahead) -> float:
    """NVIDIA's one-frame-in, one-frame-out path vs the offline network output."""
    from models.stfts import mag_phase_stft
    from online_inference import StreamingSEMamba

    n_fft, hop, win = stft_sizes(cfg, sr)
    mag, pha, _ = mag_phase_stft(torch.from_numpy(wav)[None], n_fft, hop, win, "relu_log1p", center=True)
    ref_mag, ref_pha, _ = model(mag, pha, exit_layer, look_ahead)
    stream = StreamingSEMamba(model).eval()
    stream.reset(mag.shape[1] // 2 + 1)
    pad = torch.zeros(1, mag.shape[1], look_ahead)
    mag, pha = torch.cat([mag, pad], -1), torch.cat([pha, pad], -1)
    outs = [stream(mag[:, :, : look_ahead + 1], pha[:, :, : look_ahead + 1], exit_layer, look_ahead)[0]]
    for t in range(look_ahead + 1, mag.shape[-1]):
        outs.append(stream(mag[:, :, t : t + 1], pha[:, :, t : t + 1], exit_layer, look_ahead)[0])
    online = torch.stack(outs, -1)[0]
    return float((online - ref_mag[0]).abs().max())


def main() -> None:
    root = Path(snapshot_download(REPO, revision=REVISION))
    sys.path.insert(0, str(root))
    install_stubs()
    import json

    from models.streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv import SEMamba_decoder_list

    cfg = json.loads((root / "config.json").read_text())
    model = SEMamba_decoder_list(cfg).eval()
    model.load_state_dict(load_file(root / "model.safetensors"), strict=True)
    torch.set_num_threads(8)

    audio, sr = sf.read(root / "noisy_audio" / "mic_test.wav", dtype="float32")
    assert sr == 44100
    wav16 = resample_poly(audio, 160, 441).astype(np.float32)

    OUT.mkdir(parents=True, exist_ok=True)
    cases = [
        # (name, waveform, sr, exit layer, look-ahead)
        ("sr16k_exit8_la0", wav16[16000 : 16000 + 3 * 16000 + 123], 16000, 8, 0),
        ("sr16k_exit12_la2", wav16[16000 : 16000 + 3 * 16000 + 123], 16000, 12, 2),
        ("sr16k_exit3_la1", wav16[16000 : 16000 + 3 * 16000 + 123], 16000, 3, 1),
        ("sr44k_exit6_la1", audio[2 * 44100 : 4 * 44100 + 77], 44100, 6, 1),
        # The whole recording, with its loudest frames, at the default setting.
        ("sr44k_full_exit8_la0", audio, 44100, 8, 0),
    ]
    with torch.no_grad():
        for name, wav, rate, layer, ahead in cases:
            wav = np.ascontiguousarray(wav, dtype=np.float32)
            enhanced, mag, pha, noisy_mag, noisy_pha = offline(model, cfg, wav, rate, layer, ahead)
            save_file(
                {
                    "input": wav,
                    "output": enhanced.astype(np.float32),
                    "mag": np.ascontiguousarray(mag.T),  # (T, F) compressed magnitude
                    "pha": np.ascontiguousarray(pha.T),
                    "noisy_mag": np.ascontiguousarray(noisy_mag.T),
                    "noisy_pha": np.ascontiguousarray(noisy_pha.T),
                    "config": np.array([rate, layer, ahead], dtype=np.int32),
                },
                OUT / f"{name}.safetensors",
            )
            print(f"{name}: {len(wav)} samples, frames {mag.shape[1]}, rms out {np.sqrt((enhanced**2).mean()):.4f}")

        diff = online_matches_offline(model, cfg, wav16[16000:24000], 16000, 5, 2)
        print(f"NVIDIA online vs offline network magnitude, max |diff| = {diff:.2e}")


if __name__ == "__main__":
    main()
