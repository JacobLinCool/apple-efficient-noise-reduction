# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "deepfilternet==0.5.6",
#   "torch==2.3.1",
#   "torchaudio==2.3.1",
#   "numpy<2",
#   "scipy",
#   "soundfile",
#   "safetensors",
#   "huggingface_hub",
# ]
# ///
"""DeepFilterNet3 weights and golden outputs for the Swift port, from the official package.

    uv run reference/golden_dfn3.py

Writes
- `Sources/EfficientNoiseReduction/Resources/DeepFilterNet3.safetensors`: the checkpoint's state dict
  (float32, PyTorch names and layouts) plus the ERB band widths. DeepFilterNet is
  MIT / Apache-2.0, so the weights ship with the package.
- `Tests/EfficientNoiseReductionTests/Fixtures/dfn3_*.safetensors`: input audio, the output of
  `df.enhance.enhance` (whole utterance, `pad=True`), and the intermediate STFT,
  features, ERB mask and deep-filter coefficients.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from df.enhance import enhance, init_df
from df.enhance import df_features
from huggingface_hub import hf_hub_download
from safetensors.numpy import save_file
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "Tests" / "EfficientNoiseReductionTests" / "Fixtures"
WEIGHTS = ROOT / "Sources" / "EfficientNoiseReduction" / "Resources" / "DeepFilterNet3.safetensors"


def main() -> None:
    model, df_state, _ = init_df(log_level="ERROR")  # downloads DeepFilterNet3
    model.eval()
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().float().numpy().copy() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    state["erb_widths"] = np.asarray(df_state.erb_widths(), dtype=np.int32)
    save_file(state, WEIGHTS)
    print(f"weights: {len(state)} tensors, {sum(v.size for v in state.values())} values -> {WEIGHTS}")

    wav = hf_hub_download(
        "nvidia/Real-time_RE-USE", "noisy_audio/mic_test.wav", revision="4c0c186272ca853cd18d832fc5c28777b62f32d1"
    )
    audio, sr = sf.read(wav, dtype="float32")
    audio48 = resample_poly(audio, 160, 147).astype(np.float32)
    rng = np.random.default_rng(0)
    cases = {
        "dfn3_speech": audio48[48000 : 48000 + 3 * 48000 + 321],
        # Same speech under extra white noise, to exercise the deep-filter path.
        "dfn3_noisy": audio48[2 * 48000 : 4 * 48000 + 17]
        + 0.02 * rng.standard_normal(2 * 48000 + 17).astype(np.float32),
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for name, x in cases.items():
            x = np.ascontiguousarray(x, dtype=np.float32)
            out = enhance(model, df_state, torch.from_numpy(x)[None]).numpy()[0]
            # Intermediates of the same call (enhance pads n_fft zeros at the end).
            padded = torch.nn.functional.pad(torch.from_numpy(x)[None], (0, df_state.fft_size()))
            spec, erb_feat, spec_feat = df_features(padded, df_state, 96)
            spec_e, m, lsnr, coefs = model(spec.clone(), erb_feat, spec_feat)
            save_file(
                {
                    "input": x,
                    "output": out.astype(np.float32),
                    "spec": spec[0, 0].numpy(),  # (T, 481, 2)
                    "erb_feat": erb_feat[0, 0].numpy(),  # (T, 32)
                    "spec_feat": spec_feat[0, 0].numpy(),  # (T, 96, 2)
                    "mask": m[0, 0].numpy(),  # (T, 32)
                    "coefs": coefs[0].permute(1, 2, 0, 3).contiguous().numpy(),  # (T, 96, 5, 2)
                    "spec_e": spec_e[0, 0].numpy(),  # (T, 481, 2)
                },
                FIXTURES / f"{name}.safetensors",
            )
            print(f"{name}: {len(x)} samples, {spec.shape[2]} frames, rms in {np.sqrt((x**2).mean()):.4f} out {np.sqrt((out**2).mean()):.4f}")


if __name__ == "__main__":
    main()
