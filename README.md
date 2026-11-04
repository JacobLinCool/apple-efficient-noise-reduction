# Efficient Noise Reduction for Apple Silicon

Two streaming speech enhancement models in Swift for Apple silicon, behind one streaming interface:

- **Real-time RE-USE** (NVIDIA): universal speech enhancement for noise, reverberation, bandwidth limitation, codec artifacts and other degradations, at any sample rate. Runs on the GPU: convolutions on MPSGraph, Mamba blocks on MLX with custom Metal kernels. Latency and compute are chosen at run time, from frame-by-frame real time to whole-file offline.
- **DeepFilterNet3**: 48 kHz noise suppression with a fixed 40 ms latency. Runs on the CPU with Accelerate, never starts the GPU, and its weights ship with the package.

However a stream is chunked, both produce the same output as processing it in one piece, and both match their official PyTorch implementations numerically.

- RE-USE papers: [One Model, Many Latencies](https://arxiv.org/abs/2606.25621), [Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement](https://arxiv.org/abs/2603.02641); weights [`nvidia/Real-time_RE-USE`](https://huggingface.co/nvidia/Real-time_RE-USE), pinned to revision `4c0c186`
- DeepFilterNet3: [DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement](https://arxiv.org/abs/2305.08227); weights from the official [`deepfilternet`](https://github.com/Rikorose/DeepFilterNet) 0.5.6

## Installation

Add the dependency with Swift Package Manager:

```swift
.package(url: "https://github.com/JacobLinCool/apple-efficient-noise-reduction", from: "0.1.0"),
// in your target
.product(name: "EfficientNoiseReduction", package: "apple-efficient-noise-reduction"),
```

Requires macOS 14 / iOS 17 or later, built with Xcode (or `xcodebuild`): MLX's Metal shaders only compile through Xcode's build system, so a `swift build` binary cannot find them when it runs RE-USE. DeepFilterNet3 does not use MLX and also works with `swift build`. The library compiles for iOS but has not been tested on a device.

## Quick start

Try it with the command-line tool (inside the repository):

```bash
make build                                   # .build/xcode/Build/Products/Release/noise-reduction
CLI=.build/xcode/Build/Products/Release/noise-reduction
$CLI download                                # RE-USE weights to ~/Library/Application Support/EfficientNoiseReduction/
$CLI enhance noisy.wav clean.wav             # default: exit layer 8, no look-ahead, 0.5 s latency
$CLI enhance noisy.wav clean.wav --offline -e 12 -l 2   # best-quality offline setting
$CLI live --monitor --latency 0.1            # enhance the microphone live and play it back (use headphones)
$CLI bench --sample-rate 16000               # measure the settings on this machine
$CLI enhance noisy.wav clean.wav --engine dfn3          # DeepFilterNet3 (input resampled to 48 kHz)
$CLI enhance-batch noisy/ enhanced/ --engine dfn3       # a whole directory, model loaded once
```

RE-USE also uses weights already in the Hugging Face cache, or the directory in `REUSE_MODEL_DIR`. DeepFilterNet3 needs no download.

## Choosing a latency

Three independent settings decide the latency:

| Setting | Range | Effect |
|---|---|---|
| `exitLayer` | 3–12 | How many Mamba blocks run, with the decoder trained for that depth. Deeper sounds better and costs more |
| `lookAhead` | 0–2 | Each future 20 ms frame waited for adds 20 ms of algorithmic latency; helps ASR accuracy most |
| `chunk` | 1 frame to offline | Frames per network call. Changes only when output appears and the throughput, never the output itself |

The algorithmic latency is the 40 ms window plus the look-ahead and the chunk buffering. `.latency(0.5)` picks the largest chunk that fits that budget; the compute time of a chunk comes on top, see below.

Measured on an M5 Pro with the default precision (on AC power, normal power mode, background load around 3; `noise-reduction bench`, 20 s of audio per setting, pushed in 10 ms blocks). RTF is compute time divided by audio duration; lower is better.

| Sample rate | Exit layer | Frames per call | Algorithmic latency | Compute per call (mean / p95) | RTF |
|---|---|---|---|---|---|
| 16 kHz | 3 | 1 | 40 ms | 2.9 / 3.2 ms | 0.146 |
| 16 kHz | 8 | 1 | 40 ms | 3.7 / 3.9 ms | 0.186 |
| 16 kHz | 12 | 1 | 40 ms | 4.4 / 4.6 ms | 0.220 |
| 16 kHz | 8 | 24 | 500 ms | 7.8 / 9.2 ms | 0.026 |
| 16 kHz | 12 | 24 | 500 ms | 9.0 / 9.2 ms | 0.028 |
| 16 kHz | 12 | offline (100) | — | 29.4 / 40.0 ms | 0.024 |
| 48 kHz | 8 | 1 | 40 ms | 5.2 / 5.5 ms | 0.259 |
| 48 kHz | 12 | 1 | 40 ms | 6.4 / 6.7 ms | 0.322 |
| 48 kHz | 8 | 24 | 500 ms | 17.6 / 18.7 ms | 0.046 |
| 48 kHz | 12 | 24 | 500 ms | 24.4 / 25.6 ms | 0.061 |
| 48 kHz | 12 | offline (33) | — | 30.5 / 31.5 ms | 0.056 |

Frame by frame, every step finishes well within its 20 ms hop, so the 40 ms setting runs in real time; the cost there is mostly the fixed overhead of kernel dispatch and GPU synchronisation, hence the higher RTF. Chunks of about 0.5 s amortise that overhead: about 35–40× real time at 16 kHz and 15–20× at 48 kHz.

Memory and energy were measured on about 65 s of speech (`noise-reduction enhance` prints the memory; power comes from the SoC's IOReport energy counters, as the extra power over idle). "Real time" feeds the audio at its real-time pace, so the GPU idles between chunks.

| Setting (exit layer 8) | Peak process footprint | Extra GPU power | Energy per minute of audio |
|---|---|---|---|
| 16 kHz, real time, 0.5 s | 0.54 GB | 0.13 W | — |
| 16 kHz, real time, 40 ms frame by frame | 0.27 GB | 0.39 W | — |
| 16 kHz, offline | 0.99 GB | 17 W (full speed, 31 W for the SoC) | about 38 J |
| 48 kHz, real time, 0.5 s | 1.2 GB | 0.76 W | — |
| 48 kHz, offline | 1.2 GB | 19 W (full speed, 29 W for the SoC) | about 79 J |

In real time the extra power is small enough that the SoC total drowns in background fluctuations, so only the GPU share is listed; offline runs at full speed for a short time, and its total is reliable. Every setting stays within 1.5 GB of memory, exit layer 12 included.

## RE-USE vs DeepFilterNet3

Quality on the VoiceBank-DEMAND test set (824 utterances, 16 kHz; `reference/quality_vbd.py`). Neither model was trained on it. DeepFilterNet3's input is upsampled to 48 kHz and its output downsampled back to 16 kHz for scoring.

| System | Algorithmic latency | PESQ | ESTOI | SI-SDR | DNSMOS OVRL | SIG | BAK |
|---|---|---|---|---|---|---|---|
| Unprocessed | — | 1.97 | 0.787 | 8.5 | 2.70 | 3.35 | 3.13 |
| DeepFilterNet3 | 40 ms | 2.69 | 0.843 | 17.7 | 3.22 | 3.46 | 4.11 |
| RE-USE, exit 3 | 40 ms | 2.59 | 0.848 | 18.1 | 3.18 | 3.45 | 4.06 |
| RE-USE, exit 8 | 40 ms | 2.74 | 0.859 | 18.6 | 3.24 | 3.49 | 4.11 |
| RE-USE, exit 8, look-ahead 1 | 60 ms | 2.80 | 0.862 | 18.7 | 3.26 | 3.50 | 4.12 |
| RE-USE, exit 12, look-ahead 2 | 80 ms | 2.83 | 0.862 | 18.4 | 3.27 | 3.51 | 4.13 |

These are within 0.03 PESQ of the corresponding values in Table II of the RE-USE paper (unprocessed 1.97, DeepFilterNet3 2.71, exit 8 at 2.76 / 2.82), which is further evidence that both ports are correct. For noise alone the two are close: at the same 40 ms, RE-USE exit 8 is slightly better and exit 3 slightly worse. What RE-USE adds is quality in exchange for latency, and the degradations DeepFilterNet3 does not address: reverberation, bandwidth limitation, codec artifacts.

Efficiency differs by more than an order of magnitude. 48 kHz, about 65 s of speech, same conditions as above:

| Setting | RTF | Peak process footprint | Notes |
|---|---|---|---|
| DeepFilterNet3, real time, frame by frame (10 ms chunks) | 0.11 | 56 MB | CPU only |
| DeepFilterNet3, offline | 0.020 | 78 MB | about 10 W extra for the SoC at full speed, about 13 J per minute of audio |
| RE-USE exit 3, real time, 0.5 s | 0.076 | 0.66 GB | |
| RE-USE exit 8, real time, frame by frame (20 ms chunks) | 0.72 | 0.46 GB | |
| RE-USE exit 8, real time, 0.5 s | 0.092 | 1.2 GB | |
| RE-USE exit 8, offline | 0.039 | 1.2 GB | about 29 W extra for the SoC at full speed, about 79 J per minute of audio |

DeepFilterNet3 does about 3.4M MACs per 10 ms; RE-USE exit 8 at 48 kHz about 1.5G MACs per 20 ms, more than two orders of magnitude apart, and memory and energy follow. For always-on background use or plain noise, DeepFilterNet3 costs almost nothing; reach for RE-USE for the other degradations, or to trade latency for quality.

## Usage

```swift
import EfficientNoiseReduction

let model = try ReUseModel(directory: ModelLocator.locate())

// Whole recording
let clean = try model.enhance(samples, sampleRate: 16000, exitLayer: 12, lookAhead: 2)

// Streaming: blocks of any size in, final samples out; finish() returns the tail
let stream = try StreamingEnhancer(
    model: model, sampleRate: 48000,
    configuration: .init(exitLayer: 8, lookAhead: 1, chunk: .latency(0.5)))
stream.warmUp()
let out = stream.process(block)
let tail = stream.finish()

// DeepFilterNet3: the same streaming interface, 48 kHz, chunkFrames in 10 ms hops
let dfn3 = DeepFilterNet3Enhancer(model: try DeepFilterNet3(), chunkFrames: 1)

// Audio callbacks: any SpeechEnhancer; write and read only copy samples, the network runs on its own thread
let rt = RealtimeEnhancer(dfn3)
rt.write(captured)        // capture callback
rt.read(into: playback)   // playback callback
rt.stop()
```

`StreamingEnhancer` and `DeepFilterNet3Enhancer` both implement `SpeechEnhancer` and are not thread-safe: use a stream from one thread. Models (`ReUseModel`, `DeepFilterNet3`) can be shared by many streams; RE-USE loads each (exit layer, look-ahead) network once.

## Correctness

`reference/golden.py` runs NVIDIA's original PyTorch model code, with only the CUDA-only `mamba_ssm` replaced by a pure-PyTorch reference scan, and writes golden data to `Tests/EfficientNoiseReductionTests/Fixtures/` (16 kHz and 44.1 kHz, several exit layers and look-aheads). The same script checks that NVIDIA's frame-by-frame online path matches its offline path, which is what lets this implementation chunk freely.

| Test | Result |
|---|---|
| Network output, fed NVIDIA's STFT features (`.exact`) | magnitude max error about 2e-6, weighted phase error about 1e-6 rad |
| Whole-pipeline waveform vs NVIDIA (including a full 6.5 s recording) | 39–47 dB SNR |
| Frame-by-frame, 7-frame and 0.5 s chunks vs one piece | 119 dB or more (MPSGraph picks convolution algorithms per shape, so not bit-identical) |
| Real-time pipeline (writes and reads at real-time pace) | the one-piece output behind a constant delay, no underruns |

The network itself matches the reference; what remains in the waveform comes from the input: the phase of near-silent bins differs at rounding level between FFT implementations, and the network output is sensitive to it.

The default precision is `.fast`: convolutions and Mamba run in fp16, with the SSM state and the residual stream in fp32. Its output is 55–57 dB SNR from `.exact`; on the full 6.5 s test recording the two are 40.3 and 40.8 dB from NVIDIA, closer than two FFT implementations agree. Use `.exact` (`--exact` in the CLI) to compare with the reference; everything then runs in full fp32.

For DeepFilterNet3, `reference/golden_dfn3.py` produces golden data and the weights file from the official `deepfilternet` package:

| Test | Result |
|---|---|
| STFT vs libDF | max error 2.5e-10 |
| Network output (ERB gains, deep-filter coefficients), fed the official features | max error about 3e-6 |
| Whole-pipeline waveform vs `df.enhance.enhance` | 125–135 dB SNR |
| Frame-by-frame, 7-frame and irregular push sizes vs one piece | 136 dB or more |

The golden data contains third-party audio and NVIDIA model outputs, so it is not in the repository; generate it locally:

```bash
make golden   # generate golden data (needs uv; downloads the RE-USE weights and the deepfilternet package)
make test     # tests without golden data or RE-USE weights are skipped
```

## Implementation notes

- **Convolutions on MPSGraph, Mamba on MLX**: the encoder and both decoders (dense blocks, sub-pixel up-sampling, channel LayerNorm, PReLU) are MPSGraph executables, the same convolution stack Core ML uses on the GPU; fp16 for `.fast`, fp32 for `.exact`. Each conv's streaming cache stays on the GPU, and executables are compiled and cached per (bins, chunk frames). The Mamba blocks are one compiled MLX graph. Each chunk runs encoder → Mamba → decoder, with one synchronisation per stage.
- **Mamba in fp16 for `.fast`**: GEMMs, conv1d and scan inputs and outputs are fp16; the SSM state, the arithmetic inside the scan and the residual stream between blocks stay fp32. Output is 55–57 dB SNR from `.exact`, and the VoiceBank-DEMAND scores are unchanged. The LayerNorm-plus-residual after each block's output projection is one fused Metal kernel, since MLX's LayerNorm is slow on 64-channel rows.
- **Custom Metal kernels for Mamba**: conv1d and SiLU form one stencil kernel; the selective scan fuses dt_proj, softplus, the D skip and the gate, with one thread per channel holding its 16 states in registers and broadcast loads of B and C. The forward and backward frequency Mambas run as one batch, and `out_proj` and `output_proj` are folded into one matrix at load time.
- **Everything stays in the native (time, freq, channel) layout**: scanning along time or frequency differs only in stride, so nothing is transposed.
- **STFT / iSTFT on the CPU**: each frame's result does not depend on the batch size, so every chunking gives the same output; overlap-add uses vDSP.
- **DeepFilterNet3 entirely on the CPU**: the model is small (about 3.4M MACs per 10 ms) and steps a GRU frame by frame, so GPU dispatch would cost more than the arithmetic. Convolutions are im2col plus BLAS (Accelerate uses the AMX units), depthwise convolutions are plain loops, and BatchNorm is folded at load time. The weights are read by a small safetensors parser, not through MLX, so Metal is never initialised.

### Why not the Neural Engine

`reference/ane_experiment.py` converts RE-USE's convolutional part (encoder and both decoders, about 40% of the MACs) to an fp16 Core ML model and times it on each compute unit, with power from the SoC energy counters (M5 Pro, Low Power Mode, 16 kHz, 24 frames per call):

| Core ML compute unit | Per call | Energy per call |
|---|---|---|
| Neural Engine | 7.5 ms | 26 mJ |
| GPU | 2.0 ms | 22 mJ |
| CPU | 11.5 ms | 88 mJ |

The Neural Engine is more than three times slower than the GPU and uses no less energy, and fp16 adds about 1.5% error to the magnitude output. Mamba's selective scan cannot run on the Neural Engine at all, so a hybrid would also hand data between units twice per chunk. RE-USE therefore stays entirely on the GPU.

The experiment also showed Core ML's GPU convolutions running several times faster than the MLX implementation of the time, which is why the convolutions moved to MPSGraph. Encoder plus decoders under the same conditions (16 kHz, exit layer 8, Low Power Mode):

| Implementation | 1 frame per call | 24 frames per call | 96 frames per call |
|---|---|---|---|
| MLX, convolutions as GEMMs (fp32 / TF32) | — | about 10 ms | about 27 ms |
| MLX, fp16 with a fused LayerNorm Metal kernel | 1.3 ms | 3.5 ms | 19 ms |
| Core ML, GPU, fp16 (reference) | 1.6 ms | 2.0 ms | 6.5 ms |
| MPSGraph, fp16 (current) | about 1.9 ms | about 3 ms | about 10–12 ms |

The convolutions alone are as fast on MPSGraph as on Core ML: the whole stack as a single graph takes 1.75 ms for 24 frames. In the pipeline, data is handed to and from MLX's Mamba blocks and the encoder and decoders run as two separate executions, so small chunks cost a little more. Most of each call is now spent in the Mamba blocks.

## License

The code is MIT licensed.

- The RE-USE weights are released by NVIDIA under the [NVIDIA One-Way Noncommercial License (NSCLv1)](https://github.com/NVlabs/HMAR/blob/main/LICENSE), for noncommercial research and development only. This repository does not include them; `download` fetches them on request.
- The DeepFilterNet3 weights are dual licensed MIT / Apache-2.0 and ship as safetensors in `Sources/EfficientNoiseReduction/Resources/`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the license text.
