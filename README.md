# Efficient Noise Reduction for Apple Silicon

Streaming speech enhancement for Apple silicon in Swift. You can use it as a menu bar app that gives meeting apps a cleaned-up microphone, or as a library for your own audio pipeline. Two models sit behind one streaming interface: one that removes noise at almost no cost, and one that repairs most kinds of degraded speech at a latency you choose.

| Model | What it fixes | Runs on | Latency | Weights |
|---|---|---|---|---|
| [**DeepFilterNet3**](https://arxiv.org/abs/2305.08227) | Background noise, 48 kHz | CPU (Accelerate) | 40 ms | Bundled (MIT / Apache-2.0) |
| [**Real-time RE-USE**](https://arxiv.org/abs/2606.25621) (NVIDIA) | Noise, reverberation, bandwidth limits, codec artifacts; any sample rate | GPU (MPSGraph + MLX) | 40 ms to offline, chosen at run time | Downloaded on request ([NVIDIA noncommercial license](https://huggingface.co/nvidia/Real-time_RE-USE)) |

However a stream is chunked, its output is the same as processing it in one piece, and both models match their official PyTorch implementations numerically ([Correctness](#correctness)).

**Want a cleaner microphone in meetings?** Go to [Noise Reduction for meetings](#noise-reduction-for-meetings). **Building on the library?** Go to [Library](#library).

## Noise Reduction for meetings

**Noise Reduction** is a menu bar app that adds **Noise Reduction Microphone** to macOS. Select it in Zoom, Teams, Meet or any other app, and that app hears your real microphone with the noise removed.

- The app runs only while some app is recording from the virtual microphone. Between meetings it does no audio work, and the microphone indicator stays off.
- The panel shows the signal path (microphone → model → meetings) with live levels at both ends. You can choose the input, the model and, for RE-USE, its quality and latency.
- Keep the default model, DeepFilterNet3, for everyday noise. Switch to RE-USE for echoey rooms or poor-quality microphones. Its weights are downloaded on first use, it runs on the GPU and needs about 0.5 GB of memory, and NVIDIA licenses it for **noncommercial use only**.
- Keep the app running during calls; **⋯ › Open at Login** starts it with your Mac. When the app is off or quit, the virtual microphone passes silence. It never falls back to the raw microphone.

**Install**

1. Download `NoiseReduction-<version>.zip` from [Releases](https://github.com/JacobLinCool/apple-efficient-noise-reduction/releases), unzip it, and move **Noise Reduction** to Applications. It is signed and notarized, and needs macOS 14 or later on Apple silicon.
2. Open it, click the waveform icon in the menu bar, and choose **Install…**. This asks for an administrator password, adds the audio driver to `/Library/Audio/Plug-Ins/HAL/`, and restarts Core Audio, which interrupts sound for about a second.
3. In your meeting app, select **Noise Reduction Microphone**.

To remove the driver, use **⋯ › Remove Virtual Microphone…**. The driver can also be installed from a terminal:

```bash
sudo cp -R "/Applications/Noise Reduction.app/Contents/Resources/EfficientNoiseReduction.driver" /Library/Audio/Plug-Ins/HAL/
sudo killall coreaudiod
```

End-to-end delay at 48 kHz, from the real microphone to the virtual one (M5 Pro):

| Model in the app | Delay | Output vs offline processing of the same input |
|---|---|---|
| DeepFilterNet3 | about 73 ms | correlation 0.999 |
| RE-USE, Balanced (exit layer 8), 40 ms | about 95 ms | correlation 0.986 |

The delay is made up of the 40 ms algorithmic latency, a compute margin and the device buffers. For RE-USE, the offline reference was given the same leading silence, because its output shifts slightly with where the 20 ms frame grid falls. A two-minute live run had no underruns or overruns.

## Choosing a model

Quality on VoiceBank-DEMAND (824 utterances, 16 kHz; `reference/quality_vbd.py`). Neither model was trained on it. DeepFilterNet3's input is upsampled to 48 kHz, and its output downsampled back to 16 kHz for scoring. "Exit" is RE-USE's depth setting ([below](#re-use-latency-and-speed)).

| System | Latency | PESQ | ESTOI | SI-SDR | DNSMOS OVRL | SIG | BAK |
|---|---|---|---|---|---|---|---|
| Unprocessed | — | 1.97 | 0.787 | 8.5 | 2.70 | 3.35 | 3.13 |
| DeepFilterNet3 | 40 ms | 2.69 | 0.843 | 17.7 | 3.22 | 3.46 | 4.11 |
| RE-USE, exit 3 | 40 ms | 2.59 | 0.848 | 18.1 | 3.18 | 3.45 | 4.06 |
| RE-USE, exit 8 | 40 ms | 2.74 | 0.859 | 18.6 | 3.24 | 3.49 | 4.11 |
| RE-USE, exit 8, look-ahead 1 | 60 ms | 2.80 | 0.862 | 18.7 | 3.26 | 3.50 | 4.12 |
| RE-USE, exit 12, look-ahead 2 | 80 ms | 2.83 | 0.862 | 18.4 | 3.27 | 3.51 | 4.13 |

Cost at 48 kHz on about 65 s of speech (M5 Pro). RTF is compute time divided by audio duration. Energy is the SoC's extra energy over idle, read from its IOReport counters:

| Setting | RTF | Peak memory | Energy |
|---|---|---|---|
| DeepFilterNet3, real time, 10 ms chunks | 0.11 | 56 MB | not reliably measurable (runs varied in sign) |
| DeepFilterNet3, offline | 0.020 | 78 MB | about 13 J per minute of audio |
| RE-USE exit 3, real time, 0.5 s | 0.076 | 0.66 GB | |
| RE-USE exit 8, real time, 20 ms chunks | 0.72 | 0.46 GB | |
| RE-USE exit 8, real time, 0.5 s | 0.092 | 1.2 GB | GPU only: 0.76 W extra |
| RE-USE exit 8, offline | 0.039 | 1.2 GB | about 79 J per minute of audio |

On plain noise the two models are close at the same latency. Where the RE-USE paper reports the same settings, these scores are within 0.03 PESQ of its figures. DeepFilterNet3 needs more than two orders of magnitude less compute, and a fraction of the memory and energy, so it is the default for always-on use. RE-USE is worth its cost for reverberant, band-limited or codec-damaged speech, or when you can trade latency for quality.

## RE-USE latency and speed

Three settings decide RE-USE's latency:

| Setting | Range | Effect |
|---|---|---|
| `exitLayer` | 3–12 | Mamba blocks to run, each depth with its own trained decoder. Deeper sounds better and costs more |
| `lookAhead` | 0–2 | Future 20 ms frames to wait for; each adds 20 ms. Helps ASR most |
| `chunk` | 1 frame to offline | Frames per network call. Changes when output appears and the throughput, never the output itself |

The algorithmic latency is the 40 ms window, plus the look-ahead, plus the chunk buffering. Compute time per call comes on top. Measured with `noise-reduction bench` on an M5 Pro (AC power, normal power mode, background load around 3), with 20 s of audio per setting pushed in 10 ms blocks:

| Sample rate | Exit layer | Frames per call | Latency | Compute per call (mean / p95) | RTF |
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

Frame by frame, each call finishes well within its 20 ms hop. Most of that time is fixed dispatch and synchronisation overhead, which chunks of about 0.5 s amortise.

Memory and power at exit layer 8. In real time, the GPU idles between chunks:

| Setting | Peak memory | Extra GPU power | Energy per minute of audio |
|---|---|---|---|
| 16 kHz, real time, 40 ms | 0.27 GB | 0.39 W | — |
| 16 kHz, real time, 0.5 s | 0.54 GB | 0.13 W | — |
| 16 kHz, offline | 0.99 GB | 17 W (31 W for the SoC) | about 38 J |
| 48 kHz, real time, 0.5 s | 1.2 GB | 0.76 W | — |
| 48 kHz, offline | 1.2 GB | 19 W (29 W for the SoC) | about 79 J |

In real time, the extra SoC power is lost in background fluctuations, so only the GPU's share is listed. Every setting stays within 1.5 GB, including exit layer 12.

## Library

Add the package with Swift Package Manager:

```swift
.package(url: "https://github.com/JacobLinCool/apple-efficient-noise-reduction", from: "0.2.0"),
// in your target
.product(name: "EfficientNoiseReduction", package: "apple-efficient-noise-reduction"),
```

It requires macOS 14 or iOS 17, built with Xcode or `xcodebuild`, because MLX's Metal shaders only compile through Xcode's build system. DeepFilterNet3 does not use MLX and also works with `swift build`. The iOS build compiles but has not been tested on a device.

```swift
import EfficientNoiseReduction

// RE-USE weights: download once (see below), then locate
// try await ModelLocator.download(to: ModelLocator.defaultDirectory)
let model = try ReUseModel(directory: ModelLocator.locate(), precision: .fast)  // or .exact

// Whole recording; samples: [Float], mono
let clean = try model.enhance(samples, sampleRate: 16000, exitLayer: 12, lookAhead: 2)

// Streaming: blocks of any size in, final samples out; finish() returns the tail
let stream = try StreamingEnhancer(
    model: model, sampleRate: 48000,
    configuration: .init(exitLayer: 8, lookAhead: 1, chunk: .latency(0.5)))
stream.warmUp()
let out = stream.process(block)   // block: [Float] or UnsafeBufferPointer<Float>, mono
let tail = stream.finish()

// DeepFilterNet3: the same interface, 48 kHz, chunkFrames in 10 ms hops
let dfn3 = DeepFilterNet3Enhancer(model: try DeepFilterNet3(), chunkFrames: 1)
let denoised = dfn3.process(block48k) + dfn3.finish()

// Audio callbacks: write and read only copy samples; the network runs on its own thread
let rt = RealtimeEnhancer(DeepFilterNet3Enhancer(model: try DeepFilterNet3()))
rt.write(captured)        // capture callback, UnsafeBufferPointer<Float>
rt.read(into: playback)   // playback callback, UnsafeMutableBufferPointer<Float>
rt.stop()
```

Streams (`StreamingEnhancer`, `DeepFilterNet3Enhancer`) are not thread-safe, so use each one from a single thread. A loaded model can be shared by many streams.

RE-USE weights (`nvidia/Real-time_RE-USE`, pinned to revision `4c0c186`) are looked up in `REUSE_MODEL_DIR`, then `~/Library/Application Support/EfficientNoiseReduction/`, then the Hugging Face cache. `ModelLocator.download(to:)` and `noise-reduction download` fetch them.

### Command line

```bash
git clone https://github.com/JacobLinCool/apple-efficient-noise-reduction && cd apple-efficient-noise-reduction
make build                                   # builds the CLI with xcodebuild
CLI=.build/xcode/Build/Products/Release/noise-reduction
$CLI download                                # RE-USE weights
$CLI enhance noisy.wav clean.wav             # RE-USE, exit layer 8, 0.5 s latency
$CLI enhance noisy.wav clean.wav --offline -e 12 -l 2   # best quality
$CLI enhance noisy.wav clean.wav --engine dfn3          # DeepFilterNet3
$CLI enhance-batch noisy/ enhanced/ --engine dfn3       # a directory, model loaded once
$CLI live --monitor --latency 0.1            # the microphone, live (use headphones)
$CLI bench --sample-rate 16000               # the tables above, on your machine
```

## Correctness

Golden data comes from the reference implementations. `reference/golden.py` runs NVIDIA's PyTorch code, with a pure-PyTorch scan in place of the CUDA-only `mamba_ssm`. `reference/golden_dfn3.py` runs the official `deepfilternet` 0.5.6.

| Check | RE-USE | DeepFilterNet3 |
|---|---|---|
| Network output, given the reference features | max error about 2e-6 (`.exact`) | max error about 3e-6 |
| Whole-pipeline waveform vs the reference | 39–47 dB SNR | 125–135 dB SNR |
| Chunked vs one piece | 119 dB SNR or more (frame by frame, 7 frames, 0.5 s) | 136 dB SNR or more (frame by frame, 7 frames, irregular pushes) |
| Real-time pipeline at real-time pace | the one-piece output behind a constant delay, no underruns | — |

RE-USE's remaining waveform error comes from its input, not the network. The phase of near-silent STFT bins differs at rounding level between FFT implementations, and the model is sensitive to it. The default `.fast` precision (fp16, with the state in fp32) is 55–57 dB SNR from `.exact` (full fp32), and scores the same on VoiceBank-DEMAND.

The golden data contains third-party audio and model outputs, so it is generated locally rather than committed:

```bash
make golden   # needs uv; downloads the RE-USE weights and deepfilternet
make test     # tests without golden data or weights are skipped
```

## Building from source

```bash
make build        # CLI
make test         # library tests
make app          # menu bar app with its driver, Developer ID signed (needs XcodeGen)
make app SIGN=-   # the same, ad hoc signed for your own machine
make dist         # notarized zip (needs `xcrun notarytool store-credentials notary`)
```

Pushing a version tag runs `.github/workflows/release.yml`, which builds, signs and notarizes the app and attaches the zip to a GitHub release. `App/Tools/` has two command-line checks for an installed driver: a loopback test, and a recorder that uses the virtual microphone the way a meeting app does.

## How it works

- **RE-USE on the GPU.** The encoder and decoders are MPSGraph executables, compiled per chunk shape, with their streaming caches kept on the GPU. The Mamba blocks are a compiled MLX graph with custom Metal kernels: a fused conv1d and SiLU, a selective scan that keeps each channel's state in registers, and a fused residual LayerNorm. Everything stays in one (time, frequency, channel) layout, so nothing is transposed.
- **STFT on the CPU.** Each frame's result is independent of batch size, so every chunking gives the same output.
- **DeepFilterNet3 on the CPU.** At about 3.4M MACs per 10 ms, GPU dispatch would cost more than the arithmetic. Convolutions are im2col plus BLAS, BatchNorm is folded at load time, and Metal is never initialised.
- **Not the Neural Engine.** For RE-USE's convolutional part, Core ML took 7.5 ms per call on the Neural Engine against 2.0 ms on the GPU, at no lower energy, and the Mamba scan cannot run there at all (`reference/ane_experiment.py`).
- **Virtual microphone.** The driver (`Driver/`) publishes two devices on one clock: the visible microphone, and a hidden output the app plays into. The microphone reads back exactly what was written. The app compensates for the real microphone's separate clock by playing one sample more or fewer per callback, interpolated; this has not yet been tested with a microphone whose clock actually drifts.

## License

The code, including the virtual microphone driver, is MIT licensed.

- The RE-USE weights are released by NVIDIA under the [NVIDIA One-Way Noncommercial License (NSCLv1)](https://github.com/NVlabs/HMAR/blob/main/LICENSE), for noncommercial research and development only. They are not included here; they are downloaded on request.
- The DeepFilterNet3 weights are dual licensed MIT / Apache-2.0 and ship as safetensors in `Sources/EfficientNoiseReduction/Resources/`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
