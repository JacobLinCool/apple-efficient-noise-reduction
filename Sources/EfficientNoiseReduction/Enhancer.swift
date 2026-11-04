import Accelerate
import ComplexModule
import Foundation
import MLX

/// Loaded Real-time RE-USE weights. Thread-safe; create one per process and make an
/// enhancer per audio stream.
public final class ReUseModel: @unchecked Sendable {
    public let config: ModelConfig
    let weights: Weights
    private var networks: [String: RealtimeNetwork] = [:]
    private let lock = NSLock()

    /// Float32 matmul precision on GPUs with matrix units (M5 and later).
    public enum Precision: Sendable {
        /// TF32 matmuls: about twice as fast on M5; the waveform stays ~52 dB SNR from
        /// `.exact`, closer than two FFT implementations agree on the same input.
        case fast
        /// Full float32: matches NVIDIA's PyTorch network output to ~2e-6.
        case exact
    }

    /// `directory` holds `config.json` and `model.safetensors` from `nvidia/Real-time_RE-USE`.
    ///
    /// `precision` is applied through `MLX_ENABLE_TF32`, which MLX reads once per process
    /// at the first float32 matmul: it takes effect only if set before then, and an
    /// existing `MLX_ENABLE_TF32` in the environment wins. On GPUs without matrix units
    /// both settings run the same full-float32 kernels.
    ///
    /// Loading also caps MLX's buffer cache at 256 MB. Uncapped, MLX keeps every freed
    /// activation buffer for reuse and the process grows to several GB with no speed
    /// gain; set `Memory.cacheLimit` afterwards to choose another process-wide limit.
    public let precision: Precision

    public init(directory: URL, precision: Precision = .fast) throws {
        self.precision = precision
        setenv("MLX_ENABLE_TF32", precision == .fast ? "1" : "0", 0)
        Memory.cacheLimit = 256 << 20
        config = try JSONDecoder().decode(ModelConfig.self, from: Data(contentsOf: directory.appending(path: "config.json")))
        guard config.model.compressFactor == "relu_log1p" else {
            throw EnhancementError.invalidConfiguration("unsupported compress_factor \(config.model.compressFactor)")
        }
        weights = try Weights(url: directory.appending(path: "model.safetensors"))
    }

    func network(exitLayer: Int, lookAhead: Int) throws -> RealtimeNetwork {
        lock.lock()
        defer { lock.unlock() }
        let key = "\(exitLayer)/\(lookAhead)"
        if let n = networks[key] { return n }
        let n = try RealtimeNetwork(
            weights: weights, config: config, exitLayer: exitLayer, lookAhead: lookAhead, precision: precision)
        networks[key] = n
        return n
    }

    /// Enhance a complete recording; same output as NVIDIA's `offline_inference.py`.
    public func enhance(_ samples: [Float], sampleRate: Int, exitLayer: Int = 8, lookAhead: Int = 0) throws -> [Float] {
        let e = try StreamingEnhancer(
            model: self, sampleRate: sampleRate,
            configuration: .init(exitLayer: exitLayer, lookAhead: lookAhead, chunk: .offline))
        return e.process(samples) + e.finish()
    }
}

/// Streaming speech enhancement of one mono stream: push samples in any block size, get
/// back the samples that are final. Output sample `i` always equals sample `i` of
/// whole-file processing; the configuration only decides how soon it comes out.
///
/// Not thread-safe: call from one thread (or actor) at a time.
public final class StreamingEnhancer: SpeechEnhancer {
    public let geometry: STFTGeometry
    public let configuration: EnhancerConfiguration
    /// STFT frames per network call.
    public let chunkFrames: Int
    public var sampleRate: Int { Int(geometry.sampleRate) }

    private let network: RealtimeNetwork
    private var state: RealtimeNetwork.State?
    private let window: [Float]  // periodic Hann, centred in nFFT
    private let windowSquared: [Float]
    private let windowMLX: MLXArray
    private var half: Int { geometry.nFFT / 2 }

    // Analysis: the reflect-padded input from absolute padded index `paddedStart`.
    private var padded: [Float] = []
    private var paddedStart = 0
    private var started = false  // left reflection written
    private var received = 0
    private var nextFrame = 0
    private var pendingFrames: [Float] = []  // whole frames waiting for a full chunk

    // Synthesis: overlap-add buffers from absolute padded index `olaStart`.
    private var ola: [Float] = []
    private var envelope: [Float] = []
    private var olaStart = 0
    private var framesOut = 0
    private var emitted = 0
    private var finished = false

    public init(model: ReUseModel, sampleRate: Int, configuration: EnhancerConfiguration = .init()) throws {
        geometry = STFTGeometry(sampleRate: sampleRate, base: model.config.stft)
        self.configuration = configuration
        chunkFrames = configuration.chunkFrames(geometry)
        network = try model.network(exitLayer: configuration.exitLayer, lookAhead: configuration.lookAhead)
        let n = geometry.nFFT, w = geometry.window
        var win = [Float](repeating: 0, count: n)
        for k in 0 ..< w { win[(n - w) / 2 + k] = Float(0.5 - 0.5 * cos(2 * Double.pi * Double(k) / Double(w))) }
        window = win
        windowSquared = win.map { $0 * $0 }
        windowMLX = MLXArray(win)
    }

    /// Worst-case delay from an input sample to its enhanced output, excluding compute:
    /// one window, the look-ahead hops and the chunk buffering.
    public var latencySeconds: Double {
        Double(geometry.algorithmicLatencySamples(lookAhead: configuration.lookAhead) + (chunkFrames - 1) * geometry.hop)
            / geometry.sampleRate
    }

    /// Trace and compile the network for this stream's chunk shapes ahead of time, so
    /// the first real chunk does not stall a live stream. Does not touch the stream.
    public func warmUp() {
        let zeros = MLXArray.zeros([chunkFrames, geometry.bins])
        var scratch: RealtimeNetwork.State?
        for _ in 0 ..< 2 {  // first call (look-ahead padding), then steady state
            if let out = network(mag: zeros, phase: zeros, state: &scratch) {
                eval([out.mag, out.phase] + scratch!.arrays)
            }
        }
    }

    public func process(_ samples: UnsafeBufferPointer<Float>) -> [Float] {
        precondition(!finished, "process(_:) after finish()")
        received += samples.count
        padded.append(contentsOf: samples)
        if !started {
            guard received > half else { return [] }
            // torch.stft(center=True, pad_mode="reflect"): x[half], ..., x[1] before x[0].
            padded.insert(contentsOf: padded[1 ... half].reversed(), at: 0)
            started = true
        }
        collectFrames()
        while pendingFrames.count >= chunkFrames * geometry.nFFT {
            runChunk(frames: chunkFrames, trailingZeros: 0)
        }
        return emit(upTo: min((framesOut - 1) * geometry.hop, framesOut * geometry.hop - half))
    }

    /// Flush the end of the stream (right reflection, look-ahead) and return the rest of
    /// the output, so the total output length equals the total input length.
    public func finish() -> [Float] {
        precondition(!finished, "finish() called twice")
        finished = true
        guard started, received > half else {
            // Too short for a reflect-padded frame; NVIDIA's code errors here, we pass through.
            let rest = Array(padded.suffix(received - emitted))
            emitted = received
            return rest
        }
        // Right reflection: x[L-2], ..., x[L-1-half].
        let end = padded.count
        padded.append(contentsOf: padded[(end - 1 - half) ..< (end - 1)].reversed())
        collectFrames()
        repeat {
            let n = min(chunkFrames, pendingFrames.count / geometry.nFFT)
            let last = n * geometry.nFFT == pendingFrames.count
            runChunk(frames: n, trailingZeros: last ? configuration.lookAhead : 0)
        } while !pendingFrames.isEmpty
        let totalFrames = nextFrame
        var out = emit(upTo: min(received, (totalFrames - 1) * geometry.hop))
        if emitted < received {
            out += [Float](repeating: 1e-8, count: received - emitted)
            emitted = received
        }
        return out
    }

    private func collectFrames() {
        let n = geometry.nFFT, hop = geometry.hop
        while (nextFrame * hop + n) - paddedStart <= padded.count {
            let s = nextFrame * hop - paddedStart
            pendingFrames.append(contentsOf: padded[s ..< s + n])
            nextFrame += 1
        }
        // Keep the last half + 1 input samples: finish() reflects them.
        let drop = min(nextFrame * hop - paddedStart, padded.count - half - 1)
        if drop > 4096 {
            padded.removeFirst(drop)
            paddedStart += drop
        }
    }

    /// Analysis STFT, network and synthesis for `frames` pending frames, plus
    /// `trailingZeros` all-zero network inputs that release the look-ahead at the end.
    private func runChunk(frames: Int, trailingZeros: Int) {
        let n = geometry.nFFT
        var mag: [MLXArray] = [], pha: [MLXArray] = []
        if frames > 0 {
            let x = MLXArray(pendingFrames[0 ..< frames * n], [frames, n])
            pendingFrames.removeFirst(frames * n)
            let spec = rfft(x * windowMLX, axis: -1, stream: .cpu)
            mag.append(log1p(abs(spec)))
            pha.append(phase(spec))
        }
        if trailingZeros > 0 {
            let z = MLXArray.zeros([trailingZeros, geometry.bins])
            mag.append(z)
            pha.append(z)
        }
        guard !mag.isEmpty else { return }
        guard let out = network(mag: concatenated(mag), phase: concatenated(pha), state: &state) else {
            eval(state!.arrays)
            return
        }
        // NVIDIA's "strange sweep artifact" fix: silence frames where most bins are <= 0.
        let F = out.mag.dim(1)
        let nonPositive = (out.mag .<= 0).asType(.int32).sum(axis: -1, keepDims: true)
        let keep = (nonPositive * 2 .<= F).asType(.float32)
        let lin = expm1(maximum(out.mag, 0)) * keep
        let synth = irfft(lin * cos(out.phase) + lin * sin(out.phase) * MLXArray(Complex<Float>(0, 1)), n: n, axis: -1, stream: .cpu)
            * windowMLX
        eval([synth] + state!.arrays)
        overlapAdd(synth.asArray(Float.self), frames: synth.dim(0))
    }

    private func overlapAdd(_ frames: [Float], frames count: Int) {
        let n = geometry.nFFT, hop = geometry.hop
        let needed = (framesOut + count - 1) * hop + n - olaStart
        if needed > ola.count {
            ola += [Float](repeating: 0, count: needed - ola.count)
            envelope += [Float](repeating: 0, count: needed - envelope.count)
        }
        frames.withUnsafeBufferPointer { src in
            ola.withUnsafeMutableBufferPointer { dst in
                envelope.withUnsafeMutableBufferPointer { env in
                    for t in 0 ..< count {
                        let o = (framesOut + t) * hop - olaStart
                        vDSP_vadd(dst.baseAddress! + o, 1, src.baseAddress! + t * n, 1, dst.baseAddress! + o, 1, vDSP_Length(n))
                        vDSP_vadd(env.baseAddress! + o, 1, windowSquared, 1, env.baseAddress! + o, 1, vDSP_Length(n))
                    }
                }
            }
        }
        framesOut += count
    }

    /// Return output samples [emitted, limit), normalised by the window envelope.
    private func emit(upTo limit: Int) -> [Float] {
        guard limit > emitted else { return [] }
        let from = emitted + half - olaStart, to = limit + half - olaStart
        var out = [Float](repeating: 0, count: to - from)
        for i in 0 ..< out.count {
            let e = envelope[from + i]
            out[i] = e > 1e-11 ? ola[from + i] / e : ola[from + i]
        }
        emitted = limit
        let drop = emitted + half - olaStart
        if drop > 4096 {
            ola.removeFirst(drop)
            envelope.removeFirst(drop)
            olaStart += drop
        }
        return out
    }
}

/// `torch.angle` of an STFT. Adding +0 turns -0 into +0, so the purely real DC and
/// Nyquist bins get phase 0 or π like PyTorch's, never -π.
func phase(_ spec: MLXArray) -> MLXArray {
    atan2(spec.imaginaryPart() + 0, spec.realPart() + 0)
}
