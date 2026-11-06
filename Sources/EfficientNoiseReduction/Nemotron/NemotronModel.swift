import Accelerate
import ComplexModule
import Foundation
import MLX

/// NVIDIA NemotronLabs RE-USE (arXiv:2603.02641): the offline universal speech
/// enhancement model, a USEMamba with 30 time-frequency blocks that are bidirectional in
/// both time and frequency. Higher quality than Real-time RE-USE at the cost of latency:
/// it sees a whole chunk at once. Weights are under the OpenMDW License 1.1.
///
/// Thread-safe; load once per process and make an enhancer per audio stream.
public final class NemotronModel: @unchecked Sendable {
    public let config: ModelConfig
    public let precision: ReUseModel.Precision
    let network: OfflineNetwork

    /// `directory` holds `config.json` and `model.safetensors` from
    /// `nvidia/NVIDIA-NemotronLabs-RE-USE` (see `ModelLocator.locate(.nemotron)`).
    ///
    /// `.fast` runs the convolutions and Mamba GEMMs in float16 with float32 statistics,
    /// SSM state and residual stream; `.exact` runs everything in float32 (and sets
    /// `MLX_ENABLE_TF32` as `ReUseModel` does).
    ///
    /// Loading sets MLX's buffer cache to at least 1 GB (see `ReUseModel`). A chunk's Mamba activations
    /// are hundreds of MB; with a smaller cache every block frees and re-allocates them,
    /// which makes 5 s chunks at 48 kHz several times slower.
    public init(directory: URL, precision: ReUseModel.Precision = .fast) throws {
        self.precision = precision
        setenv("MLX_ENABLE_TF32", precision == .fast ? "1" : "0", 0)
        BufferCache.require(1 << 30)
        config = try JSONDecoder().decode(ModelConfig.self, from: Data(contentsOf: directory.appending(path: "config.json")))
        guard config.model.compressFactor == "relu_log1p" else {
            throw EnhancementError.invalidConfiguration("unsupported compress_factor \(config.model.compressFactor)")
        }
        let weights = try Weights(url: directory.appending(path: "model.safetensors"))
        network = try OfflineNetwork(weights: weights, config: config, precision: precision)
    }

    /// Enhance a complete recording. `.chunks(seconds: 5)` is NVIDIA's
    /// `inference_chunk.py` default; `.whole` is `inference.py`.
    public func enhance(
        _ samples: [Float], sampleRate: Int, chunking: NemotronEnhancer.Chunking = .chunks(seconds: 5)
    ) -> [Float] {
        let e = NemotronEnhancer(model: self, sampleRate: sampleRate, chunking: chunking)
        return e.process(samples) + e.finish()
    }
}

/// The NemotronLabs RE-USE network for one chunk: encoder (MPSGraph) → 30 Mamba blocks
/// (one compiled MLX graph per chunk shape) → decoders (MPSGraph).
final class OfflineNetwork: @unchecked Sendable {
    let convs: OfflineConvStack
    let blocks: [OfflineTFBlock]
    let dtype: DType

    init(weights w: Weights, config: ModelConfig, precision: ReUseModel.Precision) throws {
        dtype = precision == .fast ? .float16 : .float32
        convs = try OfflineConvStack(weights: w, precision: precision)
        blocks = try (0 ..< config.model.numTFMamba).map { [dtype] in try OfflineTFBlock(w, prefix: "TSMamba.\($0)", dtype: dtype) }
    }

    /// One chunk at a time: streams sharing the model take turns on the GPU, and a
    /// shape's first call (its MLX trace) never runs twice concurrently.
    private let lock = NSLock()

    /// Compressed magnitude and phase (T, F) of one chunk in, the same out.
    func callAsFunction(mag: [Float], phase: [Float], frames T: Int, bins F: Int) -> (mag: [Float], phase: [Float]) {
        lock.lock()
        defer { lock.unlock() }
        // NVIDIA appends two zero frames and two zero bins before encoding
        // ("prevent unpredictable errors"), then stacks (magnitude, phase) as channels.
        let width = F + 2
        var features = [Float](repeating: 0, count: (T + 2) * width * 2)
        for t in 0 ..< T {
            for f in 0 ..< F {
                features[(t * width + f) * 2] = mag[t * F + f]
                features[(t * width + f) * 2 + 1] = phase[t * F + f]
            }
        }
        let encoded = convs.encode(features, frames: T, bins: F)
        let mixed = mamba(frames: T, bins: F)([encoded])[0]
        eval(mixed)
        return convs.decode(mixed, frames: T, bins: F)
    }

    /// Compiled graphs for the few most recent chunk shapes. A stream uses one or two
    /// (its chunk and, at the end, a shorter one); whole-file processing of many files
    /// sees a new shape per file. Releasing a compiled function frees its traces.
    private let compiled = RecentCache<[Int], ([MLXArray]) -> [MLXArray]>(capacity: 3)

    private func mamba(frames T: Int, bins F: Int) -> ([MLXArray]) -> [MLXArray] {
        compiled.value(for: [T, F]) { compile { [unowned self] a in body(a) } }
    }

    private func body(_ a: [MLXArray]) -> [MLXArray] {
        var x = a[0]
        for block in blocks { x = block(x) }
        return [x.asType(.float32)]
    }
}

/// A small thread-safe cache that keeps the `capacity` most recently used values.
final class RecentCache<Key: Hashable, Value>: @unchecked Sendable {
    private let capacity: Int
    private var values: [Key: Value] = [:]
    private var order: [Key] = []
    private let lock = NSLock()

    init(capacity: Int) { self.capacity = capacity }

    func value(for key: Key, make: () -> Value) -> Value {
        lock.lock()
        defer { lock.unlock() }
        if let v = values[key] {
            order.removeAll { $0 == key }
            order.append(key)
            return v
        }
        let v = make()
        values[key] = v
        order.append(key)
        if order.count > capacity { values[order.removeFirst()] = nil }
        return v
    }
}

/// One time-frequency block of the offline model: bidirectional Mamba over time, then
/// over frequency, each wrapped as `x + LayerNorm(output_proj([fw(x) + x, bw(x) + x]))`,
/// with out_proj folded into output_proj as in `TFBlock`.
struct OfflineTFBlock: @unchecked Sendable {
    let time: MambaGroup
    let timeOut: TFBlock.Projection
    let freq: MambaGroup
    let freqOut: TFBlock.Projection

    /// Scan outputs and the pre-norm projections of this model reach 1e4–1e5 on ordinary
    /// speech, past float16's range; float16 storage computes both divided by 64.
    init(_ w: Weights, prefix p: String, dtype: DType) throws {
        let scale = dtype == .float16 ? 64 : 1
        time = try MambaGroup(
            w, prefixes: ["\(p).time_mamba.forward_blocks", "\(p).time_mamba.backward_blocks"], dtype: dtype,
            outputScale: scale)
        timeOut = try TFBlock.Projection(w, prefix: "\(p).time_mamba", outProj: time.outProj, dtype: dtype, scale: scale)
        freq = try MambaGroup(
            w, prefixes: ["\(p).freq_mamba.forward_blocks", "\(p).freq_mamba.backward_blocks"], dtype: dtype,
            outputScale: scale)
        freqOut = try TFBlock.Projection(w, prefix: "\(p).freq_mamba", outProj: freq.outProj, dtype: dtype, scale: scale)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = timeOut(scan: time.timeBidirectional(x), input: x)
        return freqOut(scan: freq.freq(x), input: x)
    }
}

/// Streaming NemotronLabs RE-USE for one mono stream, following NVIDIA's
/// `inference_chunk.py`: overlapping chunks, each enhanced on its own (STFT, network,
/// silent-frame gating, iSTFT) and cross-faded with a Hann window the length of the
/// chunk. The last chunk is aligned to the end of the stream. Output sample `i` equals
/// sample `i` of NVIDIA's whole-file chunked inference, whatever block sizes are pushed.
///
/// The model sees a chunk at a time, so the latency is one chunk: sample `t` is final
/// once sample `t + chunk` has arrived (no future chunk can reach back further). Longer
/// chunks give the model more context; the hop sets the compute, one chunk per hop.
///
/// Not thread-safe: call from one thread (or actor) at a time.
public final class NemotronEnhancer: SpeechEnhancer {
    public enum Chunking: Sendable, Hashable {
        /// The whole stream in one pass (NVIDIA's `inference.py`); output comes at
        /// `finish()`. Memory grows with the length: meant for short clips.
        case whole
        /// Chunks of `seconds`, starting every `hop` × `seconds` (NVIDIA's
        /// `hop_length_portion`); NVIDIA's default is 5 s with a hop of 0.5.
        case chunks(seconds: Double, hop: Double = 0.5)
    }

    public let geometry: STFTGeometry
    public let chunking: Chunking
    public var sampleRate: Int { Int(geometry.sampleRate) }
    /// Chunk and hop in samples (nil chunk: whole stream).
    public let chunkSamples: Int?
    public let hopSamples: Int

    private let model: NemotronModel
    private let stft: ChunkSTFT
    private let crossfade: [Float]  // Hann(chunk), clamped to 1e-7

    private var input: [Float] = []
    private var inputStart = 0  // absolute index of input[0]
    private var received = 0
    private var nextChunk = 0  // absolute start of the next regular chunk
    private var accumulated: [Float] = []  // Σ chunk output × window, from `emitted`
    private var weights: [Float] = []  // Σ window, from `emitted`
    private var emitted = 0
    private var finished = false
    /// Chunks enhanced so far (each is one network call).
    public private(set) var chunksProcessed = 0

    public init(model: NemotronModel, sampleRate: Int, chunking: Chunking = .chunks(seconds: 5)) {
        self.model = model
        self.chunking = chunking
        geometry = STFTGeometry(sampleRate: sampleRate, base: model.config.stft)
        stft = ChunkSTFT(geometry: geometry)
        switch chunking {
        case .whole:
            chunkSamples = nil
            hopSamples = 0
            crossfade = []
        case .chunks(let seconds, let hop):
            precondition(seconds > 0 && hop > 0 && hop <= 1, "chunks need a positive length and a hop in (0, 1]")
            // NVIDIA: chunk_size = int(seconds * sr), hop_length = int(portion * chunk_size).
            let chunk = max(1, Int(seconds * Double(sampleRate)))
            chunkSamples = chunk
            hopSamples = max(1, Int(hop * Double(chunk)))
            crossfade = (0 ..< chunk).map { n in
                max(Float(0.5 - 0.5 * cos(2 * Double.pi * Double(n) / Double(chunk))), 1e-7)
            }
        }
    }

    /// One chunk: an input sample's output is final once the chunk after it has arrived.
    public var latencySeconds: Double {
        chunkSamples.map { Double($0) / geometry.sampleRate } ?? .infinity
    }

    /// Compile the network for this stream's chunk shape ahead of time (silence through
    /// the network). Does not touch the stream.
    public func warmUp() {
        guard let chunk = chunkSamples, chunk > geometry.nFFT / 2 else { return }
        let T = chunk / geometry.hop + 1, F = geometry.bins
        let zeros = [Float](repeating: 0, count: T * F)
        _ = model.network(mag: zeros, phase: zeros, frames: T, bins: F)
    }

    public func process(_ samples: UnsafeBufferPointer<Float>) -> [Float] {
        precondition(!finished, "process(_:) after finish()")
        input.append(contentsOf: samples)
        received += samples.count
        guard let chunk = chunkSamples else { return [] }
        while nextChunk + chunk <= received {
            run(start: nextChunk, length: chunk, window: crossfade)
            nextChunk += hopSamples
        }
        // Keep what a future regular chunk or the end-aligned last one may read.
        let keep = min(nextChunk, max(0, received - chunk))
        if keep - inputStart > 1 << 16 {
            input.removeFirst(keep - inputStart)
            inputStart = keep
        }
        return emit(upTo: received - chunk)
    }

    public func finish() -> [Float] {
        precondition(!finished, "finish() called twice")
        finished = true
        let L = received
        guard L > 0 else { return [] }
        if let chunk = chunkSamples {
            if L < chunk {
                // Shorter than a chunk: one pass over everything with the window's head.
                run(start: 0, length: L, window: Array(crossfade[0 ..< L]))
            } else if (L - chunk) % hopSamples != 0 {
                run(start: L - chunk, length: chunk, window: crossfade)
            }
        } else {
            let out = enhanceOnce(input[...])
            accumulated = out
            weights = [Float](repeating: 1, count: out.count)
        }
        return emit(upTo: L)
    }

    /// Enhance input[start ..< start + length] and add it to the cross-fade.
    private func run(start: Int, length: Int, window: [Float]) {
        chunksProcessed += 1
        let out = enhanceOnce(input[(start - inputStart) ..< (start - inputStart + length)])
        let offset = start - emitted
        let needed = offset + length
        if needed > accumulated.count {
            accumulated += [Float](repeating: 0, count: needed - accumulated.count)
            weights += [Float](repeating: 0, count: needed - weights.count)
        }
        for i in 0 ..< length {
            accumulated[offset + i] += out[i] * window[i]
            weights[offset + i] += window[i]
        }
    }

    /// NVIDIA's per-file body: STFT, network, silent-frame gating, iSTFT, then
    /// padding with 1e-8 (or trimming) to the input length.
    private func enhanceOnce(_ x: ArraySlice<Float>) -> [Float] {
        // torch.stft's reflect padding needs more than nFFT / 2 samples; NVIDIA's code
        // errors on shorter input, we pass it through.
        guard x.count > geometry.nFFT / 2 else { return Array(x) }
        let (mag, pha, T) = stft.analyze(x)
        let out = model.network(mag: mag, phase: pha, frames: T, bins: geometry.bins)
        return stft.synthesize(mag: out.mag, phase: out.phase, frames: T, length: x.count)
    }

    private func emit(upTo limit: Int) -> [Float] {
        guard limit > emitted else { return [] }
        let n = limit - emitted
        var out = [Float](repeating: 0, count: n)
        for i in 0 ..< n { out[i] = accumulated[i] / weights[i] }
        accumulated.removeFirst(n)
        weights.removeFirst(n)
        emitted = limit
        return out
    }
}

/// `torch.stft` / `torch.istft` with `center=True`, reflect padding and a periodic Hann
/// window over a whole chunk, plus NVIDIA's magnitude compression and gating. On the CPU,
/// so a frame's result never depends on how many frames are batched with it.
struct ChunkSTFT {
    let geometry: STFTGeometry
    private let window: [Float]
    private let windowSquared: [Float]
    private let windowMLX: MLXArray

    init(geometry: STFTGeometry) {
        self.geometry = geometry
        let n = geometry.nFFT, w = geometry.window
        var win = [Float](repeating: 0, count: n)
        for k in 0 ..< w { win[(n - w) / 2 + k] = Float(0.5 - 0.5 * cos(2 * Double.pi * Double(k) / Double(w))) }
        window = win
        windowSquared = win.map { $0 * $0 }
        windowMLX = MLXArray(win)
    }

    /// Compressed magnitude log1p(|X|) and phase, (T, F) row-major, T = 1 + L / hop.
    func analyze(_ x: ArraySlice<Float>) -> (mag: [Float], phase: [Float], frames: Int) {
        let n = geometry.nFFT, hop = geometry.hop, half = n / 2
        let L = x.count, s = x.startIndex
        // Reflect padding: x[half] … x[1], x, x[L-2] … x[L-1-half].
        var padded = [Float](repeating: 0, count: L + 2 * half)
        for i in 0 ..< half { padded[i] = x[s + half - i] }
        for i in 0 ..< L { padded[half + i] = x[s + i] }
        for i in 0 ..< half { padded[half + L + i] = x[s + L - 2 - i] }
        let T = 1 + L / hop
        var frames = [Float](repeating: 0, count: T * n)
        for t in 0 ..< T { frames[(t * n) ..< (t * n + n)] = padded[(t * hop) ..< (t * hop + n)] }
        let spec = rfft(MLXArray(frames, [T, n]) * windowMLX, axis: -1, stream: .cpu)
        let mag = log1p(abs(spec)), pha = phase(spec)
        eval(mag, pha)
        return (mag.asArray(Float.self), pha.asArray(Float.self), T)
    }

    /// Gate silent frames, decompress, inverse STFT, then pad with 1e-8 or trim to `length`.
    func synthesize(mag: [Float], phase: [Float], frames T: Int, length: Int) -> [Float] {
        let n = geometry.nFFT, hop = geometry.hop, half = n / 2, F = geometry.bins
        let m = MLXArray(mag, [T, F]), p = MLXArray(phase, [T, F])
        // NVIDIA's "strange sweep artifact" fix: silence frames where most bins are <= 0.
        let nonPositive = (m .<= 0).asType(.int32).sum(axis: -1, keepDims: true)
        let keep = (nonPositive * 2 .<= F).asType(.float32)
        let lin = expm1(maximum(m, 0)) * keep
        let synth = irfft(lin * cos(p) + lin * sin(p) * MLXArray(Complex<Float>(0, 1)), n: n, axis: -1, stream: .cpu) * windowMLX
        let frames = synth.asArray(Float.self)
        // Overlap-add and divide by the summed squared window, as torch.istft does.
        let total = n + hop * (T - 1)
        var ola = [Float](repeating: 0, count: total), envelope = [Float](repeating: 0, count: total)
        frames.withUnsafeBufferPointer { src in
            ola.withUnsafeMutableBufferPointer { dst in
                envelope.withUnsafeMutableBufferPointer { env in
                    for t in 0 ..< T {
                        let o = t * hop
                        vDSP_vadd(dst.baseAddress! + o, 1, src.baseAddress! + t * n, 1, dst.baseAddress! + o, 1, vDSP_Length(n))
                        vDSP_vadd(env.baseAddress! + o, 1, windowSquared, 1, env.baseAddress! + o, 1, vDSP_Length(n))
                    }
                }
            }
        }
        let produced = hop * (T - 1)  // torch.istft(center=True) length
        var out = [Float](repeating: 1e-8, count: length)
        for i in 0 ..< min(produced, length) {
            let e = envelope[half + i]
            out[i] = e > 1e-11 ? ola[half + i] / e : ola[half + i]
        }
        return out
    }
}
