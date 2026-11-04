import Foundation

/// The two latency knobs of Real-time RE-USE (arXiv:2606.25621), plus the chunking that
/// trades a little extra delay for throughput.
///
/// - `exitLayer` (3...12) sets the compute cost: inference stops after that many
///   time-frequency Mamba blocks and uses the decoder trained for that depth.
/// - `lookAhead` (0...2) sets the algorithmic latency: each look-ahead frame waits for
///   one more 20 ms hop of future audio and selects the matching parallel input conv.
/// - `chunk` sets how many STFT frames are processed per network call. The network is
///   causal and carries its state between calls, so every chunking produces the same
///   output; larger chunks only amortise per-call overhead and add buffering delay.
public struct EnhancerConfiguration: Sendable, Hashable {
    public var exitLayer: Int
    public var lookAhead: Int
    public var chunk: Chunk

    public enum Chunk: Sendable, Hashable {
        /// A fixed number of 20 ms hops per network call (1 = frame-by-frame).
        case frames(Int)
        /// The largest chunk whose buffering keeps the total algorithmic latency (window,
        /// look-ahead and chunk buffering) within this many seconds.
        case latency(Double)
        /// Whole-file processing: the largest chunk that keeps memory near 1-2 GB.
        case offline
    }

    public init(exitLayer: Int = 8, lookAhead: Int = 0, chunk: Chunk = .latency(0.5)) {
        self.exitLayer = exitLayer
        self.lookAhead = lookAhead
        self.chunk = chunk
    }

    /// Frames per network call for a given geometry.
    func chunkFrames(_ geometry: STFTGeometry) -> Int {
        switch chunk {
        case .frames(let n):
            return max(1, n)
        case .latency(let seconds):
            let budget = seconds * geometry.sampleRate - Double(geometry.algorithmicLatencySamples(lookAhead: lookAhead))
            return max(1, Int(budget / Double(geometry.hop)) + 1)
        case .offline:
            // Throughput is flat from about 0.5 s per call up; activations grow with
            // frames × bins, so cap that product: 2 s at 16 kHz, about 0.7 s at 48 kHz,
            // which keeps the process near 1-2 GB.
            return min(100, max(24, 32_000 / geometry.bins))
        }
    }
}

/// Sampling-frequency-independent STFT sizes: a fixed 40 ms window and 20 ms hop at any rate.
public struct STFTGeometry: Sendable, Hashable {
    public let sampleRate: Double
    public let nFFT: Int
    public let hop: Int
    public let window: Int

    public var bins: Int { nFFT / 2 + 1 }

    /// Matches NVIDIA's `make_even(size * sr // 8000)`, including the integer division.
    public init(sampleRate: Int, base: ModelConfig.STFT) {
        func makeEven(_ v: Int) -> Int { v % 2 == 0 ? v : v + 1 }
        self.sampleRate = Double(sampleRate)
        nFFT = makeEven(base.nFFT * sampleRate / base.sampleRate)
        hop = makeEven(base.hopSize * sampleRate / base.sampleRate)
        window = makeEven(base.winSize * sampleRate / base.sampleRate)
    }

    /// Input needed before the first sample of a frame can be emitted: one window plus the
    /// look-ahead hops (the paper's `w + look_ahead * h`).
    public func algorithmicLatencySamples(lookAhead: Int) -> Int {
        nFFT + lookAhead * hop
    }
}

/// `config.json` from the Hugging Face repository.
public struct ModelConfig: Decodable, Sendable {
    public struct Model: Decodable, Sendable {
        public let hidFeature: Int
        public let dState: Int
        public let dConv: Int
        public let expand: Int
        public let numTFMamba: Int
        public let compressFactor: String

        enum CodingKeys: String, CodingKey {
            case hidFeature = "hid_feature", dState = "d_state", dConv = "d_conv", expand
            case numTFMamba = "num_tfmamba", compressFactor = "compress_factor"
        }
    }

    public struct STFT: Decodable, Sendable {
        public let nFFT: Int
        public let hopSize: Int
        public let winSize: Int
        public let sampleRate: Int

        enum CodingKeys: String, CodingKey {
            case nFFT = "n_fft", hopSize = "hop_size", winSize = "win_size", sampleRate = "sampling_rate"
        }
    }

    public let model: Model
    public let stft: STFT

    enum CodingKeys: String, CodingKey {
        case model = "model_cfg", stft = "stft_cfg"
    }
}
