import ArgumentParser
import Foundation
import MLX
import EfficientNoiseReduction

@main
struct NoiseReductionCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "noise-reduction",
        abstract: "Real-time RE-USE and DeepFilterNet3 speech enhancement on Apple silicon.",
        subcommands: [Enhance.self, EnhanceBatch.self, Live.self, Bench.self, Download.self])
}

enum Engine: String, ExpressibleByArgument, CaseIterable {
    case reuse, dfn3
}

struct ModelOptions: ParsableArguments {
    @Option(help: "Enhancement model: reuse (Real-time RE-USE, GPU) or dfn3 (DeepFilterNet3, CPU, 48 kHz).")
    var engine: Engine = .reuse

    @Option(help: "RE-USE weights directory with config.json and model.safetensors (default: REUSE_MODEL_DIR or the Hugging Face cache).")
    var model: String?

    @Flag(help: "RE-USE: full float32 matmuls instead of TF32 on M5-class GPUs (slower; matches PyTorch to ~2e-6).")
    var exact = false

    func load() throws -> ReUseModel {
        let dir = try model.map { URL(filePath: $0) } ?? ModelLocator.locate()
        return try ReUseModel(directory: dir, precision: exact ? .exact : .fast)
    }

    /// The sample rate the engine runs at for `rate` input: RE-USE takes any rate, DFN3 48 kHz.
    func runRate(for rate: Int) -> Int { engine == .dfn3 ? DeepFilterNet3.sampleRate : rate }

    /// Loads the model once; the returned closure makes a fresh stream per call.
    func factory(latency: LatencyOptions) throws -> (Int) throws -> any SpeechEnhancer {
        switch engine {
        case .reuse:
            let m = try load()
            let config = latency.configuration
            return { try StreamingEnhancer(model: m, sampleRate: $0, configuration: config) }
        case .dfn3:
            let m = try DeepFilterNet3()
            // 10 ms hops; 40 ms of window and look-ahead before the chunk buffering.
            let frames =
                if latency.offline { 100 }
                else if let f = latency.frames { f }
                else { max(1, Int(((latency.latency ?? 0.5) - 0.04) / 0.01 + 1e-9) + 1) }
            return { rate in
                precondition(rate == DeepFilterNet3.sampleRate)
                return DeepFilterNet3Enhancer(model: m, chunkFrames: frames)
            }
        }
    }

    func makeEnhancer(sampleRate: Int, latency: LatencyOptions) throws -> any SpeechEnhancer {
        try factory(latency: latency)(sampleRate)
    }
}

struct LatencyOptions: ParsableArguments {
    @Option(name: .shortAndLong, help: "RE-USE early-exit depth, 3-12: lower is faster, higher sounds better.")
    var exitLayer = 8

    @Option(name: .shortAndLong, help: "RE-USE look-ahead frames, 0-2: each adds 20 ms of algorithmic latency.")
    var lookAhead = 0

    @Option(help: "Latency budget in seconds; sets the chunk size (e.g. 0.5).")
    var latency: Double?

    @Option(help: "Frames per network call (RE-USE 20 ms hops, DFN3 10 ms); overrides --latency.")
    var frames: Int?

    @Flag(help: "Whole-file processing in the largest chunks that keep memory near 1-2 GB.")
    var offline = false

    var configuration: EnhancerConfiguration {
        let chunk: EnhancerConfiguration.Chunk =
            if offline { .offline }
            else if let frames { .frames(frames) }
            else { .latency(latency ?? 0.5) }
        return .init(exitLayer: exitLayer, lookAhead: lookAhead, chunk: chunk)
    }
}

struct Enhance: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Enhance an audio file (written as float WAV).")

    @Argument var input: String
    @Argument var output: String
    @OptionGroup var latency: LatencyOptions
    @OptionGroup var model: ModelOptions

    @Option(help: "Input block size in milliseconds, simulating a live stream.")
    var blockMs = 10.0

    @Flag(help: "Feed blocks at their real-time pace, as a live stream would arrive.")
    var realtime = false

    func run() throws {
        let (decoded, fileRate) = try AudioFile.read(URL(filePath: input))
        let rate = model.runRate(for: fileRate)
        let samples = rate == fileRate ? decoded : try AudioFile.resample(decoded, from: fileRate, to: rate)
        let e = try model.makeEnhancer(sampleRate: rate, latency: latency)
        let block = max(1, Int(blockMs / 1000 * Double(rate)))
        var out: [Float] = []
        out.reserveCapacity(samples.count)
        e.warmUp()
        let start = ContinuousClock.now
        var busy = Duration.zero
        var i = 0
        while i < samples.count {
            let n = min(block, samples.count - i)
            let t0 = ContinuousClock.now
            out += samples.withUnsafeBufferPointer { e.process(UnsafeBufferPointer(rebasing: $0[i ..< i + n])) }
            busy += ContinuousClock.now - t0
            i += n
            if realtime {
                let due = start + .seconds(Double(i) / Double(rate))
                if due > .now { Thread.sleep(forTimeInterval: (due - .now).seconds) }
            }
        }
        let t0 = ContinuousClock.now
        out += e.finish()
        busy += ContinuousClock.now - t0
        try AudioFile.write(out, sampleRate: rate, to: URL(filePath: output))
        let duration = Double(samples.count) / Double(rate)
        print(
            String(
                format: "%@: %@, %.1f s at %d Hz, latency %.0f ms + compute, RTF %.4f (%.0fx real time)",
                output, model.engine.rawValue, duration, rate, e.latencySeconds * 1000, busy.seconds / duration,
                duration / busy.seconds))
        print(
            String(
                format: "memory: MLX peak %.1f MB, process footprint peak %.1f MB",
                Double(Memory.peakMemory) / 1e6, Double(peakFootprint()) / 1e6))
    }
}

struct EnhanceBatch: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "enhance-batch",
        abstract: "Enhance every audio file in a directory (float WAV, at the engine's sample rate).")

    @Argument var inputDir: String
    @Argument var outputDir: String
    @OptionGroup var latency: LatencyOptions
    @OptionGroup var model: ModelOptions

    func run() throws {
        let make = try model.factory(latency: latency)
        let fm = FileManager.default
        try fm.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
        let files = try fm.contentsOfDirectory(atPath: inputDir).filter { $0.hasSuffix(".wav") || $0.hasSuffix(".flac") }.sorted()
        var audio = 0.0, busy = Duration.zero
        for name in files {
            let (decoded, fileRate) = try AudioFile.read(URL(filePath: inputDir).appending(path: name))
            let rate = model.runRate(for: fileRate)
            let samples = rate == fileRate ? decoded : try AudioFile.resample(decoded, from: fileRate, to: rate)
            let e = try make(rate)
            let t0 = ContinuousClock.now
            let out = e.process(samples) + e.finish()
            busy += ContinuousClock.now - t0
            audio += Double(samples.count) / Double(rate)
            let base = (name as NSString).deletingPathExtension
            try AudioFile.write(out, sampleRate: rate, to: URL(filePath: outputDir).appending(path: base + ".wav"))
        }
        print(String(format: "%d files, %.1f s of audio, RTF %.4f", files.count, audio, busy.seconds / audio))
    }
}

struct Bench: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Measure compute per network call and real-time factor across latency settings.")

    @OptionGroup var model: ModelOptions

    @Option(help: "Sample rate of the synthetic test signal.")
    var sampleRate = 16000

    @Option(help: "Seconds of audio per setting.")
    var seconds = 20.0

    @Option(parsing: .upToNextOption, help: "Exit layers to test.")
    var exitLayers = [3, 6, 8, 12]

    @Option(parsing: .upToNextOption, help: "Chunk sizes in frames (0 = offline).")
    var chunks = [1, 5, 12, 24, 49, 0]

    @Option(help: "Look-ahead frames.")
    var lookAhead = 0

    func run() throws {
        let rate = model.runRate(for: sampleRate)
        // Speech-like test signal: noise bursts with a pitch-modulated tone.
        let n = Int(seconds * Double(rate))
        var rng = SystemRandomNumberGenerator()
        let signal: [Float] = (0 ..< n).map { i in
            let t = Double(i) / Double(rate)
            let env = 0.5 + 0.5 * sin(2 * .pi * 3 * t)
            return Float(0.1 * env * sin(2 * .pi * (150 + 30 * sin(2 * .pi * t)) * t) + 0.02 * Double.random(in: -1 ... 1, using: &rng))
        }
        // One loaded model per engine; a fresh stream per setting.
        let make: (Int, Int) throws -> any SpeechEnhancer
        let layers: [Int]
        switch model.engine {
        case .reuse:
            let m = try model.load()
            layers = exitLayers
            make = { layer, chunk in
                try StreamingEnhancer(
                    model: m, sampleRate: rate,
                    configuration: .init(exitLayer: layer, lookAhead: lookAhead, chunk: chunk == 0 ? .offline : .frames(chunk)))
            }
        case .dfn3:
            let m = try DeepFilterNet3()
            layers = [0]
            make = { _, chunk in DeepFilterNet3Enhancer(model: m, chunkFrames: chunk == 0 ? 100 : chunk) }
        }
        print("\(model.engine.rawValue), \(rate) Hz, \(seconds) s per setting, look-ahead \(lookAhead)")
        print("exit  frames/call  latency(ms)  compute/call ms (mean / p95 / max)  RTF      total latency p95 (ms)")
        for layer in layers {
            for chunk in chunks {
                // Warm up kernels and shapes on a fresh stream.
                let warm = try make(layer, chunk)
                warm.warmUp()
                _ = warm.process(Array(signal.prefix(rate * 2)))
                _ = warm.finish()

                let e = try make(layer, chunk)
                let block = rate / 100  // 10 ms capture callbacks
                var times: [Double] = []
                var total = 0.0
                signal.withUnsafeBufferPointer { buf in
                    var i = 0
                    while i < n {
                        let k = min(block, n - i)
                        let t0 = ContinuousClock.now
                        let out = e.process(UnsafeBufferPointer(rebasing: buf[i ..< i + k]))
                        let dt = (ContinuousClock.now - t0).seconds
                        total += dt
                        if !out.isEmpty { times.append(dt) }
                        i += k
                    }
                    let t0 = ContinuousClock.now
                    _ = e.finish()
                    total += (ContinuousClock.now - t0).seconds
                }
                times.sort()
                let mean = times.reduce(0, +) / Double(max(1, times.count))
                let p95 = times.isEmpty ? 0 : times[min(times.count - 1, Int(Double(times.count) * 0.95))]
                print(
                    String(
                        format: "%4@  %11@  %11.0f  %10.3f / %6.3f / %6.3f          %.4f   %@",
                        model.engine == .reuse ? "\(layer)" : "-", chunk == 0 ? "offline" : "\(chunk)",
                        e.latencySeconds * 1000, mean * 1000, p95 * 1000, (times.last ?? 0) * 1000, total / seconds,
                        chunk == 0 ? "-" : String(format: "%.0f", (e.latencySeconds + p95) * 1000)))
            }
        }
    }
}

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download nvidia/Real-time_RE-USE (NVIDIA One-Way Noncommercial License).")

    @Option(help: "Destination directory.")
    var to = ModelLocator.defaultDirectory.path

    func run() async throws {
        let dir = try await ModelLocator.download(to: URL(filePath: to))
        print("downloaded to \(dir.path)")
    }
}

extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) * 1e-18 }
}

/// Peak physical footprint of this process (what Activity Monitor reports), including
/// Metal buffers.
func peakFootprint() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Int(info.ledger_phys_footprint_peak) : 0
}
