import Foundation
import MLX
import XCTest

@testable import EfficientNoiseReduction

/// Compares the NemotronLabs RE-USE port with golden outputs from NVIDIA's own PyTorch
/// code (`reference/golden_nemotron.py`). Needs the model weights; skipped without them.
final class NemotronTests: XCTestCase {
    static let exact: NemotronModel? = load(.exact)
    static let fast: NemotronModel? = load(.fast)

    static func load(_ precision: ReUseModel.Precision) -> NemotronModel? {
        guard let dir = try? ModelLocator.locate(.nemotron) else { return nil }
        return try? NemotronModel(directory: dir, precision: precision)
    }

    func model(_ precision: ReUseModel.Precision = .exact) throws -> NemotronModel {
        guard let m = precision == .exact ? Self.exact : Self.fast else {
            throw XCTSkip("NemotronLabs RE-USE weights not found (set NEMOTRON_MODEL_DIR)")
        }
        return m
    }

    struct Fixture {
        let arrays: [String: MLXArray]
        var input: [Float] { arrays["input"]!.asArray(Float.self) }
        var output: [Float] { arrays["output"]!.asArray(Float.self) }
        var config: [Int] { arrays["config"]!.asArray(Int32.self).map(Int.init) }
    }

    func fixture(_ name: String) throws -> Fixture {
        Fixture(arrays: try loadArrays(url: try Fixtures.url(name)))
    }

    func snr(_ ref: [Float], _ out: [Float]) -> Double {
        var s = 0.0, e = 0.0
        for (r, o) in zip(ref, out) {
            s += Double(r) * Double(r)
            e += Double(r - o) * Double(r - o)
        }
        return 10 * log10(s / max(e, 1e-30))
    }

    func run(_ e: NemotronEnhancer, _ input: [Float], blocks: [Int]) -> [Float] {
        var out: [Float] = []
        var i = 0, k = 0
        while i < input.count {
            let n = min(blocks[k % blocks.count], input.count - i)
            out += e.process(Array(input[i ..< i + n]))
            i += n
            k += 1
        }
        return out + e.finish()
    }

    /// The network alone, fed NVIDIA's STFT features: the compressed magnitude and the
    /// phase it predicts. The phase is compared where the magnitude is audible; below
    /// that its angle is ill-conditioned.
    func testNetworkMatchesReference() throws {
        let m = try model()
        for name in ["nemotron_whole_16k", "nemotron_whole_44k", "nemotron_whole_48k"] {
            let f = try fixture(name)
            let noisyMag = f.arrays["noisy_mag"]!, noisyPha = f.arrays["noisy_pha"]!
            let (T, F) = (noisyMag.dim(0), noisyMag.dim(1))
            let out = m.network(
                mag: noisyMag.asArray(Float.self), phase: noisyPha.asArray(Float.self), frames: T, bins: F)
            let refMag = f.arrays["mag"]!.asArray(Float.self), refPha = f.arrays["pha"]!.asArray(Float.self)
            var magErr: Float = 0, phaErr: Float = 0, weight: Float = 0
            for i in refMag.indices {
                magErr = max(magErr, abs(out.mag[i] - refMag[i]))
                if refMag[i] > 0.05 {
                    var d = abs(out.phase[i] - refPha[i])
                    d = min(d, 2 * .pi - d)
                    phaErr += d * refMag[i]
                    weight += refMag[i]
                }
            }
            print("\(name): \(T)x\(F) magnitude max |err| \(magErr), weighted phase err \(phaErr / max(weight, 1e-9)) rad")
            XCTAssertLessThan(magErr, 2e-3, name)
            XCTAssertLessThan(phaErr / max(weight, 1e-9), 2e-3, name)
        }
    }

    /// Whole-file inference against NVIDIA's `inference.py`.
    func testWholeMatchesReference() throws {
        let m = try model()
        for name in ["nemotron_whole_16k", "nemotron_whole_44k", "nemotron_whole_48k"] {
            let f = try fixture(name)
            let out = m.enhance(f.input, sampleRate: f.config[0], chunking: .whole)
            XCTAssertEqual(out.count, f.output.count, name)
            let db = snr(f.output, out)
            print("\(name): SNR vs NVIDIA inference.py \(String(format: "%.1f", db)) dB")
            XCTAssertGreaterThan(db, 30, name)
        }
    }

    /// Chunked inference against NVIDIA's `inference_chunk.py`, pushed in irregular blocks.
    func testChunkedMatchesReference() throws {
        let m = try model()
        for name in ["nemotron_chunk_16k", "nemotron_chunk_48k", "nemotron_chunk_16k_short"] {
            let f = try fixture(name)
            let (rate, chunk, hop) = (f.config[0], f.config[1], f.config[2])
            let seconds = Double(chunk) / Double(rate)
            let e = NemotronEnhancer(
                model: m, sampleRate: rate, chunking: .chunks(seconds: seconds, hop: Double(hop) / Double(chunk)))
            XCTAssertEqual(e.chunkSamples, chunk, name)
            XCTAssertEqual(e.hopSamples, hop, name)
            let out = run(e, f.input, blocks: [480, 1, 7919, 160, 33333])
            XCTAssertEqual(out.count, f.output.count, name)
            let db = snr(f.output, out)
            print("\(name): SNR vs NVIDIA inference_chunk.py \(String(format: "%.1f", db)) dB")
            XCTAssertGreaterThan(db, 30, name)
        }
    }

    /// Push sizes change only when output appears: the stream equals one-shot processing,
    /// and nothing is emitted earlier than one chunk after its input.
    func testStreamingIsBlockSizeInvariant() throws {
        let m = try model()
        let f = try fixture("nemotron_chunk_16k")
        let input = f.input
        let make = { NemotronEnhancer(model: m, sampleRate: 16000, chunking: .chunks(seconds: 1.5, hop: 0.5)) }
        let once = make().process(input)
        let reference: [Float] = {
            let e = make()
            return e.process(input) + e.finish()
        }()
        XCTAssertEqual(once.count, input.count - 24000)  // everything but the last chunk is final
        let e = make()
        var out: [Float] = []
        var i = 0
        while i < input.count {
            let n = min(1234, input.count - i)
            out += e.process(Array(input[i ..< i + n]))
            i += n
            XCTAssertLessThanOrEqual(out.count, max(0, i - 24000))
        }
        out += e.finish()
        XCTAssertEqual(out.count, reference.count)
        XCTAssertGreaterThan(snr(reference, out), 100)
    }

    /// float16 storage with float32 statistics and state against full float32.
    func testFastPrecisionIsClose() throws {
        let exact = try model(.exact), fast = try model(.fast)
        let f = try fixture("nemotron_whole_16k")
        let a = exact.enhance(f.input, sampleRate: 16000, chunking: .whole)
        let b = fast.enhance(f.input, sampleRate: 16000, chunking: .whole)
        let db = snr(a, b)
        print("fast vs exact: \(String(format: "%.1f", db)) dB; fast vs NVIDIA \(String(format: "%.1f", snr(f.output, b))) dB")
        XCTAssertGreaterThan(db, 25)
    }
}
