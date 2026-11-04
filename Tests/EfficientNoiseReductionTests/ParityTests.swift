import Foundation
import MLX
import XCTest

@testable import EfficientNoiseReduction

/// Compares the Swift port with golden outputs from NVIDIA's own PyTorch code
/// (`reference/golden.py`). Needs the model weights; skipped when they are missing.
final class ParityTests: XCTestCase {
    static let model: ReUseModel? = {
        guard let dir = try? ModelLocator.locate() else { return nil }
        return try? ReUseModel(directory: dir, precision: .exact)
    }()

    struct Fixture {
        let input: [Float]
        let output: [Float]
        let sampleRate: Int
        let exitLayer: Int
        let lookAhead: Int
    }

    func fixture(_ name: String) throws -> Fixture {
        let url = try Fixtures.url(name)
        let a = try loadArrays(url: url)
        let cfg = a["config"]!.asArray(Int32.self).map(Int.init)
        return Fixture(
            input: a["input"]!.asArray(Float.self), output: a["output"]!.asArray(Float.self),
            sampleRate: cfg[0], exitLayer: cfg[1], lookAhead: cfg[2])
    }

    func requireModel() throws -> ReUseModel {
        guard let m = Self.model else { throw XCTSkip("Real-time RE-USE weights not found (set REUSE_MODEL_DIR)") }
        return m
    }

    /// Signal-to-difference ratio of `out` against `ref`, in dB.
    func snr(_ ref: [Float], _ out: [Float]) -> Double {
        var s = 0.0, e = 0.0
        for (r, o) in zip(ref, out) {
            s += Double(r) * Double(r)
            e += Double(r - o) * Double(r - o)
        }
        return 10 * log10(s / max(e, 1e-30))
    }

    func run(_ f: Fixture, chunk: EnhancerConfiguration.Chunk, blocks: [Int]? = nil) throws -> [Float] {
        let e = try StreamingEnhancer(
            model: try requireModel(), sampleRate: f.sampleRate,
            configuration: .init(exitLayer: f.exitLayer, lookAhead: f.lookAhead, chunk: chunk))
        var out: [Float] = []
        var i = 0, k = 0
        while i < f.input.count {
            let n = min(blocks.map { $0[k % $0.count] } ?? f.input.count, f.input.count - i)
            out += e.process(Array(f.input[i ..< i + n]))
            i += n
            k += 1
        }
        out += e.finish()
        return out
    }

    /// End to end against NVIDIA's waveform. The network itself agrees to ~2e-6
    /// (`testNetworkMatchesReference`); what remains comes from the input phase of
    /// near-silent bins, which differs at float rounding level between FFT
    /// implementations and which the network output is sensitive to.
    func testOfflineMatchesReference() throws {
        for name in ["sr16k_exit8_la0", "sr16k_exit12_la2", "sr16k_exit3_la1", "sr44k_exit6_la1", "sr44k_full_exit8_la0"] {
            let f = try fixture(name)
            let out = try run(f, chunk: .offline)
            XCTAssertEqual(out.count, f.output.count, name)
            let db = snr(f.output, out)
            print("\(name): SNR vs NVIDIA reference \(String(format: "%.1f", db)) dB")
            XCTAssertGreaterThan(db, 35, name)
        }
    }

    /// Chunk size and push block size change only when output appears, never what it is.
    func testStreamingMatchesOffline() throws {
        for name in ["sr16k_exit8_la0", "sr16k_exit12_la2"] {
            let f = try fixture(name)
            let offline = try run(f, chunk: .offline)
            for (chunk, blocks) in [
                (EnhancerConfiguration.Chunk.frames(1), [160, 333, 1]),
                (.frames(7), [4096]),
                (.latency(0.5), [512, 37]),
            ] {
                let out = try run(f, chunk: chunk, blocks: blocks)
                XCTAssertEqual(out.count, offline.count)
                let db = snr(offline, out)
                print("\(name) \(chunk): SNR vs offline \(String(format: "%.1f", db)) dB")
                XCTAssertGreaterThan(db, 100, "\(name) \(chunk)")
            }
        }
    }

    /// Output never runs ahead of the latency the configuration promises.
    func testOutputArrivesWithinLatency() throws {
        let f = try fixture("sr16k_exit8_la0")
        let model = try requireModel()
        for la in 0 ... 2 {
            let e = try StreamingEnhancer(
                model: model, sampleRate: 16000, configuration: .init(exitLayer: 3, lookAhead: la, chunk: .frames(1)))
            let latency = Int(e.latencySeconds * 16000)
            var fed = 0, out = 0
            while fed + 160 <= f.input.count {
                out += e.process(Array(f.input[fed ..< fed + 160])).count
                fed += 160
                XCTAssertGreaterThanOrEqual(out, fed - latency, "look-ahead \(la)")
            }
        }
    }
}

extension ParityTests {
    /// The network alone, fed NVIDIA's exact STFT features: compressed magnitude and
    /// phase outputs must agree to float32 accumulation error.
    func testNetworkMatchesReference() throws {
        let model = try requireModel()
        for name in ["sr16k_exit8_la0", "sr16k_exit12_la2", "sr16k_exit3_la1", "sr44k_exit6_la1"] {
            let url = try Fixtures.url(name)
            let a = try loadArrays(url: url)
            let f = try fixture(name)
            let z = MLXArray.zeros([f.lookAhead, a["noisy_mag"]!.dim(1)])
            let net = try model.network(exitLayer: f.exitLayer, lookAhead: f.lookAhead)
            var state: RealtimeNetwork.State? = nil
            let out = try XCTUnwrap(
                net(
                    mag: concatenated([a["noisy_mag"]!, z]), phase: concatenated([a["noisy_pha"]!, z]),
                    state: &state))
            let magDiff = abs(out.mag - a["mag"]!).max().item(Float.self)
            // Phase error weighted by output magnitude; wrap differences into (-π, π].
            let d = out.phase - a["pha"]!
            let wrapped = abs(atan2(sin(d), cos(d)))
            let w = expm1(maximum(a["mag"]!, 0))
            let phaseErr = ((wrapped * w).sum() / w.sum()).item(Float.self)
            print("\(name): magnitude max|d| \(magDiff), weighted phase error \(phaseErr) rad")
            XCTAssertLessThan(magDiff, 1e-4, name)
            XCTAssertLessThan(phaseErr, 1e-4, name)
        }
    }
}



extension ParityTests {
    /// The stateful network fed one frame at a time equals one whole-utterance call.
    func testNetworkFrameByFrameMatchesWhole() throws {
        let model = try requireModel()
        let url = try Fixtures.url("sr16k_exit12_la2")
        let a = try loadArrays(url: url)
        let z = MLXArray.zeros([2, a["noisy_mag"]!.dim(1)])
        let mag = concatenated([a["noisy_mag"]!, z]), pha = concatenated([a["noisy_pha"]!, z])
        let net = try model.network(exitLayer: 12, lookAhead: 2)
        var whole: RealtimeNetwork.State? = nil
        let ref = try XCTUnwrap(net(mag: mag, phase: pha, state: &whole))
        var state: RealtimeNetwork.State? = nil
        var mags: [MLXArray] = [], phases: [MLXArray] = []
        for t in 0 ..< mag.dim(0) {
            if let o = net(mag: mag[t ..< t + 1], phase: pha[t ..< t + 1], state: &state) {
                mags.append(o.mag)
                phases.append(o.phase)
            }
            eval(state!.arrays + mags.suffix(1))
        }
        let m = concatenated(mags), p = concatenated(phases)
        XCTAssertEqual(m.shape, ref.mag.shape)
        let magDiff = abs(m - ref.mag).max().item(Float.self)
        let d = p - ref.phase
        let w = expm1(maximum(ref.mag, 0))
        let phaseErr = ((abs(atan2(sin(d), cos(d))) * w).sum() / w.sum()).item(Float.self)
        print("frame-by-frame vs whole: magnitude max|d| \(magDiff), weighted phase error \(phaseErr) rad")
        XCTAssertLessThan(magDiff, 1e-4)
        XCTAssertLessThan(phaseErr, 1e-4)
    }
}
