import Foundation
import MLX
import XCTest

@testable import EfficientNoiseReduction

/// DeepFilterNet3 against golden outputs of the official `deepfilternet` package
/// (`reference/golden_dfn3.py`). The weights ship with the package, so these always run.
final class DeepFilterNet3Tests: XCTestCase {
    static let model = try! DeepFilterNet3()

    func fixture(_ name: String) throws -> [String: MLXArray] {
        let url = try Fixtures.url(name)
        return try loadArrays(url: url)
    }

    func maxDiff(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }

    func snr(_ ref: [Float], _ out: [Float]) -> Double {
        var s = 0.0, e = 0.0
        for (r, o) in zip(ref, out) {
            s += Double(r) * Double(r)
            e += Double(r - o) * Double(r - o)
        }
        return 10 * log10(s / max(e, 1e-30))
    }

    func run(_ input: [Float], chunk: Int, blocks: [Int]) -> [Float] {
        let e = DeepFilterNet3Enhancer(model: Self.model, chunkFrames: chunk)
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

    /// libDF's analysis: vorbis window over the previous and current hop, FFT, 1/n_fft.
    func testSTFTMatchesLibDF() throws {
        let a = try fixture("dfn3_speech")
        let x = a["input"]!.asArray(Float.self) + [Float](repeating: 0, count: 960)
        let ref = a["spec"]!  // (T, 481, 2)
        let fft = STFT(size: 960)
        let window = (0 ..< 960).map { i -> Float in
            let s = sin(0.5 * Double.pi * (Double(i) + 0.5) / 480)
            return Float(sin(0.5 * Double.pi * s * s))
        }
        var worst: Float = 0
        for t in [0, 1, 5, 100, ref.dim(0) - 1] {
            var buf = [Float](repeating: 0, count: 960)
            for i in 0 ..< 960 {
                let j = (t - 1) * 480 + i
                buf[i] = (j >= 0 ? x[j] : 0) * window[i]
            }
            let spec = fft.forward(buf).map { $0 / 960 }
            worst = max(worst, maxDiff(spec, ref[t].asArray(Float.self)))
        }
        print("STFT max |diff| \(worst)")
        XCTAssertLessThan(worst, 1e-6)

        // Inverse: the unnormalised sum, as realfft's c2r.
        let probe = (0 ..< 960).map { Float(sin(Double($0) * 0.37)) }
        let back = fft.inverse(fft.forward(probe)).map { $0 / 960 }
        XCTAssertLessThan(maxDiff(probe, back), 1e-5)
    }

    /// The network alone, fed the reference features: ERB gains and deep-filter coefficients.
    func testNetworkMatchesReference() throws {
        for name in ["dfn3_speech", "dfn3_noisy"] {
            let a = try fixture(name)
            let T = a["erb_feat"]!.dim(0)
            // DfNet.pad_feat: features shift two frames earlier, zeros at the end.
            let erb = a["erb_feat"]![2...].asArray(Float.self) + [Float](repeating: 0, count: 2 * 32)
            let spec = a["spec_feat"]![2...].asArray(Float.self) + [Float](repeating: 0, count: 2 * 96 * 2)
            var state = DeepFilterNet3.State(Self.model)
            let (mask, coefs) = Self.model(erb: erb, spec: spec, frames: T, state: &state)
            let dm = maxDiff(mask, a["mask"]!.asArray(Float.self))
            let dc = maxDiff(coefs, a["coefs"]!.asArray(Float.self))
            print("\(name): mask max |diff| \(dm), coefs max |diff| \(dc)")
            XCTAssertLessThan(dm, 1e-4, name)
            XCTAssertLessThan(dc, 1e-4, name)
        }
    }

    func testEnhanceMatchesReference() throws {
        for name in ["dfn3_speech", "dfn3_noisy"] {
            let a = try fixture(name)
            let ref = a["output"]!.asArray(Float.self)
            let out = run(a["input"]!.asArray(Float.self), chunk: 100, blocks: [1 << 20])
            XCTAssertEqual(out.count, ref.count)
            let db = snr(ref, out)
            print("\(name): SNR vs deepfilternet \(String(format: "%.1f", db)) dB")
            XCTAssertGreaterThan(db, 60, name)
        }
    }

    /// Chunk and push sizes change only when output appears.
    func testStreamingMatchesWhole() throws {
        let x = try fixture("dfn3_noisy")["input"]!.asArray(Float.self)
        let whole = run(x, chunk: 100, blocks: [1 << 20])
        for (chunk, blocks) in [(1, [480]), (1, [77, 1000, 3]), (7, [256])] {
            let out = run(x, chunk: chunk, blocks: blocks)
            XCTAssertEqual(out.count, whole.count)
            let db = snr(whole, out)
            print("chunk \(chunk) blocks \(blocks): SNR vs whole \(String(format: "%.1f", db)) dB")
            XCTAssertGreaterThan(db, 100)
        }
    }
}
