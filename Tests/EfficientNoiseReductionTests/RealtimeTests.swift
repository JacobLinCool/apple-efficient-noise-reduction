import Foundation
import XCTest

@testable import EfficientNoiseReduction

final class RealtimeTests: XCTestCase {
    /// Capture and playback callbacks paced in real time: playback is the offline result
    /// behind one constant delay, without dropouts.
    func testRealtimePlaybackIsDelayedOfflineOutput() throws {
        guard let model = ParityTests.model else { throw XCTSkip("Real-time RE-USE weights not found") }
        let f = try ParityTests().fixture("sr16k_exit8_la0")
        let config = EnhancerConfiguration(exitLayer: 3, lookAhead: 1, chunk: .latency(0.2))
        let offline = try StreamingEnhancer(model: model, sampleRate: 16000, configuration: config)
        let reference = offline.process(f.input) + offline.finish()

        let rt = RealtimeEnhancer(
            try StreamingEnhancer(model: model, sampleRate: 16000, configuration: config), margin: 0.05)
        let block = 160  // 10 ms callbacks
        var played: [Float] = []
        var out = [Float](repeating: 0, count: block)
        let start = ContinuousClock.now
        var i = 0
        while i + block <= f.input.count {
            f.input.withUnsafeBufferPointer { rt.write(UnsafeBufferPointer(rebasing: $0[i ..< i + block])) }
            out.withUnsafeMutableBufferPointer { rt.read(into: $0) }
            played += out
            i += block
            let next = start + .milliseconds(10 * (i / block))
            if next > .now { Thread.sleep(forTimeInterval: Double((next - .now).components.attoseconds) * 1e-18) }
        }
        rt.stop()
        XCTAssertEqual(rt.underruns, 0)
        XCTAssertEqual(rt.overruns, 0)

        // The delay is where the played stream lines up with the reference.
        let probe = 4000 ..< 8000
        var best = (offset: -1, error: Float.infinity)
        for offset in 0 ..< min(played.count - probe.upperBound, 16000) {
            var e: Float = 0
            for k in probe { e += abs(played[offset + k] - reference[k]) }
            if e < best.error { best = (offset, e) }
        }
        let delay = Double(best.offset) / 16000
        print(String(format: "real-time delay %.0f ms (algorithmic + margin %.0f ms)", delay * 1000, rt.latencySeconds * 1000))
        XCTAssertGreaterThanOrEqual(delay, rt.latencySeconds - 0.011)
        XCTAssertLessThan(delay, rt.latencySeconds + 0.1)
        let n = played.count - best.offset
        var s = 0.0, e = 0.0
        for k in 0 ..< n {
            s += Double(reference[k] * reference[k])
            e += Double((played[best.offset + k] - reference[k]) * (played[best.offset + k] - reference[k]))
        }
        XCTAssertGreaterThan(10 * log10(s / max(e, 1e-30)), 100)
    }
}
