import Foundation

/// A streaming speech enhancer for one mono stream: push samples in any block size, get
/// back the samples that became final; `finish()` returns the rest, so the total output
/// length equals the total input length. Not thread-safe.
public protocol SpeechEnhancer: AnyObject {
    var sampleRate: Int { get }
    /// Worst-case delay from an input sample to its output, excluding compute time.
    var latencySeconds: Double { get }
    /// Prepare kernels for this stream's shapes so the first chunk does not stall.
    func warmUp()
    func process(_ samples: UnsafeBufferPointer<Float>) -> [Float]
    func finish() -> [Float]
}

extension SpeechEnhancer {
    public func process(_ samples: [Float]) -> [Float] {
        samples.withUnsafeBufferPointer { process($0) }
    }
}
