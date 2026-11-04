import Foundation
import os

/// A `SpeechEnhancer` behind a pair of ring buffers, for audio I/O callbacks.
///
/// `write` (capture callback) and `read` (playback callback) only copy samples under a
/// short lock and never allocate or wait for the GPU; a dedicated thread runs the
/// network. Playback of the enhanced stream starts `latencySeconds` after the input,
/// padded with silence, so the delay stays fixed for the life of the stream and every
/// chunk may run up to `margin` seconds late without a dropout.
///
/// The worker thread keeps the enhancer alive until `stop()`.
public final class RealtimeEnhancer: @unchecked Sendable {
    public let enhancer: any SpeechEnhancer
    public let margin: Double

    private let input: RingBuffer
    private let output: RingBuffer
    private let wake = DispatchSemaphore(value: 0)
    private let flags = OSAllocatedUnfairLock(initialState: Flags())
    private let onOutput: (@Sendable ([Float]) -> Void)?

    private struct Flags {
        var stopped = false
        var playing = false
        var underruns = 0
        var overruns = 0
    }

    /// - Parameters:
    ///   - margin: slack for compute time, in seconds: a chunk may finish this long after
    ///     its input is complete without a dropout.
    ///   - onOutput: called on the worker thread with every enhanced block, for example
    ///     to record it; keep it fast.
    public init(
        _ enhancer: any SpeechEnhancer, margin: Double = 0.03, onOutput: (@Sendable ([Float]) -> Void)? = nil
    ) {
        self.enhancer = enhancer
        self.margin = margin
        self.onOutput = onOutput
        let capacity = enhancer.sampleRate * 10 + Int(enhancer.latencySeconds * Double(enhancer.sampleRate)) * 2
        input = RingBuffer(capacity: capacity)
        output = RingBuffer(capacity: capacity)
        enhancer.warmUp()
        let thread = Thread { self.run() }
        thread.name = "ReUse enhancer"
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    /// The stream's delay: the enhancer's worst-case algorithmic latency plus `margin`
    /// (more only if the first chunk takes longer than `margin` to compute).
    public var latencySeconds: Double { enhancer.latencySeconds + margin }

    /// Playback callbacks that found too few enhanced samples (filled with silence).
    public var underruns: Int { flags.withLock { $0.underruns } }
    /// Capture callbacks dropped because the worker fell more than the ring behind.
    public var overruns: Int { flags.withLock { $0.overruns } }
    /// Enhanced samples queued for `read`. When capture and playback run on different
    /// clocks, its level drifts; the reader can play slightly more or fewer samples to hold it.
    public var bufferedOutput: Int { output.count }

    /// Queue captured samples. Real-time safe.
    public func write(_ samples: UnsafeBufferPointer<Float>) {
        if !input.write(samples) { flags.withLock { $0.overruns += 1 } }
        wake.signal()
    }

    /// Fill `buffer` with enhanced samples, or silence until the stream is primed or when
    /// the worker is late. Real-time safe.
    public func read(into buffer: UnsafeMutableBufferPointer<Float>) {
        let playing = flags.withLock { $0.playing }
        guard playing, output.read(into: buffer) else {
            buffer.update(repeating: 0)
            if playing { flags.withLock { $0.underruns += 1 } }
            return
        }
    }

    /// Stop the worker; buffered input that has not been processed is dropped.
    public func stop() {
        let wasRunning = flags.withLock { s in
            defer { s.stopped = true }
            return !s.stopped
        }
        if wasRunning { wake.signal() }
    }

    private func run() {
        var block = [Float](repeating: 0, count: input.capacity)
        var fed = 0
        while true {
            wake.wait()
            if flags.withLock({ $0.stopped }) { return }
            let n = block.withUnsafeMutableBufferPointer { input.readAvailable(into: $0) }
            guard n > 0 else { continue }
            fed += n
            let out = block.withUnsafeBufferPointer { enhancer.process(UnsafeBufferPointer(rebasing: $0[..<n])) }
            guard !out.isEmpty else { continue }
            onOutput?(out)
            let starting = flags.withLock { s in
                defer { s.playing = true }
                return !s.playing
            }
            if starting {
                // Output sample i may take until input sample i + latency to exist, so play
                // sample 0 once `latency + margin` of input has arrived: every later sample
                // is then on time. The first chunk usually comes earlier than that (its
                // tail is held back by the look-ahead); silence fills the difference.
                let start = Int((enhancer.latencySeconds + margin) * Double(enhancer.sampleRate))
                let silence = [Float](repeating: 0, count: max(0, start - fed))
                _ = silence.withUnsafeBufferPointer { output.write($0) }
            }
            if !out.withUnsafeBufferPointer({ output.write($0) }) { flags.withLock { $0.overruns += 1 } }
        }
    }
}

/// Single-producer single-consumer float ring buffer guarded by an unfair lock held only
/// for the copy.
final class RingBuffer: @unchecked Sendable {
    let capacity: Int
    private let storage: UnsafeMutablePointer<Float>
    private let lock = OSAllocatedUnfairLock(initialState: (head: 0, count: 0))

    init(capacity: Int) {
        self.capacity = capacity
        storage = .allocate(capacity: capacity)
        storage.initialize(repeating: 0, count: capacity)
    }

    deinit { storage.deallocate() }

    var count: Int { lock.withLock { $0.count } }

    /// Append all of `samples`, or nothing if they do not fit.
    func write(_ samples: UnsafeBufferPointer<Float>) -> Bool {
        lock.withLockUnchecked { s in
            guard s.count + samples.count <= capacity else { return false }
            copy(from: samples.baseAddress!, count: samples.count, at: (s.head + s.count) % capacity)
            s.count += samples.count
            return true
        }
    }

    /// Fill `buffer` completely, or take nothing if fewer samples are queued.
    func read(into buffer: UnsafeMutableBufferPointer<Float>) -> Bool {
        lock.withLockUnchecked { s in
            guard s.count >= buffer.count else { return false }
            copy(to: buffer.baseAddress!, count: buffer.count, at: s.head)
            s.head = (s.head + buffer.count) % capacity
            s.count -= buffer.count
            return true
        }
    }

    /// Take everything queued (up to `buffer.count`); returns how many samples.
    func readAvailable(into buffer: UnsafeMutableBufferPointer<Float>) -> Int {
        lock.withLockUnchecked { s in
            let n = min(s.count, buffer.count)
            copy(to: buffer.baseAddress!, count: n, at: s.head)
            s.head = (s.head + n) % capacity
            s.count -= n
            return n
        }
    }

    private func copy(from src: UnsafePointer<Float>, count: Int, at start: Int) {
        let first = min(count, capacity - start)
        (storage + start).update(from: src, count: first)
        if count > first { storage.update(from: src + first, count: count - first) }
    }

    private func copy(to dst: UnsafeMutablePointer<Float>, count: Int, at start: Int) {
        let first = min(count, capacity - start)
        dst.update(from: storage + start, count: first)
        if count > first { (dst + first).update(from: storage, count: count - first) }
    }
}
