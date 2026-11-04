import AVFoundation
import EfficientNoiseReduction

enum PipelineError: LocalizedError {
    case noInput

    var errorDescription: String? {
        switch self {
        case .noInput: "The microphone has no usable input format."
        }
    }
}

/// Physical microphone → `RealtimeEnhancer` → the virtual microphone's hidden sink.
///
/// Capture and playback run on separate engines because they are separate devices with
/// separate clocks; `PlaybackReader` absorbs the drift between them.
final class Pipeline {
    let realtime: RealtimeEnhancer
    private let capture = AVAudioEngine()
    private let playback = AVAudioEngine()

    /// - Parameter margin: compute slack in seconds; see `RealtimeEnhancer`.
    init(enhancer: any SpeechEnhancer, margin: Double, input: AudioDeviceID, sink: AudioDeviceID) throws {
        let rate = Double(enhancer.sampleRate)
        realtime = RealtimeEnhancer(enhancer, margin: margin)

        try capture.inputNode.auAudioUnit.setDeviceID(input)
        let inputFormat = capture.inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else { throw PipelineError.noInput }
        let writer = CaptureWriter(realtime, from: inputFormat, to: rate)
        let sinkNode = AVAudioSinkNode { _, frames, list in
            writer.push(list, frames: Int(frames))
            return noErr
        }
        capture.attach(sinkNode)
        capture.connect(capture.inputNode, to: sinkNode, format: inputFormat)

        try playback.outputNode.auAudioUnit.setDeviceID(sink)
        let mono = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1)!
        let reader = PlaybackReader(realtime, sampleRate: rate)
        let source = AVAudioSourceNode(format: mono) { _, _, frames, list in
            reader.render(list, frames: Int(frames))
            return noErr
        }
        playback.attach(source)
        playback.connect(source, to: playback.mainMixerNode, format: mono)

        do {
            try playback.start()
            try capture.start()
        } catch {
            stop()
            throw error
        }
    }

    func stop() {
        capture.stop()
        playback.stop()
        realtime.stop()
    }
}

/// Downmixes capture buffers to mono and converts them to the enhancer's rate.
private final class CaptureWriter {
    private let realtime: RealtimeEnhancer
    private let converter: AVAudioConverter?
    private let mono: AVAudioPCMBuffer
    private let converted: AVAudioPCMBuffer

    init(_ realtime: RealtimeEnhancer, from format: AVAudioFormat, to rate: Double) {
        self.realtime = realtime
        let capacity: AVAudioFrameCount = 16384
        let monoIn = AVAudioFormat(standardFormatWithSampleRate: format.sampleRate, channels: 1)!
        let monoOut = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1)!
        mono = AVAudioPCMBuffer(pcmFormat: monoIn, frameCapacity: capacity)!
        converted = AVAudioPCMBuffer(
            pcmFormat: monoOut, frameCapacity: AVAudioFrameCount(Double(capacity) * rate / format.sampleRate) + 64)!
        converter = format.sampleRate == rate ? nil : AVAudioConverter(from: monoIn, to: monoOut)
    }

    func push(_ list: UnsafePointer<AudioBufferList>, frames: Int) {
        let n = min(frames, Int(mono.frameCapacity))
        let out = mono.floatChannelData![0]
        out.update(repeating: 0, count: n)
        var channels = 0
        for buffer in UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list)) {
            guard let data = buffer.mData?.assumingMemoryBound(to: Float.self) else { continue }
            let stride = Int(buffer.mNumberChannels)
            for c in 0 ..< stride {
                for i in 0 ..< n { out[i] += data[i * stride + c] }
            }
            channels += stride
        }
        if channels > 1 {
            let scale = 1 / Float(channels)
            for i in 0 ..< n { out[i] *= scale }
        }
        mono.frameLength = AVAudioFrameCount(n)

        guard let converter else {
            realtime.write(UnsafeBufferPointer(start: out, count: n))
            return
        }
        var consumed = false
        converted.frameLength = 0
        converter.convert(to: converted, error: nil) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return self.mono
        }
        realtime.write(UnsafeBufferPointer(start: converted.floatChannelData![0], count: Int(converted.frameLength)))
    }
}

/// Plays the enhanced stream, holding the queued amount steady against clock drift by
/// playing one sample more or fewer per callback (linearly interpolated) when the
/// minimum queue level over a window moves away from where it settled.
private final class PlaybackReader {
    private let realtime: RealtimeEnhancer
    private let scratch = UnsafeMutableBufferPointer<Float>.allocate(capacity: 16384)
    private let window: Int
    private let tolerance: Int
    private var windowMin = Int.max
    private var windowFrames = 0
    private var windowsSeen = 0
    private var target: Int?
    private var adjust = 0

    init(_ realtime: RealtimeEnhancer, sampleRate: Double) {
        self.realtime = realtime
        window = Int(sampleRate)
        tolerance = Int(sampleRate * 0.002)
    }

    deinit { scratch.deallocate() }

    func render(_ list: UnsafeMutablePointer<AudioBufferList>, frames n: Int) {
        let abl = UnsafeMutableAudioBufferListPointer(list)
        let out = UnsafeMutableBufferPointer(start: abl[0].mData!.assumingMemoryBound(to: Float.self), count: n)
        track(realtime.bufferedOutput, frames: n)
        let m = n + adjust
        guard adjust != 0, n > 1, m <= scratch.count else {
            realtime.read(into: out)
            return
        }
        let src = UnsafeMutableBufferPointer(rebasing: scratch[..<m])
        realtime.read(into: src)
        let step = Float(m - 1) / Float(n - 1)
        for i in 0 ..< n {
            let x = Float(i) * step
            let j = min(Int(x), m - 2)
            let f = x - Float(j)
            out[i] = src[j] + (src[j + 1] - src[j]) * f
        }
    }

    private func track(_ queued: Int, frames: Int) {
        // Start once the stream is playing; skip the first window while it settles.
        guard queued > 0 || windowsSeen > 0 else { return }
        windowMin = min(windowMin, queued)
        windowFrames += frames
        guard windowFrames >= window else { return }
        windowsSeen += 1
        if windowsSeen == 2 {
            target = windowMin
        } else if let target {
            adjust = windowMin > target + tolerance ? 1 : windowMin < target - tolerance ? -1 : 0
        }
        windowMin = .max
        windowFrames = 0
    }
}
