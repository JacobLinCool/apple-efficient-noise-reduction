import AVFoundation
import ArgumentParser
import Foundation
import EfficientNoiseReduction
import os

struct Live: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Enhance the default microphone live; play it back (use headphones) and/or record it.")

    @OptionGroup var latency: LatencyOptions
    @OptionGroup var model: ModelOptions

    @Option(help: "Seconds to run.")
    var duration = 10.0

    @Option(help: "Write the enhanced stream to this WAV file.")
    var record: String?

    @Flag(help: "Play the enhanced audio on the default output device.")
    var monitor = false

    @Option(help: "Silence queued ahead of the first output to absorb compute time, seconds (default: 0.03, or for NemotronLabs RE-USE 0.15 + a quarter of the chunk).")
    var margin: Double?

    var effectiveMargin: Double {
        if let margin { return margin }
        guard model.engine == .nemotron else { return 0.03 }
        // A whole chunk is computed at once: a few hundred ms per second of 48 kHz audio.
        return 0.15 + 0.25 * (latency.latency ?? 5)
    }

    func validate() throws {
        if model.engine == .nemotron && latency.whole {
            throw ValidationError("--whole waits for the end of the stream; use --latency for live NemotronLabs RE-USE")
        }
    }

    func run() throws {
        let engine = AVAudioEngine()
        let inputFormat = engine.inputNode.outputFormat(forBus: 0)
        let rate = Int(inputFormat.sampleRate)
        guard model.runRate(for: rate) == rate else {
            throw ValidationError("\(model.engine.rawValue) needs a \(model.runRate(for: rate)) Hz input device; this one runs at \(rate) Hz")
        }
        let recorded = OSAllocatedUnfairLock(initialState: [Float]())
        var onOutput: (@Sendable ([Float]) -> Void)?
        if record != nil {
            onOutput = { block in recorded.withLock { $0 += block } }
        }
        let rt = RealtimeEnhancer(
            try model.makeEnhancer(sampleRate: rate, latency: latency), margin: effectiveMargin, onOutput: onOutput)

        let channels = Int(inputFormat.channelCount)
        let scratch = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1 << 16)
        defer { scratch.deallocate() }
        engine.inputNode.installTap(onBus: 0, bufferSize: AVAudioFrameCount(rate / 100), format: inputFormat) { buffer, _ in
            let n = min(Int(buffer.frameLength), scratch.count)
            guard let data = buffer.floatChannelData else { return }
            for i in 0 ..< n {
                var v: Float = 0
                for c in 0 ..< channels { v += data[c][i] }
                scratch[i] = v / Float(channels)
            }
            rt.write(UnsafeBufferPointer(rebasing: scratch[..<n]))
        }
        if monitor {
            let mono = AVAudioFormat(standardFormatWithSampleRate: Double(rate), channels: 1)!
            let source = AVAudioSourceNode(format: mono) { _, _, frames, list in
                let abl = UnsafeMutableAudioBufferListPointer(list)
                let out = UnsafeMutableBufferPointer<Float>(
                    start: abl[0].mData!.assumingMemoryBound(to: Float.self), count: Int(frames))
                rt.read(into: out)
                return noErr
            }
            engine.attach(source)
            engine.connect(source, to: engine.mainMixerNode, format: mono)
        }
        try engine.start()
        print(
            String(
                format: "live %@ at %d Hz: latency %.0f ms + first compute", model.engine.rawValue, rate,
                rt.latencySeconds * 1000))
        Thread.sleep(forTimeInterval: duration)
        engine.stop()
        rt.stop()
        print("underruns \(rt.underruns), overruns \(rt.overruns)")
        if let record {
            let samples = recorded.withLock { $0 }
            try AudioFile.write(samples, sampleRate: rate, to: URL(filePath: record))
            print("wrote \(String(format: "%.1f", Double(samples.count) / Double(rate))) s to \(record)")
        }
    }
}
