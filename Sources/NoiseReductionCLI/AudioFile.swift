@preconcurrency import AVFoundation
import Foundation

enum AudioFile {
    /// Decode any format AVFoundation reads to mono float at the file's own rate.
    static func read(_ url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        let format = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        try file.read(into: buffer)
        let n = Int(buffer.frameLength), channels = Int(format.channelCount)
        var mono = [Float](repeating: 0, count: n)
        for c in 0 ..< channels {
            let p = buffer.floatChannelData![c]
            for i in 0 ..< n { mono[i] += p[i] / Float(channels) }
        }
        return (mono, Int(format.sampleRate))
    }

    /// Resample mono float audio with AVAudioConverter (DFN3 runs at 48 kHz only).
    static func resample(_ samples: [Float], from: Int, to: Int) throws -> [Float] {
        let src = AVAudioFormat(standardFormatWithSampleRate: Double(from), channels: 1)!
        let dst = AVAudioFormat(standardFormatWithSampleRate: Double(to), channels: 1)!
        guard let converter = AVAudioConverter(from: src, to: dst) else { throw CocoaError(.featureUnsupported) }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        let input = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: AVAudioFrameCount(samples.count))!
        input.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { input.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        let capacity = AVAudioFrameCount(Double(samples.count) * Double(to) / Double(from) + 1024)
        let output = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: capacity)!
        var fed = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if fed {
                status.pointee = .endOfStream
                return nil
            }
            fed = true
            status.pointee = .haveData
            return input
        }
        if let error { throw error }
        return Array(UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
    }

    /// Write mono float32 WAV.
    static func write(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!
        let file = try AVAudioFile(
            forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        try file.write(from: buffer)
    }
}
