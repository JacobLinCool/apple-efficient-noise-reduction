// Records "Noise Reduction Microphone" for a few seconds into a WAV file, as a meeting app
// would, to check the app end to end.
//
//     swiftc -O App/Tools/record-check.swift -o .build/record-check && .build/record-check out.wav 8
import AVFoundation
import CoreAudio

var cfUID = "EfficientNoiseReduction_Microphone" as CFString
var mic = AudioDeviceID(kAudioObjectUnknown)
var size = UInt32(MemoryLayout<AudioDeviceID>.size)
var address = AudioObjectPropertyAddress(
    mSelector: kAudioHardwarePropertyTranslateUIDToDevice, mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, UInt32(MemoryLayout<CFString>.size), &cfUID, &size, &mic)
guard mic != kAudioObjectUnknown else { print("virtual microphone not found"); exit(1) }

let path = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "record.wav"
let seconds = CommandLine.arguments.count > 2 ? Double(CommandLine.arguments[2])! : 8
let engine = AVAudioEngine()
try engine.inputNode.auAudioUnit.setDeviceID(mic)
let format = engine.inputNode.outputFormat(forBus: 0)
var samples: [Float] = []
let lock = NSLock()
var blocks: [(Double, Float)] = []
let start = Date()
engine.inputNode.installTap(onBus: 0, bufferSize: 4800, format: format) { buffer, _ in
    let x = UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength))
    let rms = sqrt(x.map { $0 * $0 }.reduce(0, +) / Float(max(1, x.count)))
    lock.lock(); blocks.append((Date().timeIntervalSince(start), rms)); samples += x; lock.unlock()
}
try engine.start()
Thread.sleep(forTimeInterval: seconds)
engine.stop()
do {
    let mono = AVAudioFormat(standardFormatWithSampleRate: format.sampleRate, channels: 1)!
    let file = try AVAudioFile(forWriting: URL(filePath: path), settings: mono.settings)
    let buffer = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: AVAudioFrameCount(samples.count))!
    buffer.frameLength = buffer.frameCapacity
    samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: $0.count) }
    try file.write(from: buffer)
}
let first = blocks.first { $0.1 > 1e-4 }
print(String(format: "recorded %.1f s to %@; first non-silent block at %.2f s; rms by second:", seconds, path, first?.0 ?? -1))
for s in 0 ..< Int(seconds) {
    let b = blocks.filter { Int($0.0) == s }.map(\.1)
    print(String(format: "  %d s: %.1f dBFS", s, 20 * log10(max(1e-9, b.isEmpty ? 0 : b.reduce(0, +) / Float(b.count)))))
}
