// Checks the installed virtual microphone end to end: plays a tone into the hidden sink and
// records it back from "Noise Reduction Microphone".
//
//     swiftc -O App/Tools/loopback-check.swift -o .build/loopback-check && .build/loopback-check
import AVFoundation
import CoreAudio

func device(uid: String) -> AudioDeviceID? {
    var cfUID = uid as CFString
    var id = AudioDeviceID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyTranslateUIDToDevice, mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    let status = withUnsafeMutablePointer(to: &cfUID) {
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, UInt32(MemoryLayout<CFString>.size), $0, &size, &id)
    }
    return status == noErr && id != kAudioObjectUnknown ? id : nil
}

guard let mic = device(uid: "EfficientNoiseReduction_Microphone"), let sink = device(uid: "EfficientNoiseReduction_Sink") else {
    print("virtual devices not found"); exit(1)
}
print("mic \(mic), sink \(sink)")

let rate = 48000.0
let tone: (Int) -> Float = { Float(0.25 * sin(2 * Double.pi * 440 * Double($0) / rate)) }

let play = AVAudioEngine()
try play.outputNode.auAudioUnit.setDeviceID(sink)
var phase = 0
let mono = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1)!
let source = AVAudioSourceNode(format: mono) { _, _, frames, list in
    let out = UnsafeMutableAudioBufferListPointer(list)[0].mData!.assumingMemoryBound(to: Float.self)
    for i in 0 ..< Int(frames) { out[i] = tone(phase + i) }
    phase += Int(frames)
    return noErr
}
play.attach(source)
play.connect(source, to: play.mainMixerNode, format: mono)

let record = AVAudioEngine()
try record.inputNode.auAudioUnit.setDeviceID(mic)
let format = record.inputNode.outputFormat(forBus: 0)
print("mic format \(format)")
let lock = NSLock()
var captured: [Float] = []
let sinkNode = AVAudioSinkNode { _, frames, list in
    let data = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))[0].mData!.assumingMemoryBound(to: Float.self)
    lock.lock(); captured += UnsafeBufferPointer(start: data, count: Int(frames)); lock.unlock()
    return noErr
}
record.attach(sinkNode)
record.connect(record.inputNode, to: sinkNode, format: format)

try record.start()
Thread.sleep(forTimeInterval: 0.5)
lock.lock(); let silent = captured.map(abs).max() ?? 0; captured.removeAll(); lock.unlock()
print(String(format: "before playback: peak %.4f (expect 0)", silent))
try play.start()
Thread.sleep(forTimeInterval: 2)
play.stop()
Thread.sleep(forTimeInterval: 0.5)
record.stop()
let x = captured
let rms = { (s: ArraySlice<Float>) in sqrt(s.map { $0 * $0 }.reduce(0, +) / Float(max(1, s.count))) }
let onset = x.firstIndex { abs($0) > 1e-3 } ?? -1
print(String(format: "captured %d samples, signal onset at %.1f ms, rms during tone %.4f (expect %.4f)",
             x.count, Double(onset) / rate * 1000, rms(x[min(x.count, 24000) ..< min(x.count, 72000)]), 0.25 / sqrt(2)))
print(String(format: "rms after stop %.5f (expect 0)", rms(x.suffix(12000))))
// Discontinuities: a clean 440 Hz tone changes by at most 2*pi*440/48000*0.25 per sample.
let maxStep = zip(x.dropFirst(onset + 1), x.dropFirst(onset)).prefix(72000).map { abs($0 - $1) }.max() ?? 0
print(String(format: "max sample step %.4f (clean tone %.4f)", maxStep, 2 * Double.pi * 440 / rate * 0.25))
