import CoreAudio
import Foundation

/// The two devices of the bundled Audio Server plug-in (see `Driver/`).
enum VirtualDevice {
    /// What meeting apps select as their microphone.
    static let microphoneUID = "EfficientNoiseReduction_Microphone"
    /// Hidden output the app plays the enhanced audio into.
    static let sinkUID = "EfficientNoiseReduction_Sink"
    static let driverName = "EfficientNoiseReduction.driver"
}

struct AudioDevice: Hashable, Identifiable {
    let id: AudioDeviceID
    let uid: String
    let name: String
}

/// Thin wrappers over the Core Audio object API.
enum AudioSystem {
    static func inputDevices() -> [AudioDevice] {
        let ids: [AudioDeviceID] = array(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDevices)
        return ids.compactMap { id in
            guard inputChannels(id) > 0, let uid = string(id, kAudioDevicePropertyDeviceUID),
                uid != VirtualDevice.microphoneUID
            else { return nil }
            return AudioDevice(id: id, uid: uid, name: string(id, kAudioObjectPropertyName) ?? uid)
        }
    }

    /// Finds a device by UID, including hidden ones.
    static func device(uid: String) -> AudioDeviceID? {
        var cfUID = uid as CFString
        var id = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyTranslateUIDToDevice, mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        let status = withUnsafeMutablePointer(to: &cfUID) {
            AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject), &address, UInt32(MemoryLayout<CFString>.size), $0, &size, &id)
        }
        return status == noErr && id != kAudioObjectUnknown ? id : nil
    }

    static var defaultInput: AudioDeviceID? {
        let id: AudioDeviceID? = value(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultInputDevice)
        return id == kAudioObjectUnknown ? nil : id
    }

    /// Whether any process other than the HAL itself is doing IO on the device.
    static func isRunningSomewhere(_ id: AudioDeviceID) -> Bool {
        let running: UInt32? = value(id, kAudioDevicePropertyDeviceIsRunningSomewhere)
        return (running ?? 0) != 0
    }

    static func inputChannels(_ id: AudioDeviceID) -> Int {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration, mScope: kAudioObjectPropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size) == noErr, size > 0 else { return 0 }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: 16)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, raw) == noErr else { return 0 }
        let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return list.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    /// Calls `handler` on the main queue whenever the property changes; returns a token that
    /// removes the listener when released.
    static func observe(
        _ object: AudioObjectID, _ selector: AudioObjectPropertySelector, handler: @escaping @MainActor () -> Void
    ) -> Listener {
        Listener(object: object, selector: selector, handler: handler)
    }

    final class Listener {
        private let object: AudioObjectID
        private var address: AudioObjectPropertyAddress
        private let block: AudioObjectPropertyListenerBlock

        init(object: AudioObjectID, selector: AudioObjectPropertySelector, handler: @escaping @MainActor () -> Void) {
            self.object = object
            address = AudioObjectPropertyAddress(
                mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
            block = { _, _ in MainActor.assumeIsolated { handler() } }
            AudioObjectAddPropertyListenerBlock(object, &address, .main, block)
        }

        deinit { AudioObjectRemovePropertyListenerBlock(object, &address, .main, block) }
    }

    private static func value<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> T? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var size = UInt32(MemoryLayout<T>.size)
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<T>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, raw) == noErr else { return nil }
        return raw.load(as: T.self)
    }

    private static func string(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
        let s: Unmanaged<CFString>? = value(object, selector)
        return s?.takeRetainedValue() as String?
    }

    private static func array<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> [T] {
        var address = AudioObjectPropertyAddress(
            mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(object, &address, 0, nil, &size) == noErr else { return [] }
        return [T](unsafeUninitializedCapacity: Int(size) / MemoryLayout<T>.stride) { buffer, count in
            count = AudioObjectGetPropertyData(object, &address, 0, nil, &size, buffer.baseAddress!) == noErr
                ? Int(size) / MemoryLayout<T>.stride : 0
        }
    }
}
