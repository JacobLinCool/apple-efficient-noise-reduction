import AVFoundation
import CoreAudio
import EfficientNoiseReduction
import Observation
import ServiceManagement
import os

enum EnhancementModel: String, CaseIterable, Identifiable {
    case deepFilterNet3, reuse, nemotron
    var id: Self { self }

    var title: String {
        switch self {
        case .deepFilterNet3: "DeepFilterNet3"
        case .reuse: "Real-time RE-USE"
        case .nemotron: "NemotronLabs RE-USE"
        }
    }

    /// Weights downloaded on request; DeepFilterNet3's ship with the app.
    var hub: HubModel? {
        switch self {
        case .deepFilterNet3: nil
        case .reuse: .realtimeReUse
        case .nemotron: .nemotron
        }
    }
}

/// App state: settings, the virtual microphone, and the pipeline that runs while some
/// app is recording from that microphone.
@MainActor @Observable
final class Controller {
    enum Status: Equatable {
        case driverMissing
        case off
        case waiting
        case starting
        case active(input: String)
        case failed(String)
    }

    var enabled: Bool { didSet { save(); update(restart: false) } }
    /// `nil` follows the system default input.
    var inputUID: String? { didSet { save(); update(restart: true) } }
    var model: EnhancementModel { didSet { save(); update(restart: true); preload() } }
    /// RE-USE early-exit depth.
    var reuseDepth: Int { didSet { save(); update(restart: true); preload() } }
    /// RE-USE frames per network call (20 ms each).
    var reuseFrames: Int { didSet { save(); update(restart: true); preload() } }
    /// NemotronLabs RE-USE chunk length in seconds: its latency.
    var nemotronChunk: Double { didSet { save(); update(restart: true); preload() } }

    private(set) var status: Status = .off {
        didSet { if status != oldValue { log.notice("status: \(String(describing: self.status), privacy: .public)") } }
    }
    private(set) var inputs: [AudioDevice] = []
    /// Models whose weights are on disk.
    private(set) var downloaded: Set<EnhancementModel> = [.deepFilterNet3]
    private(set) var downloading: EnhancementModel?
    private(set) var driverOutdated = false
    /// Both devices of the plug-in are present.
    private(set) var driverPresent = false
    private(set) var latencyMs: Int?
    var launchAtLogin = SMAppService.mainApp.status == .enabled {
        didSet {
            guard launchAtLogin != (SMAppService.mainApp.status == .enabled) else { return }
            try? launchAtLogin ? SMAppService.mainApp.register() : SMAppService.mainApp.unregister()
            launchAtLogin = SMAppService.mainApp.status == .enabled
        }
    }

    private var pipeline: Pipeline?
    private var startTask: Task<Void, Never>?
    /// Bumped whenever a setting invalidates the running pipeline.
    private var generation = 0
    private var listeners: [AudioSystem.Listener] = []
    private var microphoneListener: AudioSystem.Listener?
    private var observedMicrophone: AudioDeviceID?
    private let models = ModelCache()
    private let defaults = UserDefaults.standard

    init() {
        enabled = defaults.object(forKey: "enabled") as? Bool ?? true
        inputUID = defaults.string(forKey: "inputUID")
        model = defaults.string(forKey: "model").flatMap(EnhancementModel.init) ?? .deepFilterNet3
        reuseDepth = defaults.object(forKey: "reuseDepth") as? Int ?? 8
        reuseFrames = defaults.object(forKey: "reuseFrames") as? Int ?? 1
        nemotronChunk = defaults.object(forKey: "nemotronChunk") as? Double ?? 1
        let system = AudioObjectID(kAudioObjectSystemObject)
        listeners = [
            AudioSystem.observe(system, kAudioHardwarePropertyDevices) { [weak self] in self?.update(restart: false) },
            AudioSystem.observe(system, kAudioHardwarePropertyDefaultInputDevice) { [weak self] in
                self?.update(restart: self?.inputUID == nil)
            },
        ]
        update(restart: false)
        preload()
    }

    var defaultInputName: String? {
        resolvedInput().map(\.name)
    }

    // MARK: - Actions

    func installDriver() {
        do {
            try DriverInstaller.install()
            // Core Audio restarts; the device list listener picks the new devices up.
            Task {
                try? await Task.sleep(for: .seconds(2))
                update(restart: true)
            }
        } catch {
            status = .failed(error.localizedDescription)
        }
    }

    func uninstallDriver() {
        stopPipeline()
        do { try DriverInstaller.uninstall() } catch { status = .failed(error.localizedDescription) }
    }

    func download(_ model: EnhancementModel) {
        guard downloading == nil, let hub = model.hub else { return }
        downloading = model
        Task {
            defer { downloading = nil }
            do {
                _ = try await ModelLocator.download(hub)
                update(restart: true)
                preload()
            } catch {
                status = .failed("Download failed: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - State

    /// Re-reads the devices and starts or stops the pipeline to match the settings.
    private func update(restart: Bool) {
        inputs = AudioSystem.inputDevices()
        downloaded = Set(EnhancementModel.allCases.filter { m in m.hub.map { (try? ModelLocator.locate($0)) != nil } ?? true })
        driverOutdated = DriverInstaller.isOutdated

        let microphone = AudioSystem.device(uid: VirtualDevice.microphoneUID)
        if microphone != observedMicrophone {
            observedMicrophone = microphone
            microphoneListener = microphone.map {
                AudioSystem.observe($0, kAudioDevicePropertyDeviceIsRunningSomewhere) { [weak self] in
                    self?.update(restart: false)
                }
            }
        }
        let sink = AudioSystem.device(uid: VirtualDevice.sinkUID)
        driverPresent = microphone != nil && sink != nil
        guard let microphone, let sink else {
            stopPipeline()
            status = .driverMissing
            return
        }
        guard enabled else {
            stopPipeline()
            status = .off
            return
        }
        guard AudioSystem.isRunningSomewhere(microphone) else {
            stopPipeline()
            status = .waiting
            return
        }
        if restart {
            generation += 1
            stopPipeline()
        }
        if pipeline == nil && startTask == nil { startPipeline(sink: sink) }
    }

    private func resolvedInput() -> AudioDevice? {
        if let inputUID, let device = inputs.first(where: { $0.uid == inputUID }) { return device }
        // The system default, unless that is our own microphone.
        if let id = AudioSystem.defaultInput, let device = inputs.first(where: { $0.id == id }) { return device }
        return inputs.first
    }

    private func startPipeline(sink: AudioDeviceID) {
        guard let input = resolvedInput() else {
            status = .failed("No microphone found.")
            return
        }
        status = .starting
        let model = model, settings = settings, models = models, generation = generation
        startTask = Task {
            guard await AVCaptureDevice.requestAccess(for: .audio) else {
                startTask = nil
                status = .failed("Microphone access is off. Allow it in System Settings › Privacy & Security.")
                return
            }
            let begin = ContinuousClock.now
            let result = await Task.detached(priority: .userInitiated) {
                Result {
                    let enhancer = try models.enhancer(model, settings)
                    return try Pipeline(enhancer: enhancer, margin: settings.margin(model), input: input.id, sink: sink)
                }
            }.value
            log.notice("pipeline start took \(String(describing: ContinuousClock.now - begin), privacy: .public)")
            startTask = nil
            switch result {
            case .success(let started) where generation != self.generation:
                // Settings changed while loading.
                started.stop()
                update(restart: false)
            case .success(let started):
                pipeline = started
                latencyMs = Int((started.realtime.latencySeconds * 1000).rounded())
                status = .active(input: input.name)
                // The microphone may have stopped while loading.
                update(restart: false)
            case .failure(let error):
                status = .failed(error.localizedDescription)
            }
        }
    }

    /// Input and output levels of the running pipeline (RMS, 0 when idle).
    var levels: (input: Float, output: Float) {
        guard let levels = pipeline?.levels else { return (0, 0) }
        return (levels.input, levels.output)
    }

    /// Loads the selected model and compiles its kernels ahead of time, so the pipeline
    /// starts within a fraction of a second when a meeting app opens the microphone.
    private func preload() {
        let model = model, settings = settings, models = models
        guard downloaded.contains(model) else { return }
        Task.detached(priority: .utility) {
            try? models.enhancer(model, settings).warmUp()
        }
    }

    private func stopPipeline() {
        if let pipeline {
            log.notice("pipeline stopped: \(pipeline.realtime.underruns) underruns, \(pipeline.realtime.overruns) overruns")
        }
        pipeline?.stop()
        pipeline = nil
        latencyMs = nil
    }

    private func save() {
        defaults.set(enabled, forKey: "enabled")
        defaults.set(inputUID, forKey: "inputUID")
        defaults.set(model.rawValue, forKey: "model")
        defaults.set(reuseDepth, forKey: "reuseDepth")
        defaults.set(reuseFrames, forKey: "reuseFrames")
        defaults.set(nemotronChunk, forKey: "nemotronChunk")
    }

    private var settings: ModelSettings {
        ModelSettings(reuseDepth: reuseDepth, reuseFrames: reuseFrames, nemotronChunk: nemotronChunk)
    }
}

/// The per-model settings a pipeline is built from.
struct ModelSettings: Sendable {
    var reuseDepth: Int
    var reuseFrames: Int
    var nemotronChunk: Double

    /// Slack for compute time before playback would run dry (see `RealtimeEnhancer`).
    /// DeepFilterNet3 computes a frame in well under a millisecond on the CPU; RE-USE's
    /// GPU calls take a few and jitter more; NemotronLabs RE-USE computes a whole chunk at
    /// once, a few hundred milliseconds per second of 48 kHz audio at most.
    func margin(_ model: EnhancementModel) -> Double {
        switch model {
        case .deepFilterNet3: 0.015
        case .reuse: 0.03
        case .nemotron: 0.15 + 0.25 * nemotronChunk
        }
    }
}

private let log = Logger(subsystem: "cool.jacoblin.EfficientNoiseReduction", category: "controller")

/// Loaded weights, kept across pipeline restarts.
private final class ModelCache: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock<(DeepFilterNet3?, ReUseModel?, NemotronModel?)>(initialState: (nil, nil, nil))

    func enhancer(_ model: EnhancementModel, _ s: ModelSettings) throws -> any SpeechEnhancer {
        try lock.withLockUnchecked { cache in
            switch model {
            case .deepFilterNet3:
                let m = try cache.0 ?? DeepFilterNet3()
                cache.0 = m
                return DeepFilterNet3Enhancer(model: m, chunkFrames: 1)
            case .reuse:
                let m = try cache.1 ?? ReUseModel(directory: ModelLocator.locate(.realtimeReUse))
                cache.1 = m
                return try StreamingEnhancer(
                    model: m, sampleRate: 48000,
                    configuration: .init(exitLayer: s.reuseDepth, lookAhead: 0, chunk: .frames(s.reuseFrames)))
            case .nemotron:
                let m = try cache.2 ?? NemotronModel(directory: ModelLocator.locate(.nemotron))
                cache.2 = m
                return NemotronEnhancer(model: m, sampleRate: 48000, chunking: .chunks(seconds: s.nemotronChunk))
            }
        }
    }
}

#if DEBUG
extension Controller {
    /// Puts the controller in a given state for UI snapshots.
    func preview(status: Status, latencyMs: Int? = nil, driverPresent: Bool = true) {
        self.status = status
        self.latencyMs = latencyMs
        self.driverPresent = driverPresent
        driverOutdated = false
    }
}
#endif
