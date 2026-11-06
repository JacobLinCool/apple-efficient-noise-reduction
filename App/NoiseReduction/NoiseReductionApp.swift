import SwiftUI

@main
struct NoiseReductionApp: App {
    @State private var controller = Controller()

    init() {
        #if DEBUG
        Snapshots.runIfRequested()
        #endif
    }

    var body: some Scene {
        MenuBarExtra {
            MenuView(controller: controller)
        } label: {
            Image(systemName: controller.status.menuBarSymbol)
        }
        .menuBarExtraStyle(.window)
    }
}

extension Controller.Status {
    var isActive: Bool {
        if case .active = self { true } else { false }
    }

    var menuBarSymbol: String {
        switch self {
        case .active: "waveform.circle.fill"
        case .waiting, .starting: "waveform.circle"
        case .off: "mic.slash"
        case .driverMissing, .failed: "exclamationmark.triangle"
        }
    }
}

// MARK: - Panel

struct MenuView: View {
    @Bindable var controller: Controller

    var body: some View {
        VStack(spacing: 0) {
            Header(controller: controller)
            Divider()
            if !controller.driverPresent {
                Onboarding(controller: controller)
            } else {
                SignalFlow(controller: controller)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                Divider()
                SettingsList(controller: controller)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .disabled(!controller.enabled)
            }
        }
        .frame(width: 320)
    }
}

private struct Header: View {
    @Bindable var controller: Controller

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 26))
                .foregroundStyle(controller.status.isActive ? Color.accentColor : .secondary)
                .symbolEffect(.pulse, isActive: controller.status == .starting)
            VStack(alignment: .leading, spacing: 1) {
                Text("Noise Reduction").font(.headline)
                StatusLine(controller: controller)
            }
            Spacer(minLength: 8)
            if controller.driverPresent {
                Toggle("Enabled", isOn: $controller.enabled)
                    .toggleStyle(.switch).labelsHidden().controlSize(.small)
            }
            SettingsMenu(controller: controller)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }
}

private struct StatusLine: View {
    let controller: Controller

    var body: some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(text).font(.caption).foregroundStyle(.secondary).monospacedDigit()
        }
    }

    private var text: String {
        switch controller.status {
        case .driverMissing: "Not installed"
        case .off: "Off"
        case .waiting: "Standby"
        case .starting: "Starting"
        case .active: controller.latencyMs.map { "Active · \(Self.latency($0))" } ?? "Active"
        case .failed: "Needs attention"
        }
    }

    static func latency(_ ms: Int) -> String {
        ms < 1000 ? "\(ms) ms" : String(format: "%.1f s", Double(ms) / 1000)
    }

    private var color: Color {
        switch controller.status {
        case .active: .green
        case .waiting, .starting: .yellow
        case .driverMissing, .failed: .orange
        case .off: .secondary.opacity(0.5)
        }
    }
}

private struct SettingsMenu: View {
    @Bindable var controller: Controller

    var body: some View {
        Menu {
            Toggle("Open at Login", isOn: $controller.launchAtLogin)
            Divider()
            if DriverInstaller.isInstalled {
                Button("Reinstall Virtual Microphone…") { controller.installDriver() }
                Button("Remove Virtual Microphone…") { controller.uninstallDriver() }
                Divider()
            }
            Link("About Noise Reduction", destination: URL(string: "https://github.com/JacobLinCool/apple-efficient-noise-reduction")!)
            Divider()
            Button("Quit") { NSApplication.shared.terminate(nil) }.keyboardShortcut("q")
        } label: {
            Image(systemName: "ellipsis.circle").font(.system(size: 15))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("More")
    }
}

// MARK: - Signal flow

/// Microphone → model → meeting apps, with live levels at both ends.
private struct SignalFlow: View {
    let controller: Controller

    var body: some View {
        let active = controller.status.isActive
        VStack(spacing: 10) {
            TimelineView(.animation(minimumInterval: 1 / 20, paused: !active)) { _ in
                let levels = controller.levels
                HStack(alignment: .top, spacing: 0) {
                    FlowNode(
                        symbol: "mic.fill", title: controller.defaultInputName ?? "No microphone",
                        active: active, level: levels.input)
                    FlowArrow(active: active)
                    FlowNode(
                        symbol: controller.model == .deepFilterNet3 ? "wand.and.stars" : "sparkles",
                        title: controller.model.shortTitle, active: active, level: nil)
                    FlowArrow(active: active)
                    FlowNode(symbol: "person.2.wave.2.fill", title: "Meetings", active: active, level: levels.output)
                }
            }
            Notice(controller: controller)
        }
    }
}

private struct FlowNode: View {
    let symbol: String
    let title: String
    let active: Bool
    let level: Float?

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: symbol)
                .font(.system(size: 17, weight: .medium))
                .foregroundStyle(active ? Color.white : .secondary)
                .frame(width: 44, height: 44)
                .background(Circle().fill(active ? Color.accentColor : Color.secondary.opacity(0.15)))
            Text(title)
                .font(.caption)
                .foregroundStyle(active ? .primary : .secondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .padding(.horizontal, 2)
            LevelMeter(level: level ?? 0)
                .opacity(level == nil ? 0 : 1)
        }
        .frame(maxWidth: .infinity)
    }
}

private struct FlowArrow: View {
    let active: Bool

    var body: some View {
        Image(systemName: "chevron.right")
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(active ? Color.accentColor : Color.secondary.opacity(0.4))
            .symbolEffect(.pulse, isActive: active)
            .frame(height: 44)
    }
}

/// A short horizontal meter, -60 to 0 dBFS.
private struct LevelMeter: View {
    let level: Float

    var body: some View {
        let db = 20 * log10(max(level, 1e-6))
        let fraction = CGFloat(min(max((db + 60) / 60, 0), 1))
        Capsule()
            .fill(Color.secondary.opacity(0.15))
            .overlay(alignment: .leading) {
                Capsule().fill(Color.green).frame(width: 48 * fraction)
            }
            .frame(width: 48, height: 4)
            .animation(.linear(duration: 0.05), value: fraction)
    }
}

/// The one line of guidance the current state needs, if any.
private struct Notice: View {
    let controller: Controller

    var body: some View {
        switch controller.status {
        case .waiting:
            note("Choose “Noise Reduction Microphone” in your meeting app.", "info.circle", .secondary)
        case .failed(let message):
            note(message, "exclamationmark.triangle.fill", .orange)
        default:
            if controller.driverOutdated {
                Button("Update Virtual Microphone…") { controller.installDriver() }
                    .controlSize(.small)
            }
        }
    }

    private func note(_ text: String, _ symbol: String, _ tint: Color) -> some View {
        Label {
            Text(text).fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: symbol).foregroundStyle(tint)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Settings

private struct SettingsList: View {
    @Bindable var controller: Controller

    var body: some View {
        VStack(spacing: 10) {
            SettingRow(symbol: "mic", help: "Microphone") {
                Picker("Microphone", selection: $controller.inputUID) {
                    Text("System Default").tag(String?.none)
                    Divider()
                    ForEach(controller.inputs) { Text($0.name).tag(Optional($0.uid)) }
                }
            }
            SettingRow(symbol: "cpu", help: "Model") {
                Picker("Model", selection: $controller.model) {
                    ForEach(EnhancementModel.allCases) { model in
                        Text(model.shortTitle).tag(model).help(model.detail)
                    }
                }
                .pickerStyle(.segmented)
            }
            if !controller.downloaded.contains(controller.model) {
                SettingRow(symbol: "arrow.down.circle", help: "Model weights") {
                    HStack {
                        Button(controller.downloading == controller.model ? "Downloading…" : "Download Model · \(controller.model.downloadSize)") {
                            controller.download(controller.model)
                        }
                        .disabled(controller.downloading != nil)
                        Spacer()
                        LicenseInfo(model: controller.model)
                    }
                }
            } else if controller.model == .reuse {
                SettingRow(symbol: "gauge.with.dots.needle.67percent", help: "Quality: deeper exits sound better and use more GPU") {
                    Picker("Quality", selection: $controller.reuseDepth) {
                        Text("Light").tag(3)
                        Text("Balanced").tag(8)
                        Text("Best").tag(12)
                    }
                    .pickerStyle(.segmented)
                }
                SettingRow(symbol: "timer", help: "Latency: longer chunks use less power") {
                    Picker("Latency", selection: $controller.reuseFrames) {
                        Text("40 ms").tag(1)
                        Text("120 ms").tag(5)
                    }
                    .pickerStyle(.segmented)
                }
            } else if controller.model == .nemotron {
                SettingRow(symbol: "timer", help: "Latency: the model enhances a chunk at a time; longer chunks give it more context") {
                    Picker("Latency", selection: $controller.nemotronChunk) {
                        Text("1 s").tag(1.0)
                        Text("2 s").tag(2.0)
                    }
                    .pickerStyle(.segmented)
                }
            }
        }
        .labelsHidden()
        .controlSize(.regular)
    }
}

/// A leading symbol and one control filling the rest of the row.
private struct SettingRow<Content: View>: View {
    let symbol: String
    let help: String
    @ViewBuilder let content: Content

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: symbol)
                .foregroundStyle(.secondary)
                .frame(width: 20)
                .help(help)
            content.frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct LicenseInfo: View {
    let model: EnhancementModel
    @State private var shown = false

    var body: some View {
        Button { shown.toggle() } label: { Image(systemName: "info.circle") }
            .buttonStyle(.borderless)
            .popover(isPresented: $shown, arrowEdge: .bottom) {
                Text(model.licenseNote)
                    .font(.callout)
                    .frame(width: 220)
                    .padding(12)
            }
    }
}

// MARK: - Onboarding

private struct Onboarding: View {
    let controller: Controller

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "mic.badge.plus")
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(Color.accentColor)
                .padding(.top, 6)
            VStack(spacing: 4) {
                Text("Add the virtual microphone").font(.headline)
                Text("Meeting apps hear your voice, not the room.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Button("Install…") { controller.installDriver() }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            if case .failed(let message) = controller.status {
                Text(message).font(.caption).foregroundStyle(.orange)
            }
        }
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity)
        .padding(20)
    }
}

extension EnhancementModel {
    var shortTitle: String {
        switch self {
        case .deepFilterNet3: "DFN3"
        case .reuse: "RE-USE"
        case .nemotron: "Nemotron"
        }
    }

    var detail: String {
        switch self {
        case .deepFilterNet3: "DeepFilterNet3: noise suppression on the CPU, built in"
        case .reuse: "Real-time RE-USE: universal speech enhancement on the GPU, 40 ms"
        case .nemotron: "NemotronLabs RE-USE: NVIDIA's highest-quality enhancement on the GPU, a second or more of delay"
        }
    }

    var downloadSize: String {
        switch self {
        case .deepFilterNet3: ""
        case .reuse: "54 MB"
        case .nemotron: "39 MB"
        }
    }

    var licenseNote: String {
        switch self {
        case .deepFilterNet3: "DeepFilterNet3 is MIT / Apache-2.0 licensed and built in."
        case .reuse: "NVIDIA's Real-time RE-USE weights are licensed for noncommercial use only and are downloaded from Hugging Face."
        case .nemotron: "NVIDIA's NemotronLabs RE-USE weights are released under the OpenMDW License 1.1, which allows commercial use, and are downloaded from Hugging Face."
        }
    }
}
