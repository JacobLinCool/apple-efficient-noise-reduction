import SwiftUI

@main
struct NoiseReductionApp: App {
    @State private var controller = Controller()

    var body: some Scene {
        MenuBarExtra {
            MenuView(controller: controller)
        } label: {
            Image(systemName: controller.isProcessing ? "waveform.badge.mic" : "mic")
        }
        .menuBarExtraStyle(.window)
    }
}

extension Controller {
    var isProcessing: Bool {
        if case .active = status { true } else { false }
    }
}

struct MenuView: View {
    @Bindable var controller: Controller

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Noise Reduction").font(.headline)
                Spacer()
                Toggle("", isOn: $controller.enabled).toggleStyle(.switch).labelsHidden()
            }
            StatusView(controller: controller)

            if controller.status != .driverMissing {
                Divider()
                Form {
                    Picker("Microphone", selection: $controller.inputUID) {
                        Text("System Default").tag(String?.none)
                        ForEach(controller.inputs) { device in
                            Text(device.name).tag(Optional(device.uid))
                        }
                    }
                    Picker("Model", selection: $controller.model) {
                        ForEach(EnhancementModel.allCases) { Text($0.title).tag($0) }
                    }
                    if controller.model == .reuse {
                        ReUseSettings(controller: controller)
                    }
                }
                .formStyle(.columns)
            }

            Divider()
            HStack {
                Toggle("Open at Login", isOn: $controller.launchAtLogin).toggleStyle(.checkbox)
                Spacer()
                Menu("More") {
                    if DriverInstaller.isInstalled {
                        Button("Reinstall Driver…") { controller.installDriver() }
                        Button("Uninstall Driver…") { controller.uninstallDriver() }
                    }
                    Button("About Models") {
                        NSWorkspace.shared.open(URL(string: "https://github.com/JacobLinCool/apple-efficient-noise-reduction")!)
                    }
                    Divider()
                    Button("Quit Noise Reduction") { NSApplication.shared.terminate(nil) }
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
            }
        }
        .padding(16)
        .frame(width: 340)
    }
}

private struct StatusView: View {
    let controller: Controller

    var body: some View {
        switch controller.status {
        case .driverMissing:
            VStack(alignment: .leading, spacing: 8) {
                Text("Install the virtual microphone to use Noise Reduction in meeting apps.")
                    .font(.callout).foregroundStyle(.secondary)
                Button("Install Virtual Microphone…") { controller.installDriver() }
                    .buttonStyle(.borderedProminent)
            }
        case .off:
            label("Off. “Noise Reduction Microphone” passes silence.", "pause.circle", .secondary)
        case .waiting:
            label("Ready. Choose “Noise Reduction Microphone” in your meeting app.", "checkmark.circle", .secondary)
        case .starting:
            label("Starting…", "hourglass", .secondary)
        case .active(let input):
            let latency = controller.latencyMs.map { " · \($0) ms" } ?? ""
            label("Cleaning \(input)\(latency)", "waveform", .green)
        case .failed(let message):
            label(message, "exclamationmark.triangle", .orange)
        }
        if controller.driverOutdated {
            Button("Update Virtual Microphone…") { controller.installDriver() }
        }
    }

    private func label(_ text: String, _ symbol: String, _ color: Color) -> some View {
        Label {
            Text(text).font(.callout).fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: symbol).foregroundStyle(color)
        }
    }
}

private struct ReUseSettings: View {
    @Bindable var controller: Controller

    var body: some View {
        if controller.reuseDownloaded {
            Picker("Quality", selection: $controller.reuseDepth) {
                Text("Light").tag(3)
                Text("Balanced").tag(8)
                Text("Best").tag(12)
            }
            Picker("Latency", selection: $controller.reuseFrames) {
                Text("40 ms").tag(1)
                Text("120 ms").tag(5)
            }
        } else {
            VStack(alignment: .leading, spacing: 6) {
                Text("RE-USE weights (about 54 MB) are under NVIDIA's noncommercial license and are downloaded separately.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button(controller.downloading ? "Downloading…" : "Download RE-USE Weights") { controller.downloadReUse() }
                    .disabled(controller.downloading)
            }
        }
    }
}
