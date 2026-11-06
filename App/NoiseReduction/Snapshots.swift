#if DEBUG
import AppKit
import SwiftUI

/// `Noise Reduction --snapshots <dir>` renders the panel in each state to PNG files, for
/// reviewing the UI without clicking through the menu bar. Debug builds only.
enum Snapshots {
    @MainActor
    static func runIfRequested() {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--snapshots"), i + 1 < args.count else { return }
        let dir = URL(filePath: args[i + 1])
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let controller = Controller()
        let states: [(String, () -> Void)] = [
            ("1-install", { controller.preview(status: .driverMissing, driverPresent: false) }),
            ("2-standby", { controller.preview(status: .waiting) }),
            ("3-active", { controller.preview(status: .active(input: "MacBook Pro Microphone"), latencyMs: 55) }),
            ("4-reuse", {
                controller.model = .reuse
                controller.preview(status: .active(input: "MacBook Pro Microphone"), latencyMs: 70)
            }),
            ("4b-nemotron", {
                controller.model = .nemotron
                controller.preview(status: .active(input: "MacBook Pro Microphone"), latencyMs: 1400)
            }),
            ("5-off", { controller.enabled = false; controller.preview(status: .off) }),
            ("6-failed", {
                controller.enabled = true
                controller.model = .deepFilterNet3
                controller.preview(status: .failed("Microphone access is off. Allow it in System Settings › Privacy & Security."))
            }),
        ]
        for appearance in [NSAppearance.Name.aqua, .darkAqua] {
            for (name, apply) in states {
                apply()
                let view = NSHostingView(rootView: MenuView(controller: controller).background(.background))
                view.appearance = NSAppearance(named: appearance)
                let size = view.fittingSize
                view.frame = CGRect(origin: .zero, size: size)
                let window = NSWindow(contentRect: view.frame, styleMask: .borderless, backing: .buffered, defer: false)
                window.contentView = view
                view.layoutSubtreeIfNeeded()
                RunLoop.main.run(until: .now + 0.3)
                guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { continue }
                view.cacheDisplay(in: view.bounds, to: rep)
                let suffix = appearance == .aqua ? "light" : "dark"
                try? rep.representation(using: .png, properties: [:])?
                    .write(to: dir.appending(path: "\(name)-\(suffix).png"))
            }
        }
        exit(0)
    }
}
#endif
