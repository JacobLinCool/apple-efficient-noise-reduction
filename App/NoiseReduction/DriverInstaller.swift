import Foundation

/// Copies the bundled Audio Server plug-in into the system HAL folder (needs an
/// administrator password) and restarts Core Audio so the virtual microphone appears.
enum DriverInstaller {
    static let halDirectory = URL(filePath: "/Library/Audio/Plug-Ins/HAL")

    static var bundled: URL? {
        Bundle.main.resourceURL.map { $0.appending(path: VirtualDevice.driverName) }
    }

    static var installed: URL { halDirectory.appending(path: VirtualDevice.driverName) }

    static var isInstalled: Bool { FileManager.default.fileExists(atPath: installed.path) }

    /// Whether the installed plug-in is a different build than the one in this app.
    static var isOutdated: Bool {
        guard isInstalled, let bundled else { return false }
        return version(of: installed) != version(of: bundled)
    }

    static func install() throws {
        guard let bundled, FileManager.default.fileExists(atPath: bundled.path) else {
            throw InstallError("The driver is missing from the app bundle.")
        }
        try runAsAdministrator(
            "rm -rf \(quoted(installed.path)) && cp -R \(quoted(bundled.path)) \(quoted(halDirectory.path))/ && killall coreaudiod")
    }

    static func uninstall() throws {
        try runAsAdministrator("rm -rf \(quoted(installed.path)) && killall coreaudiod")
    }

    private static func version(of bundle: URL) -> String? {
        Bundle(url: bundle)?.infoDictionary?["CFBundleVersion"] as? String
    }

    private static func quoted(_ path: String) -> String {
        "'" + path.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    private static func runAsAdministrator(_ command: String) throws {
        let escaped = command.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
        var error: NSDictionary?
        NSAppleScript(source: "do shell script \"\(escaped)\" with administrator privileges")?.executeAndReturnError(&error)
        if let error {
            // -128: the user cancelled the password prompt.
            if error[NSAppleScript.errorNumber] as? Int == -128 { return }
            throw InstallError(error[NSAppleScript.errorMessage] as? String ?? "Installation failed.")
        }
    }

    struct InstallError: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
