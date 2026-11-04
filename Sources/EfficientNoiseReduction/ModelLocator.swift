import Foundation

/// Finds or downloads the `nvidia/Real-time_RE-USE` checkpoint.
///
/// The weights are under NVIDIA's One-Way Noncommercial License (NSCLv1); they are never
/// bundled, only fetched on request.
public enum ModelLocator {
    public static let repository = "nvidia/Real-time_RE-USE"
    /// The revision the golden tests were generated from.
    public static let revision = "4c0c186272ca853cd18d832fc5c28777b62f32d1"
    static let files = ["config.json", "model.safetensors"]

    /// Where `download(to:)` puts the weights by default.
    public static var defaultDirectory: URL {
        URL.applicationSupportDirectory.appending(path: "EfficientNoiseReduction/Real-time_RE-USE/\(revision)")
    }

    /// `REUSE_MODEL_DIR`, `defaultDirectory`, then the Hugging Face cache (`HF_HUB_CACHE`,
    /// `HF_HOME`, `~/.cache/huggingface`).
    public static func locate() throws -> URL {
        let env = ProcessInfo.processInfo.environment
        var candidates: [URL] = []
        if let dir = env["REUSE_MODEL_DIR"] { candidates.append(URL(filePath: dir)) }
        candidates.append(defaultDirectory)
        let hub: URL =
            if let c = env["HF_HUB_CACHE"] { URL(filePath: c) }
            else if let h = env["HF_HOME"] { URL(filePath: h).appending(path: "hub") }
            else { URL.homeDirectory.appending(path: ".cache/huggingface/hub") }
        let snapshots = hub.appending(path: "models--nvidia--Real-time_RE-USE/snapshots")
        candidates.append(snapshots.appending(path: revision))
        if let others = try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil) {
            candidates += others
        }
        for dir in candidates where isComplete(dir) { return dir }
        throw EnhancementError.modelNotFound(
            "Real-time RE-USE weights not found; run `noise-reduction download` or set REUSE_MODEL_DIR")
    }

    static func isComplete(_ dir: URL) -> Bool {
        files.allSatisfy { FileManager.default.fileExists(atPath: dir.appending(path: $0).path) }
    }

    /// Download the pinned revision into `directory` (skipping files already there).
    public static func download(to directory: URL) async throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for name in files {
            let dest = directory.appending(path: name)
            if FileManager.default.fileExists(atPath: dest.path) { continue }
            let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(name)")!
            let (tmp, response) = try await URLSession.shared.download(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw EnhancementError.modelNotFound("download of \(name) failed: \(response)")
            }
            try FileManager.default.moveItem(at: tmp, to: dest)
        }
        return directory
    }
}
