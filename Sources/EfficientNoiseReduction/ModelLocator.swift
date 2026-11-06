import Foundation

/// A model checkpoint on the Hugging Face Hub, pinned to the revision the golden tests
/// were generated from.
public struct HubModel: Sendable, Hashable {
    public let repository: String
    public let revision: String
    /// Environment variable that points at a local copy and takes precedence.
    public let environmentVariable: String
    let files: [String]

    /// `nvidia/Real-time_RE-USE`, under NVIDIA's One-Way Noncommercial License (NSCLv1).
    public static let realtimeReUse = HubModel(
        repository: "nvidia/Real-time_RE-USE", revision: "4c0c186272ca853cd18d832fc5c28777b62f32d1",
        environmentVariable: "REUSE_MODEL_DIR", files: ["config.json", "model.safetensors"])

    /// `nvidia/NVIDIA-NemotronLabs-RE-USE`, under the OpenMDW License 1.1 (commercial use allowed).
    public static let nemotron = HubModel(
        repository: "nvidia/NVIDIA-NemotronLabs-RE-USE", revision: "31f82f2efad1095034492f236e8dac9239ba713b",
        environmentVariable: "NEMOTRON_MODEL_DIR", files: ["config.json", "model.safetensors"])

    var name: String { String(repository.split(separator: "/").last!) }

    /// Where `ModelLocator.download` puts the weights by default.
    public var defaultDirectory: URL {
        URL.applicationSupportDirectory.appending(path: "EfficientNoiseReduction/\(name)/\(revision)")
    }
}

/// Finds or downloads model weights. Neither RE-USE checkpoint is bundled; they are
/// fetched on request.
public enum ModelLocator {
    /// The model's environment variable, `defaultDirectory`, then the Hugging Face cache
    /// (`HF_HUB_CACHE`, `HF_HOME`, `~/.cache/huggingface`).
    public static func locate(_ model: HubModel = .realtimeReUse) throws -> URL {
        let env = ProcessInfo.processInfo.environment
        var candidates: [URL] = []
        if let dir = env[model.environmentVariable] { candidates.append(URL(filePath: dir)) }
        candidates.append(model.defaultDirectory)
        let hub: URL =
            if let c = env["HF_HUB_CACHE"] { URL(filePath: c) }
            else if let h = env["HF_HOME"] { URL(filePath: h).appending(path: "hub") }
            else { URL.homeDirectory.appending(path: ".cache/huggingface/hub") }
        let snapshots = hub.appending(path: "models--\(model.repository.replacingOccurrences(of: "/", with: "--"))/snapshots")
        candidates.append(snapshots.appending(path: model.revision))
        if let others = try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil) {
            candidates += others
        }
        for dir in candidates where isComplete(dir, model) { return dir }
        throw EnhancementError.modelNotFound(
            "\(model.repository) weights not found; run `noise-reduction download` or set \(model.environmentVariable)")
    }

    static func isComplete(_ dir: URL, _ model: HubModel) -> Bool {
        model.files.allSatisfy { FileManager.default.fileExists(atPath: dir.appending(path: $0).path) }
    }

    /// Download the pinned revision into `directory` (default: `model.defaultDirectory`),
    /// skipping files already there.
    public static func download(_ model: HubModel = .realtimeReUse, to directory: URL? = nil) async throws -> URL {
        let directory = directory ?? model.defaultDirectory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for name in model.files {
            let dest = directory.appending(path: name)
            if FileManager.default.fileExists(atPath: dest.path) { continue }
            let url = URL(string: "https://huggingface.co/\(model.repository)/resolve/\(model.revision)/\(name)")!
            let (tmp, response) = try await URLSession.shared.download(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw EnhancementError.modelNotFound("download of \(name) failed: \(response)")
            }
            try FileManager.default.moveItem(at: tmp, to: dest)
        }
        return directory
    }
}
