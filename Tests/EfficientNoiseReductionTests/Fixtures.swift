import Foundation
import XCTest

/// Golden data generated locally by `make golden` (from NVIDIA's and DeepFilterNet's
/// reference code). It is not part of the repository; tests that need it skip without it.
enum Fixtures {
    static let directory = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "Fixtures")

    static func url(_ name: String) throws -> URL {
        let url = directory.appending(path: "\(name).safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("fixture \(name) missing; run `make golden`")
        }
        return url
    }
}
