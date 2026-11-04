import Foundation

/// Minimal safetensors reader for float32 and int32 tensors, without MLX, so the CPU
/// models never start Metal.
struct SafeTensors {
    struct Tensor {
        let shape: [Int]
        let values: [Float]
    }

    private(set) var tensors: [String: Tensor] = [:]

    init(url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let headerSize = data.prefix(8).withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self).littleEndian) }
        guard
            let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8 ..< 8 + headerSize))
                as? [String: Any]
        else { throw EnhancementError.invalidConfiguration("bad safetensors header in \(url.lastPathComponent)") }
        let base = 8 + headerSize
        for (name, entry) in header where name != "__metadata__" {
            guard let e = entry as? [String: Any], let dtype = e["dtype"] as? String,
                let shape = e["shape"] as? [Int], let offsets = e["data_offsets"] as? [Int]
            else { continue }
            let bytes = data.subdata(in: base + offsets[0] ..< base + offsets[1])
            let values: [Float] =
                switch dtype {
                case "F32": bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                case "I32": bytes.withUnsafeBytes { $0.bindMemory(to: Int32.self).map(Float.init) }
                default: throw EnhancementError.invalidConfiguration("unsupported dtype \(dtype) for \(name)")
                }
            tensors[name] = Tensor(shape: shape, values: values)
        }
    }

    func callAsFunction(_ name: String) throws -> Tensor {
        guard let t = tensors[name] else { throw EnhancementError.missingWeight(name) }
        return t
    }
}
