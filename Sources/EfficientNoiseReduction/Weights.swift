import Foundation
import MLX

public enum EnhancementError: Error, CustomStringConvertible {
    case missingWeight(String)
    case badShape(String, [Int])
    case invalidConfiguration(String)
    case modelNotFound(String)

    public var description: String {
        switch self {
        case .missingWeight(let k): "missing weight \(k)"
        case .badShape(let k, let s): "unexpected shape \(s) for \(k)"
        case .invalidConfiguration(let m): m
        case .modelNotFound(let m): m
        }
    }
}

/// Safetensors state dict with PyTorch-to-MLX layout conversion on access.
struct Weights {
    private let arrays: [String: MLXArray]

    init(url: URL) throws {
        arrays = try loadArrays(url: url)
    }

    func raw(_ key: String) throws -> MLXArray {
        guard let a = arrays[key] else { throw EnhancementError.missingWeight(key) }
        return a.asType(.float32)
    }

    /// Conv2d weight `(O, I, kH, kW)` as MLX `(O, kH, kW, I)`.
    func conv(_ key: String) throws -> MLXArray {
        let w = try raw(key)
        guard w.ndim == 4 else { throw EnhancementError.badShape(key, w.shape) }
        return w.transposed(0, 2, 3, 1)
    }

    /// A conv whose PyTorch kernel runs over (freq, time) on a (C, F, T) map, as an MLX
    /// `(O, kT, kF, I)` kernel over our (T, F, C) layout.
    func convFT(_ key: String) throws -> MLXArray {
        let w = try raw(key)
        guard w.ndim == 4 else { throw EnhancementError.badShape(key, w.shape) }
        return w.transposed(0, 3, 2, 1)
    }

    /// Linear weight `(out, in)` transposed for `x @ w`.
    func linear(_ key: String) throws -> MLXArray {
        try raw(key).T
    }
}
