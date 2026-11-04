import Foundation
import MLX

// Activations are (time, freq, channels) without a batch axis.

/// A copy of `x` in its own buffer. Streaming state is a few frames sliced from a
/// chunk-sized activation; kept as a view, it would hold the whole activation alive
/// until the next call.
func detached(_ x: MLXArray) -> MLXArray { x * 1 }

/// x @ w + b for a (T, F, in) activation, as one GEMM over all T·F positions.
func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray? = nil) -> MLXArray {
    let rows = x.reshaped(-1, x.dim(-1))
    let y = if let b { addMM(b, rows, w) } else { matmul(rows, w) }
    return y.reshaped(x.dim(0), x.dim(1), w.dim(1))
}
