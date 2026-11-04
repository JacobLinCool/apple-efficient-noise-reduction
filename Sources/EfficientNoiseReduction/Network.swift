import Foundation
import MLX

func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

/// Real-time RE-USE (NVIDIA, arXiv:2606.25621): a causal USEMamba with parallel input
/// convs for 0–2 look-ahead frames and a decoder per early-exit depth.
///
/// Only the parts one (exit layer, look-ahead) setting needs are loaded, which is the
/// deployment the paper describes in §IV-F.
final class RealtimeNetwork: @unchecked Sendable {
    let exitLayer: Int
    let lookAhead: Int
    let convs: ConvStack  // encoder and decoders, MPSGraph
    let blocks: [TFBlock]  // time-frequency Mamba blocks, MLX
    /// Storage type inside the Mamba blocks (GEMMs, scan inputs): float16 for `.fast`,
    /// float32 for `.exact`. The residual stream between blocks is always float32.
    let dtype: DType

    init(weights w: Weights, config: ModelConfig, exitLayer: Int, lookAhead: Int, precision: ReUseModel.Precision) throws {
        guard (1 ... config.model.numTFMamba).contains(exitLayer) else {
            throw EnhancementError.invalidConfiguration("exit layer must be in 1...\(config.model.numTFMamba)")
        }
        guard (0 ... 2).contains(lookAhead) else {
            throw EnhancementError.invalidConfiguration("look-ahead must be 0, 1 or 2 frames")
        }
        self.exitLayer = exitLayer
        self.lookAhead = lookAhead
        convs = try ConvStack(weights: w, exitLayer: exitLayer, lookAhead: lookAhead, precision: precision)
        dtype = precision == .fast ? .float16 : .float32
        blocks = try (0 ..< exitLayer).map { [dtype] in try TFBlock(w, prefix: "TSMamba.\($0)", dtype: dtype) }
    }

    /// Everything a stream carries between calls. Starts as zeros, which is what the
    /// offline model's causal padding provides at the start of an utterance.
    struct State {
        var convs: ConvStack.State
        var mamba: [MambaGroup.State]

        var arrays: [MLXArray] { mamba.flatMap { [$0.conv, $0.ssm] } }
    }

    func makeState(bins F: Int) -> State {
        let inner = (F + 2 - 3) / 2 + 1
        return State(
            convs: convs.makeState(bins: F),
            mamba: blocks.map { b in
                MambaGroup.State(
                    conv: MLXArray.zeros([b.time.kConv - 1, inner, b.time.dInner], dtype: dtype),
                    ssm: MLXArray.zeros([inner, b.time.dInner, b.time.dState]))
            })
    }

    /// Compressed magnitude and phase (T, F) in, the same out, delayed by `lookAhead`
    /// frames. Returns nil until the first output frame exists.
    ///
    /// Encoder (MPSGraph) → Mamba blocks (one compiled MLX graph per chunk shape) →
    /// decoders (MPSGraph).
    func callAsFunction(mag: MLXArray, phase: MLXArray, state: inout State?) -> (mag: MLXArray, phase: MLXArray)? {
        let (T, F) = (mag.dim(0), mag.dim(1))
        var s = state ?? makeState(bins: F)
        defer { state = s }
        // NVIDIA appends two zero bins before encoding ("prevent unpredictable errors").
        let features = padded(stacked([mag, phase], axis: -1), widths: [0, [0, 2], 0]).asArray(Float.self)
        guard let encoded = convs.encode(features, frames: T, bins: F, state: &s.convs) else { return nil }
        let out = mamba([encoded] + s.arrays)
        eval(out)
        s.mamba = stride(from: 1, to: out.count, by: 2).map { MambaGroup.State(conv: out[$0], ssm: out[$0 + 1]) }
        return convs.decode(out[0], bins: F, state: &s.convs)
    }

    private lazy var mamba: ([MLXArray]) -> [MLXArray] = compile { [unowned self] a in mambaBody(a) }

    private func mambaBody(_ a: [MLXArray]) -> [MLXArray] {
        var x = a[0]
        var states: [MLXArray] = []
        for (i, block) in blocks.enumerated() {
            var st = MambaGroup.State(conv: a[1 + 2 * i], ssm: a[2 + 2 * i])
            x = block(x, state: &st)
            states += [st.conv, st.ssm]
        }
        return [x.asType(.float32)] + states
    }
}

/// One time-frequency Mamba block: causal Mamba over time, then bidirectional over
/// frequency, each wrapped as `x + LayerNorm(output_proj(mamba(x) + x))`.
///
/// out_proj and output_proj are folded at load time, since nothing nonlinear sits
/// between them: `(y·W_out + x)·W_p = y·(W_out·W_p) + x·W_p`.
struct TFBlock {
    let time: MambaGroup
    let timeOut: Projection
    let freq: MambaGroup
    let freqOut: Projection

    struct Projection {
        let fromScan: MLXArray  // (dirs·Di, C): out_proj of each direction times output_proj
        let fromInput: MLXArray  // (C, C): output_proj summed over directions
        let bias: MLXArray
        let normWeight: MLXArray
        let normBias: MLXArray

        init(_ w: Weights, prefix p: String, outProj: [MLXArray], dtype: DType) throws {
            let wp = try w.linear("\(p).output_proj.weight")  // (dirs·C, C)
            let c = wp.dim(1)
            let parts = outProj.indices.map { wp[($0 * c) ..< (($0 + 1) * c)] }
            fromScan = concatenated(zip(outProj, parts).map { matmul($0, $1) }).asType(dtype)
            fromInput = parts.dropFirst().reduce(parts[0], +).asType(dtype)
            bias = try w.raw("\(p).output_proj.bias").asType(dtype)
            normWeight = try w.raw("\(p).norm.weight").asType(dtype)
            normBias = try w.raw("\(p).norm.bias").asType(dtype)
        }

        func callAsFunction(scan y: MLXArray, input x: MLXArray) -> MLXArray {
            let rows = x.reshaped(-1, x.dim(-1))
            // The residual stream stays float32; only the GEMMs run in the storage type.
            let p = addMM(addMM(bias, rows.asType(fromInput.dtype), fromInput), y.reshaped(rows.dim(0), -1), fromScan)
            return ResidualNorm.run(x: rows, p: p, weight: normWeight, bias: normBias).reshaped(x.shape)
        }
    }

    /// `x + LayerNorm(p)` over 64 channels in one Metal kernel: one SIMD group per row, two
    /// channels per lane, statistics in float32. MLX's LayerNorm is slow on 64-wide rows,
    /// and fusing the residual saves another pass over memory.
    enum ResidualNorm {
        private static let kernel = MLXFast.metalKernel(
            name: "reuse_residual_norm",
            inputNames: ["x", "p", "gamma", "beta", "rows"], outputNames: ["out"],
            source: """
                uint lane = thread_position_in_grid.x, row = thread_position_in_grid.y;
                if (int(row) >= rows[0]) return;
                uint i = row * 64 + lane * 2;
                float a0 = float(p[i]), a1 = float(p[i + 1]);
                float mean = simd_sum(a0 + a1) / 64.0f;
                float d0 = a0 - mean, d1 = a1 - mean;
                float inv = metal::rsqrt(simd_sum(d0 * d0 + d1 * d1) / 64.0f + 1e-5f);
                out[i] = OutT(float(x[i]) + d0 * inv * float(gamma[lane * 2]) + float(beta[lane * 2]));
                out[i + 1] = OutT(float(x[i + 1]) + d1 * inv * float(gamma[lane * 2 + 1]) + float(beta[lane * 2 + 1]));
            """)

        static func run(x: MLXArray, p: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
            precondition(x.dim(-1) == 64)
            let rows = x.dim(0)
            return kernel(
                [x, p, weight, bias, MLXArray([Int32(rows)])], template: [("OutT", x.dtype)], grid: (32, rows, 1),
                threadGroup: (32, 8, 1), outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
        }
    }

    init(_ w: Weights, prefix p: String, dtype: DType) throws {
        time = try MambaGroup(w, prefixes: ["\(p).time_mamba.forward_blocks"], dtype: dtype)
        timeOut = try Projection(w, prefix: "\(p).time_mamba", outProj: time.outProj, dtype: dtype)
        freq = try MambaGroup(
            w, prefixes: ["\(p).freq_mamba.forward_blocks", "\(p).freq_mamba.backward_blocks"], dtype: dtype)
        freqOut = try Projection(w, prefix: "\(p).freq_mamba", outProj: freq.outProj, dtype: dtype)
    }

    func callAsFunction(_ x: MLXArray, state: inout MambaGroup.State) -> MLXArray {
        let x = timeOut(scan: time.time(x, state: &state), input: x)
        return freqOut(scan: freq.freq(x), input: x)
    }
}
