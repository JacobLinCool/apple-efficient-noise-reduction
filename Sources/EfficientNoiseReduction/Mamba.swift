import Foundation
import MLX

/// Mamba v1 (`mamba_ssm.Mamba`) on a (T, F, C) map, run along time or frequency in
/// place, so neither direction needs a transpose:
///
///     x, z     = in_proj(u)
///     x        = silu(depthwise_conv1d(x))         causal (anti-causal when reversed)
///     dt, B, C = x_proj(x)
///     delta    = softplus(dt_proj(dt))
///     h_l      = exp(delta_l A) h_{l-1} + delta_l B_l x_l,   A = -exp(A_log)
///     y_l      = <C_l, h_l> + D x_l
///     out      = out_proj(y * silu(z))
///
/// `MambaGroup` holds one or two Mambas that read the same input: the causal time Mamba
/// of a block, or the forward and backward frequency Mambas, which run as one batch.
/// Per call that is two GEMMs and two Metal kernels per group, plus the output GEMM that
/// the caller folds into its own projection.
struct MambaGroup {
    let directions: Int
    let dInner: Int
    let dtRank: Int
    let dState: Int
    let kConv: Int

    let inProj: MLXArray  // (C, dirs·2·Di)
    let convWeight: MLXArray  // (dirs, Di, K)
    let convBias: MLXArray  // (dirs, Di)
    let xProj: MLXArray  // (dirs, Di, R + 2N)
    let dtProj: MLXArray  // (dirs, R, Di)
    let dtBias: MLXArray  // (dirs, Di)
    let A: MLXArray  // (dirs, Di, N), -exp(A_log)
    let D: MLXArray  // (dirs, Di)
    /// out_proj of each direction, (Di, C); the caller folds it into its projection.
    let outProj: [MLXArray]

    /// `dtype` is the storage type of activations and GEMM weights; the per-channel
    /// scan parameters and the SSM state stay float32.
    init(_ w: Weights, prefixes: [String], dtype: DType) throws {
        func stack(_ f: (String) throws -> MLXArray) throws -> MLXArray { stacked(try prefixes.map(f)) }
        directions = prefixes.count
        inProj = concatenated(try prefixes.map { try w.linear("\($0).in_proj.weight") }, axis: 1).asType(dtype)
        convWeight = try stack { try w.raw("\($0).conv1d.weight").squeezed(axis: 1) }
        convBias = try stack { try w.raw("\($0).conv1d.bias") }
        xProj = try stack { try w.linear("\($0).x_proj.weight") }.asType(dtype)
        dtProj = try stack { try w.linear("\($0).dt_proj.weight") }
        dtBias = try stack { try w.raw("\($0).dt_proj.bias") }
        A = try stack { -exp(try w.raw("\($0).A_log")) }
        D = try stack { try w.raw("\($0).D") }
        outProj = try prefixes.map { try w.linear("\($0).out_proj.weight") }
        dInner = A.dim(1)
        dState = A.dim(2)
        dtRank = dtProj.dim(1)
        kConv = convWeight.dim(2)
        guard dState == 16, kConv == 4, dtRank == 4 else {
            throw EnhancementError.invalidConfiguration("Mamba kernels expect d_state 16, d_conv 4 and dt_rank 4")
        }
    }

    /// Streaming state of the time Mamba: the last K-1 conv inputs and the SSM state.
    struct State {
        var conv: MLXArray  // (K-1, F, Di)
        var ssm: MLXArray  // (F, Di, N)
    }

    /// Causal Mamba along time, continuing from `state`. Returns the gated scan output
    /// `y * silu(z)`, (T, F, Di), before out_proj.
    func time(_ u: MLXArray, state: inout State) -> MLXArray {
        let (T, F) = (u.dim(0), u.dim(1))
        let xz = linear(u.asType(inProj.dtype), inProj)
        let xs = MambaKernels.convSilu(xz, weight: convWeight, bias: convBias, dInner: dInner, cache: state.conv)
        let x = xz[.ellipsis, ..<dInner]
        state.conv = detached(T >= kConv - 1 ? x[(T - kConv + 1)...] : concatenated([state.conv, x])[T...])
        let dbl = matmul(xs, xProj)
        let (y, h) = MambaKernels.scan(xs: xs, xz: xz, dbl: dbl, group: self, frames: T, bins: F, h0: state.ssm)
        state.ssm = h!
        return y
    }

    /// Forward and backward Mamba along frequency within each frame. Returns both gated
    /// scan outputs, (T, F, 2·Di) direction-major per position, before out_proj.
    func freq(_ u: MLXArray) -> MLXArray {
        let (T, F) = (u.dim(0), u.dim(1))
        let xz = linear(u.asType(inProj.dtype), inProj)
        let xs = MambaKernels.convSilu(xz, weight: convWeight, bias: convBias, dInner: dInner, cache: nil)
        let dbl = matmul(xs, xProj)
        return MambaKernels.scan(xs: xs, xz: xz, dbl: dbl, group: self, frames: T, bins: F, h0: nil).y
    }
}

/// Metal kernels for the parts of Mamba between GEMMs. Rows are (t, f) positions in
/// row-major order; `xz` is the in_proj output, (T, F, dirs·2·Di) holding each
/// direction's x then z. With a cache the kernels run along time (the stateful causal
/// Mamba); without, along frequency, where direction 1 runs in reverse.
enum MambaKernels {
    /// silu(depthwise conv of x), output (dirs, rows, Di). Along time, `cache` holds the
    /// K-1 frames before this chunk.
    static func convSilu(_ xz: MLXArray, weight: MLXArray, bias: MLXArray, dInner: Int, cache: MLXArray?)
        -> MLXArray
    {
        let (T, F) = (xz.dim(0), xz.dim(1))
        let dirs = weight.dim(0), rows = T * F
        let params = MLXArray([Int32(dInner), Int32(T), Int32(F), Int32(xz.dim(2))])
        let inputs: [MLXArray] = [xz, weight, bias, params] + (cache.map { [$0] } ?? [])
        return (cache == nil ? convFreq : convTime)(
            inputs, template: [("OutT", xz.dtype)], grid: (dInner, rows, dirs), threadGroup: (min(256, dInner), 1, 1),
            outputShapes: [[dirs, rows, dInner]], outputDTypes: [xz.dtype])[0]
    }

    private static let convBody = """
        uint d = thread_position_in_grid.x, row = thread_position_in_grid.y, dir = thread_position_in_grid.z;
        int Di = params[0], T = params[1], F = params[2], S = params[3];
        int t = int(row) / F, f = int(row) % F;
        auto w = weight + (dir * Di + d) * 4;
        float acc = bias[dir * Di + d];
        for (int k = 0; k < 4; ++k) {
        """

    private static let convFreq = MLXFast.metalKernel(
        name: "reuse_conv_silu_freq",
        inputNames: ["xz", "weight", "bias", "params"], outputNames: ["out"],
        source: convBody + """
                int ff = dir == 0 ? f - 3 + k : f + 3 - k;
                if (ff >= 0 && ff < F) acc += w[k] * xz[(t * F + ff) * S + dir * 2 * Di + d];
            }
            out[(dir * T * F + row) * Di + d] = OutT(acc / (1.0f + metal::exp(-acc)));
            """)

    private static let convTime = MLXFast.metalKernel(
        name: "reuse_conv_silu_time",
        inputNames: ["xz", "weight", "bias", "params", "cache"], outputNames: ["out"],
        source: convBody + """
                int tt = t - 3 + k;
                acc += w[k] * (tt >= 0 ? xz[(tt * F + f) * S + d] : cache[((tt + 3) * F + f) * Di + d]);
            }
            out[row * Di + d] = OutT(acc / (1.0f + metal::exp(-acc)));
            """)

    /// Selective scan with dt_proj, softplus, the D skip and the silu(z) gate fused in.
    /// Output (T, F, dirs·Di) and, with `h0`, the final SSM state.
    ///
    /// One thread per (sequence, channel) with its 16 states in registers: every lane of a
    /// SIMD group advances its own channel, and the B and C rows it reads are the same for
    /// the whole group, so their loads are broadcast.
    static func scan(
        xs: MLXArray, xz: MLXArray, dbl: MLXArray, group g: MambaGroup, frames T: Int, bins F: Int, h0: MLXArray?
    ) -> (y: MLXArray, h: MLXArray?) {
        let Di = g.dInner, dirs = g.directions, rows = T * F
        // Time: one sequence per bin, stepping a frame (F rows). Frequency: one per frame.
        let (sequences, length, strideB, strideL) = h0 != nil ? (F, T, 1, F) : (T, F, F, 1)
        let params = MLXArray([
            Int32(Di), Int32(length), Int32(strideB), Int32(strideL), Int32(rows), Int32(xz.dim(2)),
            Int32(dbl.dim(2)), Int32(dirs),
        ])
        let grid = (sequences * Di, dirs, 1)
        let inputs = [xs, xz, dbl, g.dtProj, g.dtBias, g.A, g.D, params]
        let yShape = [T, F, dirs * Di]
        if let h0 {
            let out = scanState(
                inputs + [h0], template: [("OutT", xs.dtype)], grid: grid, threadGroup: (min(256, Di), 1, 1),
                outputShapes: [yShape, h0.shape], outputDTypes: [xs.dtype, .float32])
            return (out[0], out[1])
        }
        let out = scanPlain(
            inputs, template: [("OutT", xs.dtype)], grid: grid, threadGroup: (min(256, Di), 1, 1),
            outputShapes: [yShape], outputDTypes: [xs.dtype])
        return (out[0], nil)
    }

    private static func scanSource(stateful: Bool) -> String {
        """
            uint gid = thread_position_in_grid.x, dir = thread_position_in_grid.y;
            int Di = params[0], L = params[1], strideB = params[2], strideL = params[3];
            int rows = params[4], S = params[5], W = params[6], dirs = params[7];
            int b = int(gid) / Di, d = int(gid) % Di;
            int pd = int(dir) * Di + d;
            float w0 = dtW[(dir * 4 + 0) * Di + d], w1 = dtW[(dir * 4 + 1) * Di + d];
            float w2 = dtW[(dir * 4 + 2) * Di + d], w3 = dtW[(dir * 4 + 3) * Di + d];
            float bdt = dtB[pd], skip = Dv[pd];
            float a[16], h[16];
            for (int n = 0; n < 16; ++n) {
                a[n] = A[pd * 16 + n];
                h[n] = \(stateful ? "h0[gid * 16 + n]" : "0.0f");
            }
            auto xsd = xs + int(dir) * rows * Di;
            auto dd = dbl + int(dir) * rows * W;
            for (int i = 0; i < L; ++i) {
                int l = dir == 1 ? L - 1 - i : i;
                int row = b * strideB + l * strideL;
                auto r = dd + row * W;
                float4 dtv = float4(r[0], r[1], r[2], r[3]);
                float dt = bdt + w0 * dtv.x + w1 * dtv.y + w2 * dtv.z + w3 * dtv.w;
                dt = dt > 20.0f ? dt : metal::log(1.0f + metal::exp(dt));
                float xv = xsd[row * Di + d];
                float dx = dt * xv;
                float acc = 0.0f;
                for (int q = 0; q < 4; ++q) {
                    auto b4 = r + 4 + 4 * q, c4 = r + 20 + 4 * q;
                    float4 bq = float4(b4[0], b4[1], b4[2], b4[3]);
                    float4 cq = float4(c4[0], c4[1], c4[2], c4[3]);
                    for (int j = 0; j < 4; ++j) {
                        int n = 4 * q + j;
                        h[n] = metal::exp(dt * a[n]) * h[n] + dx * bq[j];
                        acc += h[n] * cq[j];
                    }
                }
                float zv = xz[row * S + int(dir) * 2 * Di + Di + d];
                y[(row * dirs + int(dir)) * Di + d] = OutT((acc + skip * xv) * zv / (1.0f + metal::exp(-zv)));
            }
            \(stateful ? "for (int n = 0; n < 16; ++n) hT[gid * 16 + n] = h[n];" : "")
        """
    }

    private static let scanPlain = MLXFast.metalKernel(
        name: "reuse_scan",
        inputNames: ["xs", "xz", "dbl", "dtW", "dtB", "A", "Dv", "params"], outputNames: ["y"],
        source: scanSource(stateful: false))

    private static let scanState = MLXFast.metalKernel(
        name: "reuse_scan_state",
        inputNames: ["xs", "xz", "dbl", "dtW", "dtB", "A", "Dv", "params", "h0"], outputNames: ["y", "hT"],
        source: scanSource(stateful: true))
}
