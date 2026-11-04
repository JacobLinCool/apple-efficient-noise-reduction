import Accelerate
import Foundation

// CPU building blocks for DeepFilterNet3. Activations are (T, F, C) row-major [Float];
// convolutions are causal in time and keep their past frames in a cache, so a stream
// fed in chunks of any size reproduces the whole-utterance result.

/// C = A·B (+ C when `accumulate`), row-major, with leading dimensions for strided
/// blocks (e.g. one group of a grouped linear layer).
@inline(__always)
func sgemm(
    _ m: Int, _ n: Int, _ k: Int, _ a: UnsafePointer<Float>, lda: Int, _ b: UnsafePointer<Float>, ldb: Int,
    _ c: UnsafeMutablePointer<Float>, ldc: Int, accumulate: Bool = false
) {
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k), 1, a, Int32(lda), b, Int32(ldb),
        accumulate ? 1 : 0, c, Int32(ldc))
}

enum Activation {
    case relu, sigmoid, tanh, none

    func apply(_ x: inout [Float]) {
        switch self {
        case .relu: for i in x.indices where x[i] < 0 { x[i] = 0 }
        case .sigmoid: for i in x.indices { x[i] = 1 / (1 + expf(-x[i])) }
        case .tanh: for i in x.indices { x[i] = tanhf(x[i]) }
        case .none: break
        }
    }
}

/// `Conv2dNormAct` / `ConvTranspose2dNormAct`: a (grouped or depthwise) conv over
/// (time, freq), an optional 1x1 pointwise conv, BatchNorm folded into a per-channel
/// affine map, then the activation.
struct ConvBlock {
    enum Kind { case conv, transposed }

    let kind: Kind
    let groups: Int
    let inChannels: Int
    let outChannels: Int
    let kT: Int
    let kF: Int
    let strideF: Int
    let padF: Int
    /// Per group, (kT·kF·Cin/g, Cout/g) for the im2col GEMM; depthwise kernels keep (C, kT·kF).
    let weights: [[Float]]
    let pointwise: [Float]?  // (C, C) as x·W
    let scale: [Float]
    let shift: [Float]
    let activation: Activation

    var span: Int { kT - 1 }
    var depthwise: Bool { groups == inChannels && groups == outChannels }

    init(
        _ w: SafeTensors, conv: String, pointwise pw: String? = nil, norm: String, kind: Kind = .conv,
        groups: Int, strideF: Int = 1, padF: Int, activation: Activation
    ) throws {
        let c = try w(conv)
        let (d0, d1, kT, kF) = (c.shape[0], c.shape[1], c.shape[2], c.shape[3])
        self.kind = kind
        self.groups = groups
        // conv weight (O, I/g, kT, kF); transposed conv weight (I, O/g, kT, kF).
        (inChannels, outChannels) = kind == .conv ? (d1 * groups, d0) : (d0, d1 * groups)
        self.kT = kT
        self.kF = kF
        self.strideF = strideF
        self.padF = padF
        let cinG = inChannels / groups, coutG = outChannels / groups
        if groups == inChannels && groups == outChannels {
            weights = [c.values]  // (C, kT·kF)
        } else {
            precondition(kind == .conv, "only depthwise transposed convs")
            weights = (0 ..< groups).map { g in
                var m = [Float](repeating: 0, count: kT * kF * cinG * coutG)
                for o in 0 ..< coutG {
                    for ci in 0 ..< cinG {
                        for dt in 0 ..< kT {
                            for kf in 0 ..< kF {
                                let src = (((g * coutG + o) * cinG + ci) * kT + dt) * kF + kf
                                m[((dt * kF + kf) * cinG + ci) * coutG + o] = c.values[src]
                            }
                        }
                    }
                }
                return m
            }
        }
        if let pw {
            let p = try w(pw)  // (O, I, 1, 1)
            let n = p.shape[0]
            var t = [Float](repeating: 0, count: n * n)
            for o in 0 ..< n { for i in 0 ..< n { t[i * n + o] = p.values[o * n + i] } }
            pointwise = t
        } else {
            pointwise = nil
        }
        let gamma = try w("\(norm).weight").values, beta = try w("\(norm).bias").values
        let mean = try w("\(norm).running_mean").values, variance = try w("\(norm).running_var").values
        scale = zip(gamma, variance).map { $0 / sqrtf($1 + 1e-5) }
        shift = (0 ..< gamma.count).map { beta[$0] - mean[$0] * gamma[$0] / sqrtf(variance[$0] + 1e-5) }
        self.activation = activation
    }

    func outputBins(_ fIn: Int) -> Int {
        kind == .conv ? (fIn + 2 * padF - kF) / strideF + 1 : fIn * strideF
    }

    /// x: (T, fIn, Cin) → (T, fOut, Cout). `cache` holds the `span` frames before x.
    func callAsFunction(_ x: [Float], frames T: Int, bins fIn: Int, cache: inout [Float]) -> [Float] {
        let cin = inChannels, fOut = outputBins(fIn)
        var input = x
        if span > 0 {
            input = cache + x
            cache = Array(input.suffix(span * fIn * cin))
        }
        var y = [Float](repeating: 0, count: T * fOut * outChannels)
        if depthwise {
            depthwiseConv(input, T: T, fIn: fIn, fOut: fOut, into: &y)
        } else {
            groupedConv(input, T: T, fIn: fIn, fOut: fOut, into: &y)
        }
        if let pointwise {
            var z = [Float](repeating: 0, count: y.count)
            sgemm(T * fOut, outChannels, outChannels, y, lda: outChannels, pointwise, ldb: outChannels, &z, ldc: outChannels)
            y = z
        }
        let c = outChannels
        y.withUnsafeMutableBufferPointer { p in
            for r in 0 ..< T * fOut {
                for o in 0 ..< c { p[r * c + o] = p[r * c + o] * scale[o] + shift[o] }
            }
        }
        activation.apply(&y)
        return y
    }

    /// One kernel per channel; for transposed convs output bin f·s - pad + k gathers input f.
    private func depthwiseConv(_ x: [Float], T: Int, fIn: Int, fOut: Int, into y: inout [Float]) {
        let c = inChannels, w = weights[0], k = kT * kF
        x.withUnsafeBufferPointer { xp in
            y.withUnsafeMutableBufferPointer { yp in
                for t in 0 ..< T {
                    for dt in 0 ..< kT {
                        for kf in 0 ..< kF {
                            for fo in 0 ..< fOut {
                                let fi: Int
                                if kind == .conv {
                                    fi = fo * strideF - padF + kf
                                } else {
                                    // o = i·s + kf - pad  ⇒  i = (o + pad - kf) / s
                                    let num = fo + padF - kf
                                    guard num >= 0, num % strideF == 0 else { continue }
                                    fi = num / strideF
                                }
                                guard fi >= 0, fi < fIn else { continue }
                                let src = xp.baseAddress! + ((t + dt) * fIn + fi) * c
                                let dst = yp.baseAddress! + (t * fOut + fo) * c
                                for ch in 0 ..< c { dst[ch] += src[ch] * w[ch * k + dt * kF + kf] }
                            }
                        }
                    }
                }
            }
        }
    }

    private func groupedConv(_ x: [Float], T: Int, fIn: Int, fOut: Int, into y: inout [Float]) {
        let cin = inChannels, cinG = cin / groups, coutG = outChannels / groups
        let kdim = kT * kF * cinG
        var cols = [Float](repeating: 0, count: T * fOut * kdim)
        for g in 0 ..< groups {
            for i in cols.indices { cols[i] = 0 }
            for t in 0 ..< T {
                for fo in 0 ..< fOut {
                    let row = (t * fOut + fo) * kdim
                    for dt in 0 ..< kT {
                        for kf in 0 ..< kF {
                            let fi = fo * strideF - padF + kf
                            guard fi >= 0, fi < fIn else { continue }
                            let src = ((t + dt) * fIn + fi) * cin + g * cinG
                            let dst = row + (dt * kF + kf) * cinG
                            for ci in 0 ..< cinG { cols[dst + ci] = x[src + ci] }
                        }
                    }
                }
            }
            y.withUnsafeMutableBufferPointer { yp in
                sgemm(
                    T * fOut, coutG, kdim, cols, lda: kdim, weights[g], ldb: coutG, yp.baseAddress! + g * coutG,
                    ldc: outChannels)
            }
        }
    }
}

/// `GroupedLinearEinsum`: the input is split into `groups` contiguous slices, each with
/// its own (in/g, out/g) matrix.
struct GroupedLinear {
    let groups: Int
    let inSize: Int
    let outSize: Int
    let weight: [Float]  // (G, in/g, out/g)
    let activation: Activation

    init(_ w: SafeTensors, _ name: String, activation: Activation) throws {
        let t = try w(name)
        groups = t.shape[0]
        inSize = t.shape[0] * t.shape[1]
        outSize = t.shape[0] * t.shape[2]
        weight = t.values
        self.activation = activation
    }

    func callAsFunction(_ x: [Float], rows: Int) -> [Float] {
        var y = [Float](repeating: 0, count: rows * outSize)
        let ws = inSize / groups, hs = outSize / groups
        x.withUnsafeBufferPointer { xp in
            weight.withUnsafeBufferPointer { wp in
                y.withUnsafeMutableBufferPointer { yp in
                    for g in 0 ..< groups {
                        sgemm(
                            rows, hs, ws, xp.baseAddress! + g * ws, lda: inSize, wp.baseAddress! + g * ws * hs, ldb: hs,
                            yp.baseAddress! + g * hs, ldc: outSize)
                    }
                }
            }
        }
        activation.apply(&y)
        return y
    }
}

/// Multi-layer `nn.GRU` (gate order r, z, n), stepping one frame at a time; the input
/// projections of a whole chunk run as one GEMM per layer.
struct GRU {
    struct Layer {
        let wih: [Float]  // (in, 3H) as x·W
        let whh: [Float]  // (3H, H) row-major for h·Wᵀ via sgemv
        let bih: [Float]
        let bhh: [Float]
    }

    let layers: [Layer]
    let hidden: Int

    init(_ w: SafeTensors, prefix p: String, layers count: Int) throws {
        layers = try (0 ..< count).map { l in
            let wih = try w("\(p).weight_ih_l\(l)")  // (3H, in)
            let (rows, cols) = (wih.shape[0], wih.shape[1])
            var t = [Float](repeating: 0, count: rows * cols)
            for r in 0 ..< rows { for c in 0 ..< cols { t[c * rows + r] = wih.values[r * cols + c] } }
            return Layer(
                wih: t, whh: try w("\(p).weight_hh_l\(l)").values, bih: try w("\(p).bias_ih_l\(l)").values,
                bhh: try w("\(p).bias_hh_l\(l)").values)
        }
        hidden = layers[0].bhh.count / 3
    }

    /// x: (T, in) → (T, H); `state` holds one hidden vector per layer.
    func callAsFunction(_ x: [Float], frames T: Int, state: inout [[Float]]) -> [Float] {
        let H = hidden
        var input = x
        for (l, layer) in layers.enumerated() {
            let inSize = input.count / T
            var gi = [Float](repeating: 0, count: T * 3 * H)
            sgemm(T, 3 * H, inSize, input, lda: inSize, layer.wih, ldb: 3 * H, &gi, ldc: 3 * H)
            var h = state[l]
            var out = [Float](repeating: 0, count: T * H)
            var gh = [Float](repeating: 0, count: 3 * H)
            for t in 0 ..< T {
                gh = layer.bhh
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(3 * H), Int32(H), 1, layer.whh, Int32(H), h, 1, 1, &gh, 1)
                let base = t * 3 * H
                for j in 0 ..< H {
                    let r = 1 / (1 + expf(-(gi[base + j] + layer.bih[j] + gh[j])))
                    let z = 1 / (1 + expf(-(gi[base + H + j] + layer.bih[H + j] + gh[H + j])))
                    let n = tanhf(gi[base + 2 * H + j] + layer.bih[2 * H + j] + r * gh[2 * H + j])
                    h[j] = (1 - z) * n + z * h[j]
                }
                out.replaceSubrange(t * H ..< (t + 1) * H, with: h)
            }
            state[l] = h
            input = out
        }
        return input
    }
}

/// `SqueezedGRU_S`: grouped linear in (ReLU), GRU, optional grouped linear out (ReLU).
struct SqueezedGRU {
    let linearIn: GroupedLinear
    let gru: GRU
    let linearOut: GroupedLinear?

    init(_ w: SafeTensors, prefix p: String, layers: Int, hasOutput: Bool) throws {
        linearIn = try GroupedLinear(w, "\(p).linear_in.0.weight", activation: .relu)
        gru = try GRU(w, prefix: "\(p).gru", layers: layers)
        linearOut = hasOutput ? try GroupedLinear(w, "\(p).linear_out.0.weight", activation: .relu) : nil
    }

    func callAsFunction(_ x: [Float], frames T: Int, state: inout [[Float]]) -> [Float] {
        let h = gru(linearIn(x, rows: T), frames: T, state: &state)
        return linearOut.map { $0(h, rows: T) } ?? h
    }

    func zeroState() -> [[Float]] { gru.layers.map { _ in [Float](repeating: 0, count: gru.hidden) } }
}
