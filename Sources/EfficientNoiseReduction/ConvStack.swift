import Foundation
import MLX
import Metal
import MetalPerformanceShadersGraph

/// The convolutional encoder and both decoders of Real-time RE-USE as MPSGraph
/// executables: dense blocks of 3x3 convs with time dilation, sub-pixel up-sampling,
/// channel LayerNorm and PReLU. MPSGraph's convolutions (the GPU backend of Core ML)
/// run these shapes several times faster than GEMM-based convs in MLX.
///
/// Each conv keeps the frames it still needs in a cache that stays on the GPU between
/// calls, so chunked streaming reproduces the whole-utterance result. Executables are
/// compiled per (bins, frames) shape and cached; a stream sees at most three shapes
/// (its first chunk, the steady chunk and the final one).
final class ConvStack: @unchecked Sendable {
    struct Conv {
        let weight: [Float]  // HWIO: (kT, kF, I, O)
        let bias: [Float]
        let kT: Int, kF: Int, cin: Int, cout: Int
        var dilation = 1, stride = 1, pad = 0

        init(_ w: Weights, weight: MLXArray, bias: String, dilation: Int = 1, stride: Int = 1, pad: Int = 0) throws {
            // MLX layout (O, kT, kF, I) → (kT, kF, I, O)
            (cout, kT, kF, cin) = (weight.dim(0), weight.dim(1), weight.dim(2), weight.dim(3))
            self.weight = weight.transposed(1, 2, 3, 0).asArray(Float.self)
            self.bias = try w.raw(bias).asArray(Float.self)
            (self.dilation, self.stride, self.pad) = (dilation, stride, pad)
        }

        var span: Int { (kT - 1) * dilation }
    }

    struct NormAct {
        let gamma: [Float], beta: [Float], slope: [Float]

        init(_ w: Weights, norm: String, prelu: String) throws {
            gamma = try w.raw("\(norm).weight").asArray(Float.self)
            beta = try w.raw("\(norm).bias").asArray(Float.self)
            slope = try w.raw("\(prelu).weight").asArray(Float.self)
        }
    }

    struct Layer {
        let conv: Conv
        let act: NormAct
    }

    struct Decoder {
        let dense: [Layer]
        let upFreq: Layer  // 64 → 128 channels, shuffled to 2x bins
        let upTime: Layer  // causal kernel-3 conv over time
    }

    let lookAhead: Int
    let dtype: MPSDataType
    private let inputConv: Layer
    private let encoderDense: [Layer]
    private let encoderOut: Layer
    private let mag: Decoder
    private let phase: Decoder
    private let magOut: (w: [Float], b: [Float])  // 1x1, 64 → 1
    private let phaseOut: (w: [Float], b: [Float])  // 1x1, 64 → (real, imaginary)

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let graphDevice: MPSGraphDevice
    private var executables: [Key: Compiled] = [:]
    private let lock = NSLock()

    init(weights w: Weights, exitLayer: Int, lookAhead: Int, precision: ReUseModel.Precision) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EnhancementError.invalidConfiguration("no Metal device")
        }
        (self.device, self.queue, graphDevice) = (device, queue, MPSGraphDevice(mtlDevice: device))
        self.lookAhead = lookAhead
        dtype = precision == .fast ? .float16 : .float32

        func layer(conv: String, bias: String? = nil, norm: String, prelu: String, weight: MLXArray? = nil,
                   dilation: Int = 1, stride: Int = 1, pad: Int = 0) throws -> Layer {
            Layer(
                conv: try Conv(
                    w, weight: try weight ?? w.conv("\(conv).weight"), bias: bias ?? "\(conv).bias", dilation: dilation,
                    stride: stride, pad: pad),
                act: try NormAct(w, norm: norm, prelu: prelu))
        }
        func dense(_ p: String) throws -> [Layer] {
            try (0 ..< 4).map { i in
                try layer(conv: "\(p).dense_block.\(i).1", norm: "\(p).dense_block.\(i).2.norm", prelu: "\(p).dense_block.\(i).3",
                          dilation: 1 << i, pad: 1)
            }
        }
        let e = "dense_encoder", first = "\(e).dense_conv_1_\(lookAhead + 1)"
        inputConv = try layer(conv: "\(first).0", norm: "\(first).1.norm", prelu: "\(first).2", pad: 1)
        encoderDense = try dense("\(e).dense_block")
        encoderOut = try layer(conv: "\(e).dense_conv_2.0", norm: "\(e).dense_conv_2.1.norm", prelu: "\(e).dense_conv_2.2", stride: 2)
        func decoder(_ p: String) throws -> Decoder {
            Decoder(
                dense: try dense("\(p).dense_block"),
                upFreq: try layer(conv: "\(p).up_conv1.0.conv", norm: "\(p).up_conv1.1.norm", prelu: "\(p).up_conv1.2", pad: 1),
                upTime: try layer(
                    conv: "\(p).up_conv2.0.conv", norm: "\(p).up_conv2.1.norm", prelu: "\(p).up_conv2.2",
                    weight: try w.convFT("\(p).up_conv2.0.conv.weight")))
        }
        let i = exitLayer - 1
        mag = try decoder("mask_decoder_list.\(i)")
        phase = try decoder("phase_decoder_list.\(i)")
        magOut = (
            try w.raw("mask_decoder_list.\(i).final_conv.weight").asArray(Float.self),
            try w.raw("mask_decoder_list.\(i).final_conv.bias").asArray(Float.self)
        )
        let p = "phase_decoder_list.\(i)"
        phaseOut = (
            try w.raw("\(p).phase_conv_r.weight").asArray(Float.self) + w.raw("\(p).phase_conv_i.weight").asArray(Float.self),
            try w.raw("\(p).phase_conv_r.bias").asArray(Float.self) + w.raw("\(p).phase_conv_i.bias").asArray(Float.self)
        )
    }

    // MARK: State

    /// The conv caches of one stream, on the GPU; zeros at the start of an utterance.
    struct State {
        var input: [Float]  // input-conv context on the host: (2 - lookAhead) zero frames at first, then 2
        var inputFrames: Int
        var encoder: [MPSGraphTensorData]
        var mag: [MPSGraphTensorData]  // 4 dense + up-time
        var phase: [MPSGraphTensorData]
    }

    func makeState(bins F: Int) -> State {
        let width = F + 2, inner = (width - 3) / 2 + 1
        func zeros(_ shape: [Int]) -> MPSGraphTensorData {
            let bytes = shape.reduce(1, *) * (dtype == .float16 ? 2 : 4)
            return MPSGraphTensorData(
                device: graphDevice, data: Data(count: bytes), shape: shape.map { NSNumber(value: $0) }, dataType: dtype)
        }
        func caches(_ layers: [Layer], _ f: Int) -> [MPSGraphTensorData] {
            layers.map { zeros([1, $0.conv.span, f, $0.conv.cin]) }
        }
        let decoder = { (d: Decoder) in caches(d.dense, inner) + [zeros([1, d.upTime.conv.span, 2 * inner, 64])] }
        return State(
            input: [Float](repeating: 0, count: (2 - lookAhead) * width * 2), inputFrames: 2 - lookAhead,
            encoder: caches(encoderDense, width), mag: decoder(mag), phase: decoder(phase))
    }

    // MARK: Calls

    /// Features (T, F + 2, 2) (magnitude and phase with NVIDIA's two zero bins) → encoder
    /// output (T', (F + 1) / 2, 64) as float32, T' = T minus the look-ahead on the first
    /// call. Nil while the input conv still waits for its first frames.
    func encode(_ features: [Float], frames T: Int, bins F: Int, state s: inout State) -> MLXArray? {
        let width = F + 2, perFrame = width * 2, span = inputConv.conv.span
        let all = s.input + features
        let frames = s.inputFrames + T
        s.input = Array(all.suffix(min(frames, span) * perFrame))
        s.inputFrames = min(frames, span)
        // Tiny first chunks with look-ahead: wait until the input conv has its frames.
        guard frames > span else { return nil }
        let exe = compiled(Key(kind: .encoder, bins: F, frames: frames))
        let out = exe.run(queue: queue, inputs: [floats(all, [1, frames, width, 2])] + s.encoder)
        s.encoder = Array(out[1...])
        return MLXArray(read(out[0]), [frames - span, (width - 3) / 2 + 1, 64])
    }

    /// Mamba output (T, F', 64) → compressed magnitude and phase (T, F).
    func decode(_ x: MLXArray, bins F: Int, state s: inout State) -> (mag: MLXArray, phase: MLXArray) {
        let (T, inner) = (x.dim(0), x.dim(1))
        let exe = compiled(Key(kind: .decoder, bins: F, frames: T))
        let out = exe.run(queue: queue, inputs: [floats(x.asArray(Float.self), [1, T, inner, 64])] + s.mag + s.phase)
        s.mag = Array(out[2 ..< 7])
        s.phase = Array(out[7 ..< 12])
        return (MLXArray(read(out[0]), [T, F]), MLXArray(read(out[1]), [T, F]))
    }

    /// Graph inputs and outputs are float32; the graphs cast to and from `dtype` on the GPU.
    private func floats(_ values: [Float], _ shape: [Int]) -> MPSGraphTensorData {
        MPSGraphTensorData(
            device: graphDevice, data: values.withUnsafeBytes { Data($0) }, shape: shape.map { NSNumber(value: $0) },
            dataType: .float32)
    }

    private func read(_ t: MPSGraphTensorData) -> [Float] {
        let n = t.shape.reduce(1) { $0 * $1.intValue }
        var f = [Float](repeating: 0, count: n)
        f.withUnsafeMutableBytes { t.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        return f
    }

    // MARK: Graphs

    struct Key: Hashable {
        enum Kind { case encoder, decoder }
        let kind: Kind
        let bins: Int
        let frames: Int  // encoder: input frames including the input-conv context
    }

    struct Compiled {
        let executable: MPSGraphExecutable
        let feeds: [MPSGraphTensor]  // in the order callers pass inputs

        func run(queue: MTLCommandQueue, inputs: [MPSGraphTensorData]) -> [MPSGraphTensorData] {
            let byTensor = Dictionary(uniqueKeysWithValues: zip(feeds, inputs))
            let ordered = executable.feedTensors!.map { byTensor[$0]! }
            return executable.run(with: queue, inputs: ordered, results: nil, executionDescriptor: nil)
        }
    }

    private func compiled(_ key: Key) -> Compiled {
        lock.lock()
        defer { lock.unlock() }
        if let c = executables[key] { return c }
        let c = key.kind == .encoder ? buildEncoder(key) : buildDecoder(key)
        executables[key] = c
        return c
    }

    private func build(_ g: MPSGraph, feeds: [MPSGraphTensor], targets: [MPSGraphTensor]) -> Compiled {
        let types = Dictionary(uniqueKeysWithValues: feeds.map { ($0, MPSGraphShapedType(shape: $0.shape, dataType: $0.dataType)) })
        let exe = g.compile(with: graphDevice, feeds: types, targetTensors: targets, targetOperations: nil, compilationDescriptor: nil)
        return Compiled(executable: exe, feeds: feeds)
    }

    private func buildEncoder(_ key: Key) -> Compiled {
        let g = MPSGraph(), b = Builder(graph: g, dtype: dtype)
        let width = key.bins + 2
        let input = g.placeholder(shape: shape([1, key.frames, width, 2]), dataType: .float32, name: nil)
        let x = b.storage(input)
        let caches = encoderDense.map { g.placeholder(shape: shape([1, $0.conv.span, width, $0.conv.cin]), dataType: dtype, name: nil) }
        let (h, newCaches) = b.dense(encoderDense, b.layer(inputConv, x), caches: caches)
        let y = g.cast(b.layer(encoderOut, h), to: .float32, name: nil)
        return build(g, feeds: [input] + caches, targets: [y] + newCaches)
    }

    private func buildDecoder(_ key: Key) -> Compiled {
        let g = MPSGraph(), b = Builder(graph: g, dtype: dtype)
        let inner = (key.bins + 2 - 3) / 2 + 1
        let input = g.placeholder(shape: shape([1, key.frames, inner, 64]), dataType: .float32, name: nil)
        let x = b.storage(input)
        var feeds = [input], outputs: [MPSGraphTensor] = [], caches: [MPSGraphTensor] = []
        for d in [mag, phase] {
            let dense = d.dense.map { g.placeholder(shape: shape([1, $0.conv.span, inner, $0.conv.cin]), dataType: dtype, name: nil) }
            let up = g.placeholder(shape: shape([1, d.upTime.conv.span, 2 * inner, 64]), dataType: dtype, name: nil)
            feeds += dense + [up]
            var (h, newDense) = b.dense(d.dense, x, caches: dense)
            // Sub-pixel up-sampling: channel r·64 + c of bin f becomes channel c of bin 2f + r.
            h = b.conv(d.upFreq.conv, h)
            h = g.reshape(h, shape: shape([1, key.frames, 2 * inner, 64]), name: nil)
            h = b.normAct(d.upFreq.act, h)
            let (u, newUp) = b.causal(d.upTime, h, cache: up)
            outputs.append(u)
            caches += newDense + [newUp]
        }
        // Output heads: magnitude 1x1, phase atan2 of two 1x1 convs, first F bins, float32 out.
        let m = b.pointwise(outputs[0], w: magOut.w, b: magOut.b, cout: 1)
        let ri = b.pointwise(outputs[1], w: phaseOut.w, b: phaseOut.b, cout: 2)
        let re = g.sliceTensor(ri, dimension: 3, start: 0, length: 1, name: nil)
        let im = g.sliceTensor(ri, dimension: 3, start: 1, length: 1, name: nil)
        let crop = { (t: MPSGraphTensor) in
            g.cast(g.sliceTensor(t, dimension: 2, start: 0, length: key.bins, name: nil), to: .float32, name: nil)
        }
        let pha = g.atan2(withPrimaryTensor: crop(im), secondaryTensor: crop(re), name: nil)
        return build(g, feeds: feeds, targets: [crop(m), pha] + caches)
    }

    private func shape(_ s: [Int]) -> [NSNumber] { s.map { NSNumber(value: $0) } }

    /// Graph-building helpers over NHWC (1, time, freq, channels) tensors.
    private struct Builder {
        let graph: MPSGraph
        let dtype: MPSDataType

        func storage(_ x: MPSGraphTensor) -> MPSGraphTensor {
            dtype == .float32 ? x : graph.cast(x, to: dtype, name: nil)
        }

        func constant(_ v: [Float], _ shape: [Int]) -> MPSGraphTensor {
            let t = graph.constant(v.withUnsafeBytes { Data($0) }, shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
            return dtype == .float32 ? t : graph.cast(t, to: dtype, name: nil)
        }

        /// The conv alone, over an input that already holds its `span` past frames.
        func conv(_ c: Conv, _ x: MPSGraphTensor) -> MPSGraphTensor {
            let d = MPSGraphConvolution2DOpDescriptor(
                strideInX: c.stride, strideInY: 1, dilationRateInX: 1, dilationRateInY: c.dilation, groups: 1,
                paddingLeft: c.pad, paddingRight: c.pad, paddingTop: 0, paddingBottom: 0, paddingStyle: .explicit,
                dataLayout: .NHWC, weightsLayout: .HWIO)!
            let y = graph.convolution2D(x, weights: constant(c.weight, [c.kT, c.kF, c.cin, c.cout]), descriptor: d, name: nil)
            return graph.addition(y, constant(c.bias, [c.cout]), name: nil)
        }

        /// Channel LayerNorm and PReLU.
        func normAct(_ a: NormAct, _ x: MPSGraphTensor) -> MPSGraphTensor {
            let mean = graph.mean(of: x, axes: [3], name: nil)
            let variance = graph.variance(of: x, mean: mean, axes: [3], name: nil)
            let n = graph.normalize(
                x, mean: mean, variance: variance, gamma: constant(a.gamma, [a.gamma.count]),
                beta: constant(a.beta, [a.beta.count]), epsilon: 1e-5, name: nil)
            let zero = graph.constant(0, dataType: dtype)
            let negative = graph.multiplication(graph.minimum(n, zero, name: nil), constant(a.slope, [a.slope.count]), name: nil)
            return graph.addition(graph.maximum(n, zero, name: nil), negative, name: nil)
        }

        func layer(_ l: Layer, _ x: MPSGraphTensor) -> MPSGraphTensor { normAct(l.act, conv(l.conv, x)) }

        /// A layer causal in time: prepend the cache, return the output and the new cache.
        func causal(_ l: Layer, _ x: MPSGraphTensor, cache: MPSGraphTensor) -> (MPSGraphTensor, MPSGraphTensor) {
            let all = graph.concatTensors([cache, x], dimension: 1, name: nil)
            let frames = all.shape![1].intValue
            let span = l.conv.span
            let newCache = graph.sliceTensor(all, dimension: 1, start: frames - span, length: span, name: nil)
            return (layer(l, all), newCache)
        }

        /// Four causal layers, each fed the concatenation of every earlier output.
        func dense(_ layers: [Layer], _ x: MPSGraphTensor, caches: [MPSGraphTensor]) -> (MPSGraphTensor, [MPSGraphTensor]) {
            var skip = x, out = x, newCaches: [MPSGraphTensor] = []
            for (l, cache) in zip(layers, caches) {
                let (y, c) = causal(l, skip, cache: cache)
                newCaches.append(c)
                out = y
                skip = graph.concatTensors([y, skip], dimension: 3, name: nil)
            }
            return (out, newCaches)
        }

        func pointwise(_ x: MPSGraphTensor, w: [Float], b: [Float], cout: Int) -> MPSGraphTensor {
            // w: (cout, 64) row-major → HWIO (1, 1, 64, cout)
            let cin = w.count / cout
            var hwio = [Float](repeating: 0, count: w.count)
            for o in 0 ..< cout { for i in 0 ..< cin { hwio[i * cout + o] = w[o * cin + i] } }
            let d = MPSGraphConvolution2DOpDescriptor(
                strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingLeft: 0,
                paddingRight: 0, paddingTop: 0, paddingBottom: 0, paddingStyle: .explicit, dataLayout: .NHWC,
                weightsLayout: .HWIO)!
            let y = graph.convolution2D(x, weights: constant(hwio, [1, 1, cin, cout]), descriptor: d, name: nil)
            return graph.addition(y, constant(b, [cout]), name: nil)
        }
    }
}
