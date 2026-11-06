import Foundation
import MLX
import Metal
import MetalPerformanceShadersGraph

/// The convolutional encoder and both decoders of NemotronLabs RE-USE as MPSGraph
/// executables. Unlike Real-time RE-USE, every conv is non-causal and every norm is an
/// InstanceNorm over the whole chunk, so a chunk is always processed in one piece:
///
/// - encoder: 1x1 conv, a dense block of 3x3 convs with time dilation 1, 2, 4, 8, then a
///   (1, 3) conv with stride (4, 2) that keeps every fourth frame and halves the bins;
/// - decoders: a dense block, sub-pixel up-sampling ×2 in frequency and ×4 in time, then
///   a 1x1 head (magnitude) or two 1x1 heads and atan2 (phase).
///
/// Each conv is followed by InstanceNorm and PReLU. The norm statistics are computed in
/// float32 even when activations are float16: they are averages over up to a million
/// positions. Executables are compiled per (frames, bins) shape and cached.
final class OfflineConvStack: @unchecked Sendable {
    typealias Conv = ConvStack.Conv
    typealias NormAct = ConvStack.NormAct

    struct Layer {
        let conv: Conv
        let act: NormAct
    }

    struct Decoder {
        let dense: [Layer]
        let upFreq: Layer  // (1, 3) over frequency, 64 → 128 channels, shuffled to 2x bins
        let upTime: Layer  // (3, 1) over time, 64 → 256 channels, shuffled to 4x frames
    }

    let dtype: MPSDataType
    private let inputConv: Layer
    private let encoderDense: [Layer]
    private let encoderOut: Layer
    private let mag: Decoder
    private let phase: Decoder
    private let magOut: (w: [Float], b: [Float])  // 1x1, 64 → 1
    private let phaseOut: (w: [Float], b: [Float])  // 1x1, 64 → (real, imaginary)

    private let queue: MTLCommandQueue
    private let graphDevice: MPSGraphDevice
    /// The most recent shapes' executables (see `OfflineNetwork`'s compiled graphs).
    private let executables = RecentCache<Key, Compiled>(capacity: 6)

    init(weights w: Weights, precision: ReUseModel.Precision) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EnhancementError.invalidConfiguration("no Metal device")
        }
        (self.queue, graphDevice) = (queue, MPSGraphDevice(mtlDevice: device))
        dtype = precision == .fast ? .float16 : .float32

        func layer(_ p: String, conv: String, weight: MLXArray? = nil, dilation: Int = 1) throws -> Layer {
            Layer(
                conv: try Conv(w, weight: try weight ?? w.conv("\(conv).weight"), bias: "\(conv).bias", dilation: dilation),
                act: try NormAct(w, norm: "\(p).1", prelu: "\(p).2"))
        }
        func dense(_ p: String) throws -> [Layer] {
            try (0 ..< 4).map { i in
                try layer("\(p).dense_block.dense_block.\(i)", conv: "\(p).dense_block.dense_block.\(i).0", dilation: 1 << i)
            }
        }
        inputConv = try layer("dense_encoder.dense_conv_1", conv: "dense_encoder.dense_conv_1.0")
        encoderDense = try dense("dense_encoder")
        encoderOut = try layer("dense_encoder.dense_conv_2", conv: "dense_encoder.dense_conv_2.0")
        func decoder(_ p: String) throws -> Decoder {
            Decoder(
                dense: try dense(p),
                upFreq: try layer("\(p).up_conv1", conv: "\(p).up_conv1.0.conv"),
                upTime: try layer("\(p).up_conv2", conv: "\(p).up_conv2.0.conv", weight: try w.convFT("\(p).up_conv2.0.conv.weight")))
        }
        mag = try decoder("mask_decoder")
        phase = try decoder("phase_decoder")
        magOut = (try w.raw("mask_decoder.final_conv.weight").asArray(Float.self), try w.raw("mask_decoder.final_conv.bias").asArray(Float.self))
        phaseOut = (
            try w.raw("phase_decoder.phase_conv_r.weight").asArray(Float.self) + w.raw("phase_decoder.phase_conv_i.weight").asArray(Float.self),
            try w.raw("phase_decoder.phase_conv_r.bias").asArray(Float.self) + w.raw("phase_decoder.phase_conv_i.bias").asArray(Float.self)
        )
    }

    /// Mamba resolution for T frames and F bins: NVIDIA pads two zero frames and bins,
    /// then the encoder keeps every fourth frame and strides bins by two.
    static func innerShape(frames T: Int, bins F: Int) -> (frames: Int, bins: Int) {
        ((T + 2 - 1) / 4 + 1, (F + 2 - 3) / 2 + 1)
    }

    // MARK: Calls

    /// Features (T + 2, F + 2, 2): compressed magnitude and phase with NVIDIA's two zero
    /// frames and bins. Returns the encoder output (T', F', 64) as float32.
    func encode(_ features: [Float], frames T: Int, bins F: Int) -> MLXArray {
        let exe = compiled(Key(kind: .encoder, frames: T, bins: F))
        let out = exe.run(queue: queue, inputs: [floats(features, [1, T + 2, F + 2, 2])])
        let inner = Self.innerShape(frames: T, bins: F)
        return MLXArray(read(out[0]), [inner.frames, inner.bins, 64])
    }

    /// Mamba output (T', F', 64) → compressed magnitude and phase (T, F).
    func decode(_ x: MLXArray, frames T: Int, bins F: Int) -> (mag: [Float], phase: [Float]) {
        let exe = compiled(Key(kind: .decoder, frames: T, bins: F))
        let out = exe.run(queue: queue, inputs: [floats(x.asArray(Float.self), [1, x.dim(0), x.dim(1), 64])])
        return (read(out[0]), read(out[1]))
    }

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
        let frames: Int  // STFT frames of the chunk
        let bins: Int
    }

    private func compiled(_ key: Key) -> Compiled {
        executables.value(for: key) { key.kind == .encoder ? buildEncoder(key) : buildDecoder(key) }
    }

    private func build(_ g: MPSGraph, feeds: [MPSGraphTensor], targets: [MPSGraphTensor]) -> Compiled {
        let types = Dictionary(uniqueKeysWithValues: feeds.map { ($0, MPSGraphShapedType(shape: $0.shape, dataType: $0.dataType)) })
        let exe = g.compile(with: graphDevice, feeds: types, targetTensors: targets, targetOperations: nil, compilationDescriptor: nil)
        return Compiled(executable: exe, feeds: feeds)
    }

    private func buildEncoder(_ key: Key) -> Compiled {
        let g = MPSGraph(), b = Builder(graph: g, dtype: dtype)
        let input = g.placeholder(shape: shape([1, key.frames + 2, key.bins + 2, 2]), dataType: .float32, name: nil)
        var x = b.layer(inputConv, b.storage(input))
        x = b.dense(encoderDense, x)
        x = b.layer(encoderOut, x, strideT: 4, strideF: 2)
        return build(g, feeds: [input], targets: [g.cast(x, to: .float32, name: nil)])
    }

    private func buildDecoder(_ key: Key) -> Compiled {
        let g = MPSGraph(), b = Builder(graph: g, dtype: dtype)
        let inner = Self.innerShape(frames: key.frames, bins: key.bins)
        let input = g.placeholder(shape: shape([1, inner.frames, inner.bins, 64]), dataType: .float32, name: nil)
        let x = b.storage(input)
        var outputs: [MPSGraphTensor] = []
        for d in [mag, phase] {
            var h = b.dense(d.dense, x)
            // Sub-pixel up-sampling in frequency: channel r·64 + c of bin f becomes channel c
            // of bin 2f + r (SPConvTranspose2d, r = 2).
            h = b.conv(d.upFreq.conv, h, padF: 1)
            h = g.reshape(h, shape: shape([1, inner.frames, 2 * inner.bins, 64]), name: nil)
            h = b.normAct(d.upFreq.act, h)
            // And in time: channel r·64 + c of frame t becomes frame 4t + r (r = 4).
            h = b.conv(d.upTime.conv, h, padT: 1)
            h = g.reshape(h, shape: shape([inner.frames, 2 * inner.bins, 4, 64]), name: nil)
            h = g.transposeTensor(h, dimension: 1, withDimension: 2, name: nil)
            h = g.reshape(h, shape: shape([1, 4 * inner.frames, 2 * inner.bins, 64]), name: nil)
            outputs.append(b.normAct(d.upTime.act, h))
        }
        // Heads, cropped to the chunk's T frames and F bins, float32 out.
        let crop = { (t: MPSGraphTensor) -> MPSGraphTensor in
            let c = g.sliceTensor(g.sliceTensor(t, dimension: 1, start: 0, length: key.frames, name: nil),
                                  dimension: 2, start: 0, length: key.bins, name: nil)
            return g.cast(c, to: .float32, name: nil)
        }
        let m = b.pointwise(outputs[0], w: magOut.w, b: magOut.b, cout: 1)
        let ri = b.pointwise(outputs[1], w: phaseOut.w, b: phaseOut.b, cout: 2)
        let re = g.sliceTensor(ri, dimension: 3, start: 0, length: 1, name: nil)
        let im = g.sliceTensor(ri, dimension: 3, start: 1, length: 1, name: nil)
        let pha = g.atan2(withPrimaryTensor: crop(im), secondaryTensor: crop(re), name: nil)
        return build(g, feeds: [input], targets: [crop(m), pha])
    }

    private func shape(_ s: [Int]) -> [NSNumber] { s.map { NSNumber(value: $0) } }

    struct Compiled {
        let executable: MPSGraphExecutable
        let feeds: [MPSGraphTensor]

        func run(queue: MTLCommandQueue, inputs: [MPSGraphTensorData]) -> [MPSGraphTensorData] {
            let byTensor = Dictionary(uniqueKeysWithValues: zip(feeds, inputs))
            let ordered = executable.feedTensors!.map { byTensor[$0]! }
            return executable.run(with: queue, inputs: ordered, results: nil, executionDescriptor: nil)
        }
    }

    /// Graph-building helpers over NHWC (1, time, freq, channels) tensors.
    private struct Builder {
        let graph: MPSGraph
        let dtype: MPSDataType

        func storage(_ x: MPSGraphTensor) -> MPSGraphTensor {
            dtype == .float32 ? x : graph.cast(x, to: dtype, name: nil)
        }

        func constant(_ v: [Float], _ shape: [Int], type: MPSDataType? = nil) -> MPSGraphTensor {
            let t = graph.constant(v.withUnsafeBytes { Data($0) }, shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
            let target = type ?? dtype
            return target == .float32 ? t : graph.cast(t, to: target, name: nil)
        }

        /// A conv with symmetric zero padding (PyTorch `padding=` or an explicit pad).
        func conv(_ c: Conv, _ x: MPSGraphTensor, padT: Int = 0, padF: Int = 0, strideT: Int = 1, strideF: Int = 1)
            -> MPSGraphTensor
        {
            let d = MPSGraphConvolution2DOpDescriptor(
                strideInX: strideF, strideInY: strideT, dilationRateInX: 1, dilationRateInY: c.dilation, groups: 1,
                paddingLeft: padF, paddingRight: padF, paddingTop: padT, paddingBottom: padT, paddingStyle: .explicit,
                dataLayout: .NHWC, weightsLayout: .HWIO)!
            let y = graph.convolution2D(x, weights: constant(c.weight, [c.kT, c.kF, c.cin, c.cout]), descriptor: d, name: nil)
            return graph.addition(y, constant(c.bias, [c.cout]), name: nil)
        }

        /// InstanceNorm2d (affine, statistics over time and frequency in float32) and PReLU.
        func normAct(_ a: NormAct, _ x: MPSGraphTensor) -> MPSGraphTensor {
            let x32 = dtype == .float32 ? x : graph.cast(x, to: .float32, name: nil)
            let mean = graph.mean(of: x32, axes: [1, 2], name: nil)
            let variance = graph.variance(of: x32, mean: mean, axes: [1, 2], name: nil)
            let scale = graph.multiplication(
                graph.reverseSquareRoot(with: graph.addition(variance, graph.constant(1e-5, dataType: .float32), name: nil), name: nil),
                constant(a.gamma, [a.gamma.count], type: .float32), name: nil)
            let shift = graph.subtraction(
                constant(a.beta, [a.beta.count], type: .float32), graph.multiplication(mean, scale, name: nil), name: nil)
            let n = graph.addition(
                graph.multiplication(x, storage(scale), name: nil), storage(shift), name: nil)
            let zero = graph.constant(0, dataType: dtype)
            let negative = graph.multiplication(graph.minimum(n, zero, name: nil), constant(a.slope, [a.slope.count]), name: nil)
            return graph.addition(graph.maximum(n, zero, name: nil), negative, name: nil)
        }

        func layer(_ l: Layer, _ x: MPSGraphTensor, strideT: Int = 1, strideF: Int = 1) -> MPSGraphTensor {
            // PyTorch padding=get_padding_2d((3, 3), (dilation, 1)) for 3x3 convs, none otherwise.
            let padT = l.conv.kT == 3 ? l.conv.dilation : 0
            let padF = l.conv.kF == 3 && l.conv.kT == 3 ? 1 : 0
            return normAct(l.act, conv(l.conv, x, padT: padT, padF: padF, strideT: strideT, strideF: strideF))
        }

        /// Four layers, each fed the concatenation of every earlier output (newest first).
        func dense(_ layers: [Layer], _ x: MPSGraphTensor) -> MPSGraphTensor {
            var skip = x, out = x
            for l in layers {
                out = layer(l, skip)
                skip = graph.concatTensors([out, skip], dimension: 3, name: nil)
            }
            return out
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
