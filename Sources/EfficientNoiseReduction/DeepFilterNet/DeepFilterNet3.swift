import Accelerate
import Foundation

/// DeepFilterNet3 (Schröter et al., Interspeech 2023) on the CPU with Accelerate.
///
/// 48 kHz full-band, 20 ms window / 10 ms hop, two frames of look-ahead: 40 ms of
/// algorithmic latency. The weights (MIT / Apache-2.0) ship with the package.
public final class DeepFilterNet3: Sendable {
    public static let sampleRate = 48000
    static let fftSize = 960
    static let hop = 480
    static let erbBands = 32
    static let dfBins = 96
    static let dfOrder = 5
    static let lookAhead = 2  // conv and deep-filter look-ahead, in frames
    static let alpha: Float = 0.99  // exp(-hop / sr / 1 s), rounded as DeepFilterNet does

    let erbWidths: [Int]
    let erbInverse: [Float]  // (32, 481)

    let erbConv0: ConvBlock, erbConv1: ConvBlock, erbConv2: ConvBlock, erbConv3: ConvBlock
    let dfConv0: ConvBlock, dfConv1: ConvBlock
    let dfFcEmb: GroupedLinear
    let encGRU: SqueezedGRU
    let decGRU: SqueezedGRU
    let conv3p: ConvBlock, convt3: ConvBlock, conv2p: ConvBlock, convt2: ConvBlock
    let conv1p: ConvBlock, convt1: ConvBlock, conv0p: ConvBlock, conv0Out: ConvBlock
    let dfGRU: SqueezedGRU
    let dfSkip: GroupedLinear
    let dfConvp: ConvBlock
    let dfOut: GroupedLinear

    /// Loads the bundled checkpoint, or `url` if given.
    public init(weights url: URL? = nil) throws {
        guard let url = url ?? Bundle.module.url(forResource: "DeepFilterNet3", withExtension: "safetensors") else {
            throw EnhancementError.modelNotFound("DeepFilterNet3.safetensors missing from the bundle")
        }
        let w = try SafeTensors(url: url)
        erbWidths = try w("erb_widths").values.map { Int($0) }
        erbInverse = try w("mask.erb_inv_fb").values

        func conv(
            _ p: String, conv: Int, pointwise: Bool, groups: Int, stride: Int = 1, pad: Int = 1,
            act: Activation = .relu
        ) throws -> ConvBlock {
            try ConvBlock(
                w, conv: "\(p).\(conv).weight", pointwise: pointwise ? "\(p).\(conv + 1).weight" : nil,
                norm: "\(p).\(conv + (pointwise ? 2 : 1))", groups: groups, strideF: stride, padF: pad, activation: act)
        }
        erbConv0 = try conv("enc.erb_conv0", conv: 1, pointwise: false, groups: 1)
        erbConv1 = try conv("enc.erb_conv1", conv: 0, pointwise: true, groups: 64, stride: 2)
        erbConv2 = try conv("enc.erb_conv2", conv: 0, pointwise: true, groups: 64, stride: 2)
        erbConv3 = try conv("enc.erb_conv3", conv: 0, pointwise: true, groups: 64)
        dfConv0 = try conv("enc.df_conv0", conv: 1, pointwise: true, groups: 2)
        dfConv1 = try conv("enc.df_conv1", conv: 0, pointwise: true, groups: 64, stride: 2)
        dfFcEmb = try GroupedLinear(w, "enc.df_fc_emb.0.weight", activation: .relu)
        encGRU = try SqueezedGRU(w, prefix: "enc.emb_gru", layers: 1, hasOutput: true)

        decGRU = try SqueezedGRU(w, prefix: "erb_dec.emb_gru", layers: 2, hasOutput: true)
        conv3p = try conv("erb_dec.conv3p", conv: 0, pointwise: false, groups: 64, pad: 0)
        convt3 = try conv("erb_dec.convt3", conv: 0, pointwise: true, groups: 64)
        conv2p = try conv("erb_dec.conv2p", conv: 0, pointwise: false, groups: 64, pad: 0)
        conv1p = try conv("erb_dec.conv1p", conv: 0, pointwise: false, groups: 64, pad: 0)
        conv0p = try conv("erb_dec.conv0p", conv: 0, pointwise: false, groups: 64, pad: 0)
        func convT(_ p: String) throws -> ConvBlock {
            try ConvBlock(
                w, conv: "\(p).0.weight", pointwise: "\(p).1.weight", norm: "\(p).2", kind: .transposed, groups: 64,
                strideF: 2, padF: 1, activation: .relu)
        }
        convt2 = try convT("erb_dec.convt2")
        convt1 = try convT("erb_dec.convt1")
        conv0Out = try conv("erb_dec.conv0_out", conv: 0, pointwise: false, groups: 1, act: .sigmoid)

        dfGRU = try SqueezedGRU(w, prefix: "df_dec.df_gru", layers: 2, hasOutput: false)
        dfSkip = try GroupedLinear(w, "df_dec.df_skip.weight", activation: .none)
        dfConvp = try conv("df_dec.df_convp", conv: 1, pointwise: true, groups: 2, pad: 0)
        dfOut = try GroupedLinear(w, "df_dec.df_out.0.weight", activation: .tanh)
    }

    /// Recurrent and convolutional context carried between chunks.
    struct State {
        var erbConv0: [Float]
        var dfConv0: [Float]
        var dfConvp: [Float]
        var enc: [[Float]]
        var dec: [[Float]]
        var df: [[Float]]

        init(_ n: DeepFilterNet3) {
            erbConv0 = .init(repeating: 0, count: n.erbConv0.span * DeepFilterNet3.erbBands * 1)
            dfConv0 = .init(repeating: 0, count: n.dfConv0.span * DeepFilterNet3.dfBins * 2)
            dfConvp = .init(repeating: 0, count: n.dfConvp.span * DeepFilterNet3.dfBins * 64)
            enc = n.encGRU.zeroState()
            dec = n.decGRU.zeroState()
            df = n.dfGRU.zeroState()
        }
    }

    /// Network on T feature frames: `erb` (T, 32), `spec` (T, 96, 2). Returns the ERB gains
    /// (T, 32) and deep-filter coefficients (T, 96, order, 2).
    func callAsFunction(erb: [Float], spec: [Float], frames T: Int, state s: inout State) -> (
        mask: [Float], coefs: [Float]
    ) {
        var none: [Float] = []
        let e0 = erbConv0(erb, frames: T, bins: 32, cache: &s.erbConv0)  // (T, 32, 64)
        let e1 = erbConv1(e0, frames: T, bins: 32, cache: &none)  // (T, 16, 64)
        let e2 = erbConv2(e1, frames: T, bins: 16, cache: &none)  // (T, 8, 64)
        let e3 = erbConv3(e2, frames: T, bins: 8, cache: &none)  // (T, 8, 64)
        let c0 = dfConv0(spec, frames: T, bins: 96, cache: &s.dfConv0)  // (T, 96, 64)
        let c1 = dfConv1(c0, frames: T, bins: 96, cache: &none)  // (T, 48, 64)
        let cemb = dfFcEmb(c1, rows: T)  // (T, 512)
        let emb = encGRU(zip(e3, cemb).map(+), frames: T, state: &s.enc)  // (T, 512)

        let d = decGRU(emb, frames: T, state: &s.dec)  // (T, 8·64)
        var x = convt3(zip(conv3p(e3, frames: T, bins: 8, cache: &none), d).map(+), frames: T, bins: 8, cache: &none)
        x = convt2(zip(conv2p(e2, frames: T, bins: 8, cache: &none), x).map(+), frames: T, bins: 8, cache: &none)
        x = convt1(zip(conv1p(e1, frames: T, bins: 16, cache: &none), x).map(+), frames: T, bins: 16, cache: &none)
        let m = conv0Out(zip(conv0p(e0, frames: T, bins: 32, cache: &none), x).map(+), frames: T, bins: 32, cache: &none)

        let c = zip(dfGRU(emb, frames: T, state: &s.df), dfSkip(emb, rows: T)).map(+)  // (T, 256)
        let coefs = zip(dfOut(c, rows: T), dfConvp(c0, frames: T, bins: 96, cache: &s.dfConvp)).map(+)  // (T, 96·10)
        return (m, coefs)
    }
}
