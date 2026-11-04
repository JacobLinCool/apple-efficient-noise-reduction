import Accelerate
import Foundation

/// Streaming DeepFilterNet3 on one 48 kHz mono stream, matching `df.enhance.enhance`
/// sample for sample: push any block size, get back the samples that are final.
///
/// Not thread-safe: call from one thread at a time.
public final class DeepFilterNet3Enhancer: SpeechEnhancer {
    public let model: DeepFilterNet3
    /// Network frames (10 ms hops) per call; 1 is frame-by-frame.
    public let chunkFrames: Int
    public var sampleRate: Int { DeepFilterNet3.sampleRate }

    private typealias M = DeepFilterNet3
    private let window: [Float]
    private let fft: STFT
    private var state: DeepFilterNet3.State

    // Analysis.
    private var pending: [Float] = []  // input not yet a full hop
    private var analysisMemory = [Float](repeating: 0, count: M.hop)
    private var received = 0
    private var erbNorm: [Float]
    private var unitNorm: [Float]
    private var framesAnalysed = 0

    // Spectra by frame index (complex interleaved, 481 bins), kept while deep filtering needs them.
    private var spectra: [Int: [Float]] = [:]
    // Features waiting for the network; network input k is feature frame k + lookAhead.
    private var queuedErb: [Float] = []
    private var queuedSpec: [Float] = []
    private var queued = 0
    private var framesOut = 0  // network outputs produced = spectral frames synthesised

    // Synthesis.
    private var synthesisMemory = [Float](repeating: 0, count: M.hop)
    private var output: [Float] = []  // synthesised samples not yet returned
    private var synthesised = 0  // total samples synthesised
    private var emitted = 0
    private var finished = false

    public init(model: DeepFilterNet3, chunkFrames: Int = 1) {
        self.model = model
        self.chunkFrames = max(1, chunkFrames)
        state = .init(model)
        // Vorbis window: sin(π/2 · sin²(π (n + 0.5) / N)), in double like libDF.
        window = (0 ..< M.fftSize).map { i in
            let s = sin(0.5 * Double.pi * (Double(i) + 0.5) / Double(M.fftSize / 2))
            return Float(sin(0.5 * Double.pi * s * s))
        }
        fft = STFT(size: M.fftSize)
        erbNorm = (0 ..< M.erbBands).map { -60 + Float($0) * (-30) / Float(M.erbBands - 1) }
        unitNorm = (0 ..< M.dfBins).map { 0.001 + Float($0) * (0.0001 - 0.001) / Float(M.dfBins - 1) }
    }

    /// Input to output delay, excluding compute: the 20 ms window, two look-ahead hops
    /// and the chunk buffering.
    public var latencySeconds: Double {
        Double(M.fftSize + (M.lookAhead + chunkFrames - 1) * M.hop) / Double(M.sampleRate)
    }

    public func warmUp() {}

    public func process(_ samples: UnsafeBufferPointer<Float>) -> [Float] {
        precondition(!finished, "process(_:) after finish()")
        received += samples.count
        pending.append(contentsOf: samples)
        analyse()
        while queued >= chunkFrames { runNetwork(frames: chunkFrames) }
        // Output sample i is synthesis sample i + hop (the STFT's own delay).
        return emit(upTo: synthesised - M.hop)
    }

    public func finish() -> [Float] {
        precondition(!finished, "finish() called twice")
        finished = true
        // enhance(): pad the input with n_fft zeros; a trailing partial hop is dropped.
        pending.append(contentsOf: [Float](repeating: 0, count: M.fftSize))
        analyse()
        // The feature shift leaves `lookAhead` zero feature frames at the end.
        for _ in 0 ..< M.lookAhead {
            queuedErb += [Float](repeating: 0, count: M.erbBands)
            queuedSpec += [Float](repeating: 0, count: 2 * M.dfBins)
            queued += 1
        }
        while queued > 0 { runNetwork(frames: min(chunkFrames, queued)) }
        var out = emit(upTo: min(received, synthesised - M.hop))
        if emitted < received {
            out += [Float](repeating: 0, count: received - emitted)
            emitted = received
        }
        return out
    }

    private func analyse() {
        var used = 0
        while pending.count - used >= M.hop {
            let frame = pending[used ..< used + M.hop]
            used += M.hop
            var buffer = [Float](repeating: 0, count: M.fftSize)
            for i in 0 ..< M.hop {
                buffer[i] = analysisMemory[i] * window[i]
                buffer[M.hop + i] = frame[frame.startIndex + i] * window[M.hop + i]
            }
            analysisMemory = Array(frame)
            var spec = fft.forward(buffer)  // interleaved re, im
            let norm = 1 / Float(M.fftSize)  // libDF's wnorm for 50% overlap
            vDSP_vsmul(spec, 1, [norm], &spec, 1, vDSP_Length(spec.count))
            spectra[framesAnalysed] = spec
            if framesAnalysed >= M.lookAhead { queueFeatures(spec) } else { updateNorms(spec) }
            framesAnalysed += 1
        }
        pending.removeFirst(used)
    }

    /// ERB band energies in dB with an exponential mean norm, and the first 96 bins
    /// divided by the square root of an exponential mean magnitude.
    private func features(_ spec: [Float]) -> (erb: [Float], spec: [Float]) {
        let a = M.alpha
        var erb = [Float](repeating: 0, count: M.erbBands)
        var bin = 0
        for (b, width) in model.erbWidths.enumerated() {
            var power: Float = 0
            let k = 1 / Float(width)
            for j in bin ..< bin + width { power += (spec[2 * j] * spec[2 * j] + spec[2 * j + 1] * spec[2 * j + 1]) * k }
            bin += width
            let db = log10f(power + 1e-10) * 10
            erbNorm[b] = db * (1 - a) + erbNorm[b] * a
            erb[b] = (db - erbNorm[b]) / 40
        }
        var feat = [Float](repeating: 0, count: 2 * M.dfBins)
        for f in 0 ..< M.dfBins {
            let (re, im) = (spec[2 * f], spec[2 * f + 1])
            unitNorm[f] = (re * re + im * im).squareRoot() * (1 - a) + unitNorm[f] * a
            let s = unitNorm[f].squareRoot()
            feat[2 * f] = re / s
            feat[2 * f + 1] = im / s
        }
        return (erb, feat)
    }

    private func updateNorms(_ spec: [Float]) { _ = features(spec) }

    private func queueFeatures(_ spec: [Float]) {
        let (erb, feat) = features(spec)
        queuedErb += erb
        queuedSpec += feat
        queued += 1
    }

    private func runNetwork(frames T: Int) {
        let erb = Array(queuedErb.prefix(T * M.erbBands))
        let spec = Array(queuedSpec.prefix(T * 2 * M.dfBins))
        queuedErb.removeFirst(T * M.erbBands)
        queuedSpec.removeFirst(T * 2 * M.dfBins)
        queued -= T
        let (mask, coefs) = model(erb: erb, spec: spec, frames: T, state: &state)
        let bins = M.fftSize / 2 + 1
        var gains = [Float](repeating: 0, count: T * bins)
        sgemm(T, bins, M.erbBands, mask, lda: M.erbBands, model.erbInverse, ldb: bins, &gains, ldc: bins)
        for t in 0 ..< T {
            let k = framesOut + t
            let noisy = spectra[k]!
            var enhanced = [Float](repeating: 0, count: 2 * bins)
            for f in M.dfBins ..< bins {
                let g = gains[t * bins + f]
                enhanced[2 * f] = noisy[2 * f] * g
                enhanced[2 * f + 1] = noisy[2 * f + 1] * g
            }
            // Deep filter: Σ_n X[k - 2 + n, f] · C[k, f, n] over the lowest 96 bins.
            for n in 0 ..< M.dfOrder {
                guard let x = spectra[k - M.lookAhead + n] else { continue }  // zero padding
                for f in 0 ..< M.dfBins {
                    let c = ((t * M.dfBins + f) * M.dfOrder + n) * 2
                    let (cr, ci) = (coefs[c], coefs[c + 1])
                    let (xr, xi) = (x[2 * f], x[2 * f + 1])
                    enhanced[2 * f] += xr * cr - xi * ci
                    enhanced[2 * f + 1] += xr * ci + xi * cr
                }
            }
            synthesise(enhanced)
            spectra[k - M.lookAhead] = nil
        }
        framesOut += T
    }

    private func synthesise(_ spec: [Float]) {
        var x = fft.inverse(spec)
        vDSP_vmul(x, 1, window, 1, &x, 1, vDSP_Length(M.fftSize))
        // The first hop is the STFT's own delay; output sample i is synthesis sample i + hop.
        if synthesised > 0 { for i in 0 ..< M.hop { output.append(x[i] + synthesisMemory[i]) } }
        synthesisMemory = Array(x[M.hop...])
        synthesised += M.hop
    }

    private func emit(upTo limit: Int) -> [Float] {
        guard limit > emitted else { return [] }
        let n = limit - emitted
        let out = Array(output.prefix(n))
        output.removeFirst(n)
        emitted = limit
        return out
    }
}

/// Real FFT of an even length via vDSP, with numpy/realfft conventions: the forward
/// transform is unnormalised, the inverse is the plain (unnormalised) sum.
final class STFT: @unchecked Sendable {
    let size: Int
    private let forwardSetup: vDSP_DFT_Setup
    private let inverseSetup: vDSP_DFT_Setup

    init(size: Int) {
        self.size = size
        forwardSetup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(size), .FORWARD)!
        inverseSetup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(size), .INVERSE)!
    }

    deinit {
        vDSP_DFT_DestroySetup(forwardSetup)
        vDSP_DFT_DestroySetup(inverseSetup)
    }

    /// Real signal → size/2 + 1 bins, interleaved (re, im).
    func forward(_ x: [Float]) -> [Float] {
        let h = size / 2
        var ir = [Float](repeating: 0, count: h), ii = ir, or = ir, oi = ir
        for k in 0 ..< h {
            ir[k] = x[2 * k]
            ii[k] = x[2 * k + 1]
        }
        vDSP_DFT_Execute(forwardSetup, ir, ii, &or, &oi)
        // zrop returns twice the DFT, with the real Nyquist bin packed into oi[0].
        var out = [Float](repeating: 0, count: 2 * (h + 1))
        out[0] = or[0] / 2
        out[2 * h] = oi[0] / 2
        for k in 1 ..< h {
            out[2 * k] = or[k] / 2
            out[2 * k + 1] = oi[k] / 2
        }
        return out
    }

    /// size/2 + 1 interleaved bins → real signal (imaginary parts of DC and Nyquist ignored).
    func inverse(_ spec: [Float]) -> [Float] {
        let h = size / 2
        var ir = [Float](repeating: 0, count: h), ii = ir, or = ir, oi = ir
        ir[0] = spec[0]
        ii[0] = spec[2 * h]
        for k in 1 ..< h {
            ir[k] = spec[2 * k]
            ii[k] = spec[2 * k + 1]
        }
        vDSP_DFT_Execute(inverseSetup, ir, ii, &or, &oi)
        // zrop's inverse of the packed half spectrum is already the full-spectrum sum.
        var x = [Float](repeating: 0, count: size)
        for k in 0 ..< h {
            x[2 * k] = or[k]
            x[2 * k + 1] = oi[k]
        }
        return x
    }
}
