# Efficient Noise Reduction for Apple Silicon

在 Apple Silicon 上以 Swift 執行兩個串流語音增強模型，兩者共用同一套串流介面：

- **Real-time RE-USE**（NVIDIA）：通用語音增強，處理噪音、殘響、頻寬受限、codec 失真等多種退化，支援任意取樣率。在 GPU 上執行，卷積用 MPSGraph，Mamba 用 MLX 與自訂 Metal kernel。延遲與運算量都能在執行時選擇，從逐幀即時到整檔離線。
- **DeepFilterNet3**：48 kHz 降噪，固定 40 ms 延遲。以 Accelerate 在 CPU 上執行，不啟動 GPU，權重直接附在套件裡。

不論怎麼切塊，兩者的輸出都和整段一次處理相同，數值也與各自的官方 PyTorch 實作一致。

- RE-USE 論文：[One Model, Many Latencies](https://arxiv.org/abs/2606.25621)、[Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement](https://arxiv.org/abs/2603.02641)；權重 [`nvidia/Real-time_RE-USE`](https://huggingface.co/nvidia/Real-time_RE-USE)，固定在 revision `4c0c186`
- DeepFilterNet3：[DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement](https://arxiv.org/abs/2305.08227)；權重取自官方 [`deepfilternet`](https://github.com/Rikorose/DeepFilterNet) 0.5.6

## 安裝

以 Swift Package Manager 加入相依：

```swift
.package(url: "https://github.com/JacobLinCool/apple-efficient-noise-reduction", from: "0.1.0"),
// target 裡
.product(name: "EfficientNoiseReduction", package: "apple-efficient-noise-reduction"),
```

需要 macOS 14／iOS 17 以上，並以 Xcode（或 `xcodebuild`）建置：MLX 的 Metal shader 只能透過 Xcode 的建置系統編譯，`swift build` 編出來的程式在執行 RE-USE 時會找不到 shader。iOS 可以編譯，但還沒有在裝置上實測。

## 快速開始

以命令列工具試用（在 repo 內）：

```bash
make build                                   # .build/xcode/Build/Products/Release/noise-reduction
CLI=.build/xcode/Build/Products/Release/noise-reduction
$CLI download                                # 下載 RE-USE 權重到 ~/Library/Application Support/EfficientNoiseReduction/
$CLI enhance noisy.wav clean.wav             # 預設：exit layer 8、無 look-ahead、0.5 秒延遲
$CLI enhance noisy.wav clean.wav --offline -e 12 -l 2   # 品質最好的離線設定
$CLI live --monitor --latency 0.1            # 麥克風即時增強並播放（請戴耳機）
$CLI bench --sample-rate 16000               # 在這台機器上量測各種設定
$CLI enhance noisy.wav clean.wav --engine dfn3          # DeepFilterNet3（輸入會重採樣到 48 kHz）
$CLI enhance-batch noisy/ enhanced/ --engine dfn3       # 整個資料夾，模型只載入一次
```

RE-USE 也會使用 Hugging Face cache 裡已下載的權重，或是 `REUSE_MODEL_DIR` 指定的資料夾。DeepFilterNet3 不需要下載。

## 延遲怎麼選

延遲由三個彼此獨立的選項決定：

| 選項 | 範圍 | 影響 |
|---|---|---|
| `exitLayer` | 3–12 | 使用前幾層 Mamba 和對應的 decoder。越深品質越好，運算量也越大 |
| `lookAhead` | 0–2 | 每多等一個 20 ms 的未來 frame，演算法延遲就多 20 ms；對 ASR 辨識率幫助最明顯 |
| `chunk` | 1 frame 到離線 | 每次送進網路的 frame 數。只影響輸出出現的時間和吞吐量，不影響輸出內容 |

演算法延遲是 40 ms 的視窗、look-ahead 與分塊緩衝的總和。`.latency(0.5)` 會選出在這個預算內最大的區塊；實際延遲還要再加上該區塊的運算時間，見下表。

下表是在 M5 Pro 上以預設精度量測的結果（接電源、一般電源模式，背景負載約 3；`noise-reduction bench`，每種設定處理 20 秒音訊，以 10 ms 為單位推入）。RTF 是運算時間除以音訊長度，越小越好。

| 取樣率 | exit layer | 每次 frames | 演算法延遲 | 每次運算（平均／p95） | RTF |
|---|---|---|---|---|---|
| 16 kHz | 3 | 1 | 40 ms | 2.9／3.2 ms | 0.146 |
| 16 kHz | 8 | 1 | 40 ms | 3.7／3.9 ms | 0.186 |
| 16 kHz | 12 | 1 | 40 ms | 4.4／4.6 ms | 0.220 |
| 16 kHz | 8 | 24 | 500 ms | 7.8／9.2 ms | 0.026 |
| 16 kHz | 12 | 24 | 500 ms | 9.0／9.2 ms | 0.028 |
| 16 kHz | 12 | 離線（100） | — | 29.4／40.0 ms | 0.024 |
| 48 kHz | 8 | 1 | 40 ms | 5.2／5.5 ms | 0.259 |
| 48 kHz | 12 | 1 | 40 ms | 6.4／6.7 ms | 0.322 |
| 48 kHz | 8 | 24 | 500 ms | 17.6／18.7 ms | 0.046 |
| 48 kHz | 12 | 24 | 500 ms | 24.4／25.6 ms | 0.061 |
| 48 kHz | 12 | 離線（33） | — | 30.5／31.5 ms | 0.056 |

逐幀處理時，每一步都遠在 20 ms 的 hop 內完成，所以 40 ms 加上運算時間的設定也能即時運作；此時成本主要是 kernel 派送與 GPU 同步的固定開銷，所以 RTF 較高。區塊放大到約 0.5 秒後，固定開銷被攤平，16 kHz 下約為即時速度的 35–40 倍，48 kHz 下約 15–20 倍。

記憶體與能耗以約 65 秒的語音量測（`noise-reduction enhance` 印出記憶體；功耗取自 SoC 的 IOReport 能耗計數器，為執行期間相對閒置時的額外功耗）。「即時」是依真實時間節奏餵入音訊，GPU 在區塊之間是閒置的。

| 設定（exit layer 8） | process footprint 峰值 | 額外 GPU 功耗 | 每分鐘音訊能耗 |
|---|---|---|---|
| 16 kHz，即時，0.5 s | 0.54 GB | 0.13 W | — |
| 16 kHz，即時，逐幀 40 ms | 0.27 GB | 0.39 W | — |
| 16 kHz，離線 | 0.99 GB | 17 W（全速，SoC 合計 31 W） | 約 38 J |
| 48 kHz，即時，0.5 s | 1.2 GB | 0.76 W | — |
| 48 kHz，離線 | 1.2 GB | 19 W（全速，SoC 合計 29 W） | 約 79 J |

即時模式的額外功耗很小，SoC 合計值會被背景程式的波動淹沒，所以只列 GPU 這一項；離線模式全速執行、時間短，合計值比較可靠。所有設定的記憶體都在 1.5 GB 以內（exit layer 12 也是）。

## RE-USE 與 DeepFilterNet3

品質以 VoiceBank-DEMAND 測試集（824 句，16 kHz；`reference/quality_vbd.py`）評估。兩個模型都沒有用這個資料集訓練。DeepFilterNet3 的輸入先升頻到 48 kHz，輸出再降回 16 kHz 評分。

| 系統 | 演算法延遲 | PESQ | ESTOI | SI-SDR | DNSMOS OVRL | SIG | BAK |
|---|---|---|---|---|---|---|---|
| 未處理 | — | 1.97 | 0.787 | 8.5 | 2.70 | 3.35 | 3.13 |
| DeepFilterNet3 | 40 ms | 2.69 | 0.843 | 17.7 | 3.22 | 3.46 | 4.11 |
| RE-USE，exit 3 | 40 ms | 2.59 | 0.848 | 18.1 | 3.18 | 3.45 | 4.06 |
| RE-USE，exit 8 | 40 ms | 2.74 | 0.859 | 18.6 | 3.24 | 3.49 | 4.11 |
| RE-USE，exit 8，look-ahead 1 | 60 ms | 2.80 | 0.862 | 18.7 | 3.26 | 3.50 | 4.12 |
| RE-USE，exit 12，look-ahead 2 | 80 ms | 2.83 | 0.862 | 18.4 | 3.27 | 3.51 | 4.13 |

這些數字與 RE-USE 論文表 II 的對應值相差不到 0.03 PESQ（未處理 1.97、DeepFilterNet3 2.71、exit 8 為 2.76／2.82），可以當作兩個移植都正確的另一個佐證。只看降噪時，兩者的品質很接近：同樣是 40 ms 延遲，RE-USE exit 8 略好，exit 3 略差；RE-USE 的優勢在於多給延遲就能再提升品質，以及 DeepFilterNet3 沒有處理的殘響、頻寬受限、codec 失真等退化。

效率則差了一個數量級以上。以下是 48 kHz、約 65 秒語音的量測（接電源，條件同上）：

| 設定 | RTF | process footprint 峰值 | 備註 |
|---|---|---|---|
| DeepFilterNet3，即時，逐幀（10 ms 區塊） | 0.11 | 56 MB | 只用 CPU |
| DeepFilterNet3，離線 | 0.020 | 78 MB | 全速時 SoC 額外約 10 W，每分鐘音訊約 13 J |
| RE-USE exit 3，即時，0.5 s | 0.076 | 0.66 GB | |
| RE-USE exit 8，即時，逐幀（20 ms 區塊） | 0.72 | 0.46 GB | |
| RE-USE exit 8，即時，0.5 s | 0.092 | 1.2 GB | |
| RE-USE exit 8，離線 | 0.039 | 1.2 GB | 全速時 SoC 額外約 29 W，每分鐘音訊約 79 J |

DeepFilterNet3 每 10 ms 的運算量約 3.4M MAC，RE-USE exit 8 在 48 kHz 下每 20 ms 約 1.5G MAC，差了兩個數量級以上，這直接反映在記憶體與能耗上。需要長時間在背景執行、或只處理一般噪音時，DeepFilterNet3 幾乎不佔資源；需要處理更多種退化、或想以延遲換取品質時，再用 RE-USE。

## 在程式中使用

```swift
import EfficientNoiseReduction

let model = try ReUseModel(directory: ModelLocator.locate())

// 整檔處理
let clean = try model.enhance(samples, sampleRate: 16000, exitLayer: 12, lookAhead: 2)

// 串流：任意大小的區塊進，已確定的輸出出；最後 finish() 補齊尾端
let stream = try StreamingEnhancer(
    model: model, sampleRate: 48000,
    configuration: .init(exitLayer: 8, lookAhead: 1, chunk: .latency(0.5)))
stream.warmUp()
let out = stream.process(block)
let tail = stream.finish()

// DeepFilterNet3：同樣的串流介面，48 kHz，chunkFrames 以 10 ms 為單位
let dfn3 = DeepFilterNet3Enhancer(model: try DeepFilterNet3(), chunkFrames: 1)

// 音訊 callback：任何 SpeechEnhancer 都能放進來；寫入與讀取只做 memcpy，網路在自己的 thread 上跑
let rt = RealtimeEnhancer(dfn3)
rt.write(captured)        // 錄音 callback
rt.read(into: playback)   // 播放 callback
rt.stop()
```

`StreamingEnhancer` 和 `DeepFilterNet3Enhancer` 都實作 `SpeechEnhancer`，都不是 thread-safe，一個 stream 請只在一個 thread 上使用。模型（`ReUseModel`、`DeepFilterNet3`）可以由多個 stream 共用；RE-USE 的同一組 (exit layer, look-ahead) 網路只會載入一次。

## 正確性

`reference/golden.py` 直接執行 NVIDIA 原始的 PyTorch 模型程式碼，只把 CUDA 專用的 `mamba_ssm` 換成純 PyTorch 的 reference scan，產生 `Tests/EfficientNoiseReductionTests/Fixtures/` 裡的 golden 資料（16 kHz 與 44.1 kHz、多組 exit layer 與 look-ahead）。同一支腳本也確認 NVIDIA 的逐幀 online 路徑與 offline 路徑一致，這正是本實作可以任意分塊的前提。

| 測試 | 結果 |
|---|---|
| 網路輸出，餵入 NVIDIA 的 STFT 特徵（`.exact`） | magnitude 最大誤差約 2e-6，phase 加權誤差約 1e-6 rad |
| 整條管線的波形，相對 NVIDIA（含完整 6.5 秒錄音） | 39–47 dB SNR |
| 逐幀、7 幀、0.5 秒分塊，相對整段處理 | 119 dB 以上（MPSGraph 會依形狀選擇不同的卷積演算法，所以不是逐位相同） |
| 即時管線（真實時間節奏寫入與讀取） | 等於整段處理結果加上固定延遲，無 underrun |

網路本身與參考實作一致；波形上剩下的差距來自輸入端，近乎無聲頻點的 phase 會因 FFT 實作不同而在 rounding 層級改變，而網路輸出對它敏感。

精度預設為 `.fast`：卷積與 Mamba 都以 fp16 執行，SSM state 與殘差流維持 fp32。輸出與 `.exact` 相差 55–57 dB SNR；完整的 6.5 秒測試錄音上，兩者相對 NVIDIA 分別是 40.3 與 40.8 dB，差距比兩種 FFT 實作之間還小。需要和參考實作比對時使用 `.exact`（CLI 加 `--exact`），此時所有運算都是完整的 fp32。

DeepFilterNet3 由 `reference/golden_dfn3.py` 以官方 `deepfilternet` 套件產生 golden 資料與權重檔：

| 測試 | 結果 |
|---|---|
| STFT，相對 libDF | 最大誤差 2.5e-10 |
| 網路輸出（ERB gain、deep-filter 係數），餵入官方特徵 | 最大誤差約 3e-6 |
| 整條管線的波形，相對 `df.enhance.enhance` | 125–135 dB SNR |
| 逐幀、7 幀與不規則推入大小，相對整段處理 | 136 dB 以上 |

golden 資料含有第三方錄音與 NVIDIA 模型的輸出，不放在 repo 裡，請在本機產生：

```bash
make golden   # 產生 golden 資料（需要 uv；會下載 RE-USE 權重與 deepfilternet 套件）
make test     # 缺少 golden 資料或 RE-USE 權重的測試會略過
```

## 實作重點

- **卷積交給 MPSGraph，Mamba 交給 MLX**：encoder 與兩個 decoder（dense block、sub-pixel 上採樣、channel LayerNorm、PReLU）是 MPSGraph executable，也就是 Core ML 在 GPU 上使用的同一套卷積實作；`.fast` 時以 fp16 執行，`.exact` 時以 fp32 執行。各層卷積的串流快取一直留在 GPU 上，executable 依 (頻點數, 區塊幀數) 編譯並快取。Mamba blocks 則是一張編譯過的 MLX 圖。每個區塊依序跑 encoder → Mamba → decoder 三段，每段只同步一次。
- **Mamba 在 `.fast` 時以 fp16 執行**：GEMM、conv1d 與 scan 的輸入輸出都是 fp16；SSM state、scan 內部的運算與 block 之間的殘差流維持 fp32。輸出與 `.exact` 相差 55–57 dB SNR，VoiceBank-DEMAND 的分數完全不變。每個 block 結尾的「output projection 之後的 LayerNorm 加殘差」是一個融合的 Metal kernel，MLX 自己的 LayerNorm 在 64 通道時很慢。
- **Mamba 的自訂 Metal kernel**：conv1d 與 SiLU 合成一個 stencil kernel；selective scan 融合 dt_proj、softplus、D skip 與 gate，每個 thread 負責一個 channel，16 個 state 放在暫存器裡，B、C 用 float4 廣播讀取。頻率方向的前向與反向 Mamba 合併成一次 batch，`out_proj` 和 `output_proj` 在載入時摺疊成一個矩陣。
- **所有運算都在原本的 (時間, 頻率, 通道) 排列上進行**：沿時間或頻率掃描只是 stride 不同，不需要轉置。
- **STFT／iSTFT 在 CPU 上執行**：每個 frame 的結果和 batch 大小無關，所以任何分塊方式都得到相同輸出；overlap-add 用 vDSP。
- **DeepFilterNet3 全部在 CPU 上**：模型很小（每 10 ms 約 3.4M MAC），又是逐幀的 GRU，GPU 的派送成本會比運算本身還高。卷積用 im2col 加 BLAS（Accelerate 會用到 AMX），depthwise 卷積直接寫迴圈，BatchNorm 在載入時摺疊；權重自己解析 safetensors，不經過 MLX，所以不會初始化 Metal。

### 為什麼不用 Neural Engine

`reference/ane_experiment.py` 把 RE-USE 的卷積部分（encoder 與兩個 decoder，約佔 40% 運算量）用 coremltools 轉成 fp16 Core ML 模型，分別在各運算單元上計時，並用 SoC 的能耗計數器量功耗（M5 Pro、低耗電模式、16 kHz、每次 24 frames）：

| Core ML 運算單元 | 每次呼叫 | 每次能耗 |
|---|---|---|
| Neural Engine | 7.5 ms | 26 mJ |
| GPU | 2.0 ms | 22 mJ |
| CPU | 11.5 ms | 88 mJ |

Neural Engine 比 GPU 慢三倍多，能耗也沒有比較低；fp16 還會讓 magnitude 輸出產生約 1.5% 的誤差。Mamba 的 selective scan 本來就無法在 Neural Engine 上執行，混合方案每個區塊還要在兩個單元之間交接兩次。因此 RE-USE 維持全部在 GPU 上。

這個實驗也顯示，Core ML 的 GPU 卷積比當時的 MLX 實作快好幾倍，所以卷積部分改用 MPSGraph。在同樣條件下（16 kHz、exit layer 8、低耗電模式），encoder 加上 decoders 的時間如下：

| 實作 | 每次 1 frame | 每次 24 frames | 每次 96 frames |
|---|---|---|---|
| MLX，以 GEMM 做卷積（fp32／TF32） | — | 約 10 ms | 約 27 ms |
| MLX，fp16 加上融合 LayerNorm 的 Metal kernel | 1.3 ms | 3.5 ms | 19 ms |
| Core ML，GPU，fp16（參考） | 1.6 ms | 2.0 ms | 6.5 ms |
| MPSGraph，fp16（目前） | 約 1.9 ms | 約 3 ms | 約 10–12 ms |

只看卷積本身，MPSGraph 和 Core ML 一樣快：單獨把整個卷積堆疊做成一張圖時，24 frames 只要 1.75 ms。整合進管線後還多了兩件事：要和 MLX 的 Mamba 交接資料，而且 encoder、decoder 分成兩次執行，所以小區塊時的實際成本略高。現在每次呼叫的大部分時間花在 Mamba blocks 上。


## 授權

程式碼採 MIT 授權。

- RE-USE 權重由 NVIDIA 以 [NVIDIA One-Way Noncommercial License (NSCLv1)](https://github.com/NVlabs/HMAR/blob/main/LICENSE) 釋出，只能用於非商業的研究與開發；本專案不附帶這份權重，只在執行 `download` 時下載。
- DeepFilterNet3 權重採 MIT／Apache-2.0 雙授權，以 safetensors 格式附在 `Sources/EfficientNoiseReduction/Resources/`。授權全文見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
