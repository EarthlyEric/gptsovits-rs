# GPT-SoVITS Rust 推論器與 OpenAI TTS API 兼容伺服器實作計畫

## 1. 專案目標

開發一個**完全無 Python 執行期依賴（Zero Python Runtime Dependency）**的高效能 Rust 推論引擎與 HTTP 伺服器，直接載入並推論 GPT-SoVITS 模型（支援 v1, v2, v2Pro, v2ProPlus, v3, v4），並提供 100% 兼容 OpenAI OpenAPI 3.1.0 TTS API 規範的介面。

---

## 2. 系統架構

```
                          [ Client / OpenAI SDK ]
                                     │
                        POST /audio/speech, /v1/audio/speech
                        (Bearer Token, JSON Payload)
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │              Axum + Tokio Web Server                   │
        │  • Bearer 認證中介層 (401 錯誤格式完全對齊)            │
        │  • SpeechRequest 與 ErrorResponse OpenAPI 3.1.0 校驗   │
        │  • 支援字串與動態 Voice Object                         │
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │            Pure Rust Text & G2P Frontend               │
        │  • Symbols 表 (Symbols V1: 322, Symbols V2: 732)       │
        │  • 繁簡與數字正規化 (zh_norm, Tone Sandhi)             │
        │  • 中文/英文/日文/韓文/粵語 G2P 音標轉換               │
        │  • Hugging Face RoBERTa Tokenizer (tokenizers crate)   │
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │           Native ONNX Runtime Engine (ort 2.x)         │
        │                                                        │
        │  1. SSL Extractor (cnhubert.onnx) -> 768-d SSL 特徵    │
        │  2. BERT Model (bert.onnx) -> 1024-d 語義特徵          │
        │  3. (v2Pro/v2ProPlus) Speaker Verification (SV) 特徵   │
        │  4. T2S Encoder -> Prompt & Text 矩陣                  │
        │  5. T2S Autoregressive Decoder (KV Cache + Top-K/Top-P │
        │     Temperature, Repetition Penalty Sampler)           │
        │  6. 聲學合成器 (動態分派):                             │
        │     - v1 / v2: SynthesizerTrn (Flow + HiFiGAN, 32kHz)  │
        │     - v2Pro / v2ProPlus: Enhanced VITS + SV (32kHz)    │
        │     - v3: CFM DiT (32-step ODE) + BigVGAN (24kHz)      │
        │     - v4: CFM DiT (8/16-step ODE) + Vocoder (48kHz)    │
        └────────────────────────────┬───────────────────────────┘
                                     │ Raw PCM Waveform
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │           Audio Processing & Encoders                  │
        │  • 語速調節 (0.25x ~ 4.0x，時間拉伸 / 重採樣)          │
        │  • 多格式編碼: MP3, OPUS, AAC, FLAC, WAV, PCM          │
        │  • 輸出串流: 二進位音訊串流或 Server-Sent Events (SSE) │
        └────────────────────────────────────────────────────────┘
```

---

## 3. 模組規劃

```
/home/earthlyeric6/Projects/gptsovits-rs/
├── PLAN.md                     # 本實作計畫
├── AGENT.md                    # Agent 指引與系統維護手冊
├── Cargo.toml                  # 專案依賴
├── config.toml                 # 伺服器與模型路徑配置
├── voices.toml                 # 預設與自訂音色對照表
├── tools/
│   ├── download_models.py      # 全新環境官方預訓練底模下載器
│   └── onnx_exporter.py        # 離線一鍵模型匯出工具 (v1 ~ v4, 支援 uv)
└── src/
    ├── main.rs                 # 程式啟動入口
    ├── config.rs               # 設定解析器
    ├── server/                 # Axum Web 伺服器
    │   ├── mod.rs
    │   ├── schema.rs           # OpenAPI 3.1.0 規範型別
    │   ├── auth.rs             # Bearer Token 認證
    │   ├── error.rs            # 統一錯誤處理 (400, 401, 404, 429, 500)
    │   └── routes.rs           # /audio/speech, /v1/audio/speech, /v1/models, /v1/voices, /health
    ├── text/                   # 純 Rust G2P 與文本前處理
    │   ├── mod.rs
    │   ├── symbols.rs          # 完整 symbols_v1 (322) 與 symbols_v2 (732)
    │   ├── normalizer.rs       # 數字與標點正規化
    │   ├── g2p.rs              # 拼音、變調、音素轉換
    │   ├── tokenizer.rs        # Hugging Face RoBERTa tokenizer 封裝
    │   └── bert_align.rs       # word2ph 音標維度特徵擴展
    ├── audio/                  # 音訊處理與轉碼
    │   ├── mod.rs
    │   ├── resample.rs         # Rubato 高品質重採樣
    │   ├── speed.rs            # 語速調節 (0.25x ~ 4.0x)
    │   └── encoder.rs          # WAV, PCM, MP3, OPUS, AAC, FLAC 編碼器
    ├── voice/                  # 音色管理
    │   ├── mod.rs
    │   └── manager.rs          # 音色預設與動態 Voice Object 解析
    └── engine/                 # ONNX 原生推論核心
        ├── mod.rs
        ├── types.rs            # ModelVersion (v1, v2, v2Pro, v2ProPlus, v3, v4)
        ├── sampler.rs          # 自回歸採樣器 (Top-K, Top-P, Temperature, Repetition Penalty)
        ├── cnhubert.rs         # CNHuBERT SSL 聲學特徵提取器
        ├── roberta.rs          # RoBERTa 語意特徵提取器
        ├── t2s.rs              # Text2Semantic 自回歸生成器
        ├── vits_v1_v2.rs       # V1 / V2 / V2Pro / V2ProPlus VITS 合成通道
        ├── cfm_v3_v4.rs        # V3 / V4 CFM DiT ODE 解算器與聲碼器
        └── model_manager.rs    # 多版本模型載入與調度
```

---

## 4. 支援模型版本對照

| 版本代號 | 符號集 | 支援語言 | 合成器架構 | 輸出採樣率 | 備註 |
|:---|:---|:---|:---|:---|:---|
| **v1** | `symbols_v1` (322) | 中、英、日 | Flow + HiFiGAN | 32,000 Hz | 初代模型 |
| **v2** | `symbols_v2` (732) | 中、英、日、粵、韓 | Flow + HiFiGAN (v2) | 32,000 Hz | 擴展語言與符號表 |
| **v2Pro** | `symbols_v2` (732) | 中、英、日、粵、韓 | Enhanced VITS + SV | 32,000 Hz | 支援 SV 聲紋嵌入 |
| **v2ProPlus** | `symbols_v2` (732) | 中、英、日、粵、韓 | Refined VITS + SV | 32,000 Hz | 強化音色分離度 |
| **v3** | `symbols_v2` (732) | 中、英、日、粵、韓 | CFM DiT + BigVGAN | 24,000 Hz | 32-step ODE 解算 |
| **v4** | `symbols_v2` (732) | 中、英、日、粵、韓 | CFM DiT + Vocoder | 48,000 Hz | 8~16 step 高速解算 |

---

## 5. 實作與驗證進度

- [x] 架構規劃與設計文檔
- [x] 專案依賴與設定檔範本 (`Cargo.toml`, `config.toml`, `voices.toml`)
- [x] 純 Rust 文字處理與 G2P 模組 (`src/text/`)
- [x] 音訊重採樣、變速與編碼器模組 (`src/audio/`)
- [x] 音色管理器 (`src/voice/`)
- [x] 全版本 ONNX 推論核心 (`src/engine/`)
- [x] OpenAI TTS API Axum 伺服器 (`src/server/`)
- [x] 離線 ONNX 匯出工具支援 uv (`tools/onnx_exporter.py`)
- [x] 全新環境官方預訓練底模下載器 (`tools/download_models.py`)
- [x] Dockerfile 與 docker-compose.yml 容器化配置
- [x] GitHub Actions CI/CD 與 GitHub Packages (ghcr.io) 自動發布流程
- [x] 建構與單元測試驗證 (`cargo check`, `cargo test`)
