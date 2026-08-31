# gptsovits-rs

<p align="center">
  <strong>高效能純 Rust 原生 GPT-SoVITS 基於ONNX推論引擎與 100% OpenAI TTS API 兼容伺服器</strong>
</p>

<p align="center">
  <a href="https://github.com/earthlyeric6/gptsovits-rs/actions/workflows/build.yml"><img src="https://img.shields.io/badge/CI%2FCD-Passing-brightgreen?style=flat-square&logo=github-actions" alt="CI/CD"></a>
  <a href="https://github.com/earthlyeric6/gptsovits-rs/pkgs/container/gptsovits-rs"><img src="https://img.shields.io/badge/GHCR-Docker%20Image-blue?style=flat-square&logo=docker" alt="Docker"></a>
  <img src="https://img.shields.io/badge/Rust-1.75%2B-orange?style=flat-square&logo=rust" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/OpenAI-TTS%20API%20Compatible-412991?style=flat-square&logo=openai" alt="OpenAI Compatible">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

---

## 📖 專案簡介 (Overview)

`gptsovits-rs` 是一個使用 **純 Rust (Pure Rust)** 開發的高效能語音合成（TTS）伺服器，專為部署 GPT-SoVITS 模型而生。

- ⚡ **零 Python 依賴（Zero Python Runtime Dependency）**：伺服器為原生編譯的獨立二進位檔，執行階段**完全不依賴 Python 解譯器、PyTorch 或 CUDA Python 套件**。
- 🎙️ **全版本模型支援**：完整支援 GPT-SoVITS **v1, v2, v2Pro, v2ProPlus, v3, v4** 全部 6 大模型版本。
- 🤖 **100% 兼容 OpenAI TTS API**：完全遵循 **OpenAPI 3.1.0** 規範，相容官方 OpenAI Python/Node SDK 及所有支援 OpenAI TTS 的前端應用（如 NextChat, LibreChat, Dify 等）。
- 🚀 **硬體加速推論**：透過 `ort`（ONNX Runtime 2.x C API）直接呼叫 CPU 與 NVIDIA CUDA / TensorRT 硬體加速。
- 🎵 **豐富音訊格式輸出**：支援 `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` 格式與即時串流（Binary Audio 與 SSE）。
- 🧬 **零樣本語音複製（Zero-Shot Cloning）**：支援自訂音色名稱（如 `alloy`, `echo`）與動態傳入參考音訊的自訂 Voice Object。

---

## 🏛️ 系統架構 (Architecture)

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

## 📊 支援模型版本對照表

| 版本代號 | 符號集 | 支援語言 | 聲學合成架構 | 預設採樣率 | 特色說明 |
|:---|:---|:---|:---|:---|:---|
| **v1** | `symbols_v1` (322) | 中、英、日 | Flow + HiFiGAN | 32,000 Hz | 經典初代模型 |
| **v2** | `symbols_v2` (732) | 中、英、日、粵、韓 | Flow + HiFiGAN (v2) | 32,000 Hz | 擴展韓語/粵語，發音更自然 |
| **v2Pro** | `symbols_v2` (732) | 中、英、日、粵、韓 | Enhanced VITS + SV | 32,000 Hz | 整合 SV 聲紋特徵，提升音色相似度 |
| **v2ProPlus** | `symbols_v2` (732) | 中、英、日、粵、韓 | Refined VITS + SV | 32,000 Hz | 強化多說話人音色解耦能力 |
| **v3** | `symbols_v2` (732) | 中、英、日、粵、韓 | CFM DiT + BigVGAN | 24,000 Hz | 32-step ODE 解算，BigVGAN 高音質 |
| **v4** | `symbols_v2` (732) | 中、英、日、粵、韓 | CFM DiT + Vocoder | 48,000 Hz | 8~16 step 高速解算，48kHz 超高取樣率 |

---

## 🚀 快速開始 (Quick Start)

### 1. 編譯與執行

```bash
# 1. 複製專案
git clone https://github.com/earthlyeric6/gptsovits-rs.git
cd gptsovits-rs

# 2. 編譯生產環境 Release 二進位檔
cargo build --release

# 3. 啟動伺服器 (預設監聽 0.0.0.0:9880)
# 註：config.toml 預設 [runtime] device = "cuda" (需 CUDA 13+ 與 cuBLAS 動態庫)
# 若本機僅有 CPU 或舊版 CUDA，可將 config.toml 設為 device = "cpu" 或透過 Docker GPU 部署
./target/release/gptsovits-rs -c config.toml
```

---

### 2. 準備預訓練底模與 ONNX 權重

#### 步驟 A：全新環境一鍵下載官方底模 (使用 `uv`)
在全新機器上，可透過內建工具自動從 HuggingFace / ModelScope 下載模型權重：
```bash
# 一鍵下載全部預訓練底模 (從 HuggingFace)
uv run tools/download_models.py

# 使用國內加速鏡像
uv run tools/download_models.py --source hf-mirror

# 僅下載 V2 模型並自動轉換為 ONNX 格式
uv run tools/download_models.py --version v2 --export-onnx
```

#### 步驟 B：將既有 PyTorch 權重匯出為 ONNX
```bash
# 匯出 v2 模型
uv run tools/onnx_exporter.py \
    --gpt-path "GPT_weights/your_gpt_model.ckpt" \
    --sovits-path "SoVITS_weights/your_sovits_model.pth" \
    --version v2 \
    --output-dir "models"
```

---

## 🐳 Docker 與 Docker Compose 部署

### 使用 Docker Compose 一鍵啟動

```bash
# 前置條件：NVIDIA Driver、Docker Compose 與 nvidia-container-toolkit
# 確認 Docker 可以存取 GPU
docker run --rm --gpus all nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 nvidia-smi

# 建立目錄並啟動 CUDA 13 + cuDNN 服務
mkdir -p models voices
docker compose up -d --build

# 查看即時日誌
docker compose logs -f
```

### 從 GitHub Container Registry 拉取映像檔

```bash
docker pull ghcr.io/earthlyeric6/gptsovits-rs:latest

docker run -d \
  --name gptsovits-rs \
  -p 9880:9880 \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v $(pwd)/config.toml:/app/config.toml:ro \
  -v $(pwd)/voices.toml:/app/voices.toml:ro \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/voices:/app/voices \
  --restart unless-stopped \
  ghcr.io/earthlyeric6/gptsovits-rs:latest
```

---

## 💻 API 調用範例 (OpenAI TTS Compatible)

### 1. cURL 範例

#### 調用您訓練的專屬微調模型 (以 Sandrone 為例)
```bash
curl http://localhost:9880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sandrone",
    "voice": "sandrone",
    "input": "這是使用我訓練的 Sandrone 專屬微調權重生成的語音。",
    "response_format": "mp3"
  }' \
  --output sandrone.mp3
```

#### 基本文字轉語音 (使用官方通用底模)
```bash
curl http://localhost:9880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gpt-sovits-v2",
    "input": "先帝創業未半而中道崩殂，今天下三分，益州疲弊。",
    "voice": "default",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3
```

#### 動態 Voice 物件（零樣本語音複製 Zero-Shot）
```bash
curl http://localhost:9880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-sovits-v2",
    "input": "這是使用自訂參考音訊進行複製的語音。",
    "voice": {
      "ref_audio_path": "/path/to/reference.wav",
      "prompt_text": "我是參考音訊中的提示語音內容",
      "prompt_lang": "zh",
      "text_lang": "zh",
      "model_version": "v2",
      "top_k": 15,
      "temperature": 1.0
    },
    "response_format": "wav"
  }' \
  --output cloned_voice.wav
```

---

### 2. Python 官方 SDK 範例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9880/v1",
    api_key="your-api-key"
)

response = client.audio.speech.create(
    model="gpt-sovits-v2",
    voice="default",
    input="歡迎使用純 Rust 高效能 GPT-SoVITS 推論引擎！",
    response_format="mp3",
    speed=1.0
)

response.stream_to_file("output.mp3")
print("語音生成完成：output.mp3")
```

---

### 3. Node.js / TypeScript SDK 範例

```typescript
import OpenAI from "openai";
import fs from "fs";

const openai = new OpenAI({
  baseURL: "http://localhost:9880/v1",
  apiKey: "your-api-key",
});

async function main() {
  const mp3 = await openai.audio.speech.create({
    model: "gpt-sovits-v2",
    voice: "default",
    input: "你好！這是一段透過 Node.js 官方 SDK 生成的語音。",
    response_format: "mp3",
  });

  const buffer = Buffer.from(await mp3.arrayBuffer());
  await fs.promises.writeFile("output.mp3", buffer);
  console.log("語音生成完成：output.mp3");
}

main();
```

---

## 🛠️ API 端點參照 (Endpoints)

| 方法 | 路徑 | 說明 |
| :---: | :--- | :--- |
| `POST` | `/v1/audio/speech` | 生成語音（符合 OpenAI TTS 規範） |
| `POST` | `/audio/speech` | 生成語音別名路徑 |
| `GET` | `/v1/models` | 取得支援之底模與自訂微調模型列表 |
| `GET` | `/v1/voices` | 取得可用預設音色列表 |
| `GET` | `/health` | 服務健康檢查端點 |
| `GET` | `/docs` | 現代化 Scalar 互動式 API 文件與線上測試介面 |
| `GET` | `/swagger-ui` | Swagger UI 互動式 API 文件介面 |
| `GET` | `/openapi.json` | OpenAPI 3.1.0 規格檔案（支援匯入 Postman） |

---

## 📚 相關文檔 (Documentation)

- 📖 **[GUIDE.md](GUIDE.md)**：完整詳細的使用指南與進階參數教學。
- 📋 **[PLAN.md](PLAN.md)**：系統架構設計與實作規劃細節。
- 🤖 **[AGENT.md](AGENT.md)**：AI Agent 維護手冊與系統規範。

---

## 📄 開源授權 (License)

本專案採用 [MIT License](LICENSE) 授權開源。
