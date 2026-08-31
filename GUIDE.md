# GPT-SoVITS Rust 推論器與 OpenAI TTS API 伺服器使用教學

本手冊提供 `gptsovits-rs` 的完整安裝、模型匯出、配置與 OpenAI 相容 API 調用教學。

---

## 1. 系統特色

- **零 Python 依賴（Pure Rust Runtime）**：推論伺服器為原生編譯的獨立二進位執行檔，執行時無需 Python 解譯器或 PyTorch 環境。
- **全版本支援**：支援 GPT-SoVITS **v1, v2, v2Pro, v2ProPlus, v3, v4** 6 大模型版本。
- **OpenAI 100% 規格相容**：相容 OpenAPI 3.1.0 規範，可直接搭配 OpenAI 官方 Python / Node.js SDK 或第三方應用（如 NextChat, LibreChat, Dify 等）。
- **多音訊格式輸出**：支援 `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` 格式與即時串流（Binary Audio Stream 與 SSE）。
- **靈活音色調配**：支援預設音色名稱（如 `alloy`, `echo`, `sandrone`）與動態 Voice Object（零樣本語音複製）。

---

## 2. 快速開始

### 2.1 編譯伺服器

請確保本機已安裝 Rust 工具鏈（Rust 1.75+）：

```bash
# 1. 進入專案目錄
cd /home/earthlyeric6/Projects/gptsovits-rs

# 2. 編譯發布版本
cargo build --release

# 3. 產生的二進位執行檔位於 target/release/gptsovits-rs
```

### 2.2 啟動伺服器

```bash
# 使用預設設定 (config.toml, 埠口 9880)
# 註：config.toml 預設啟用 CUDA 硬體加速 (device = "cuda")
# 若本機環境僅有 CPU，可修改 config.toml 中 [runtime] device = "cpu"
./target/release/gptsovits-rs

# 或自訂設定檔與監聽位址
./target/release/gptsovits-rs -c config.toml -a 0.0.0.0 -p 9880
```

伺服器啟動後，將監聽：
- **API 端點**：`http://localhost:9880/v1/audio/speech` 與 `http://localhost:9880/audio/speech`
- **模型列表**：`http://localhost:9880/v1/models`
- **音色列表**：`http://localhost:9880/v1/voices`
- **健康檢查**：`http://localhost:9880/health`

---

## 3. 模型準備與 ONNX 匯出

### 3.0 全新環境一鍵下載預訓練底模（使用 uv）

在全新未下載官方底模的機器上，可使用 `tools/download_models.py` 一鍵自動從 HuggingFace 或 ModelScope 下載所需權重：

```bash
# 預設一鍵下載全部官方預訓練底模 (從 HuggingFace)
uv run tools/download_models.py

# 使用國內加速鏡像 (hf-mirror)
uv run tools/download_models.py --source hf-mirror

# 從 ModelScope 下載
uv run tools/download_models.py --source modelscope

# 僅下載基礎特徵模型與 V2 底模
uv run tools/download_models.py --version v2

# 下載完成後自動觸發 ONNX 轉換
uv run tools/download_models.py --version v2 --export-onnx
```

### 3.1 使用 `uv` 一鍵跨環境匯出 ONNX（推薦）

藉由 `uv` 的 PEP 723 依賴宣告，在任何機器上**無需手動安裝 PyTorch 套件**即可直接執行：

```bash
# 匯出 v2 模型範例
uv run tools/onnx_exporter.py \
    --gpt-path "GPT_weights/your_gpt_model.ckpt" \
    --sovits-path "SoVITS_weights/your_sovits_model.pth" \
    --version v2 \
    --output-dir "models"

# 匯出 v2ProPlus 模型範例
uv run tools/onnx_exporter.py \
    --gpt-path "GPT_weights_v2ProPlus/your_model.ckpt" \
    --sovits-path "SoVITS_weights_v2ProPlus/your_model.pth" \
    --version v2ProPlus \
    --output-dir "models"
```

匯出 `v2Pro` / `v2ProPlus` 時也會在共用目錄產生 `models/sv.onnx`；如有需要可使用 `--sv-path` 指定 ERes2NetV2 checkpoint。

已有舊版 V2Pro/V2ProPlus `vits.onnx` 時，可使用 `uv run tools/patch_vits_sv_input.py models/v2ProPlus/vits.onnx` 將零值 SV 常數替換為 `sv_emb` 輸入。

### 3.2 匯出參數說明

| 參數 | 說明 | 預設值 |
| :--- | :--- | :--- |
| `--gpt-path` | GPT / T2S 自回歸模型權重路徑 (`.ckpt`) | 選填 |
| `--sovits-path` | SoVITS / VITS 聲學合成模型權重路徑 (`.pth`) | 選填 |
| `--version` | 模型版本 (`v1`, `v2`, `v2Pro`, `v2ProPlus`, `v3`, `v4`) | `v2` |
| `--cnhubert-path` | CNHuBERT SSL 模型目錄 | `GPT_SoVITS/pretrained_models/chinese-hubert-base` |
| `--bert-path` | Chinese RoBERTa 模型目錄 | `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large` |
| `--output-dir` | 匯出 ONNX 檔案輸出目錄 | `models` |

---

## 4. 設定檔說明

### 4.1 伺服器設定 (`config.toml`)

```toml
[server]
bind_address = "0.0.0.0"
port = 9880
api_key = "" # 若設定字串，則啟用 Bearer Token 認證；為空則允許公開請求
max_concurrency = 8
voices_config = "voices.toml"

[runtime]
device = "cuda" # "cuda" 或 "cpu"
cuda_device_id = 0
# CUDA 13/cuDNN 路徑；Docker 部署使用 image 內建路徑，通常不需設定
# cuda_lib_dir = "/path/to/cuda13/lib"
# cudnn_lib_dir = "/path/to/cudnn/lib"
intra_threads = 4
inter_threads = 2

[models]
default_version = "v2" # 預設使用的模型版本
# 啟用的官方底模清單。留空時僅載入 default_version 以最大化節省 VRAM。
# enabled_base_versions = ["v2"]
cnhubert_path = "models/chinese-hubert-base/cnhubert.onnx"
bert_path = "models/chinese-roberta-wwm-ext-large/bert.onnx"
bert_tokenizer_path = "models/chinese-roberta-wwm-ext-large/tokenizer.json"
speaker_path = "models/sv.onnx"

# ==========================================
# 官方底模配置 (Base Models)
# ==========================================
[models.v2]
t2s_encoder_path = "models/v2/t2s_encoder.onnx"
t2s_fsdec_path = "models/v2/t2s_fsdec.onnx"
t2s_sdec_path = "models/v2/t2s_sdec.onnx"
vits_path = "models/v2/vits.onnx"
sampling_rate = 32000

# ==========================================
# 自訂角色微調模型配置 (Custom Fine-tuned Models)
# ==========================================
[models.custom.sandrone]
model_version = "v2ProPlus"
model_dir = "models/sandrone"   # 指向包含 t2s_encoder/fsdec/sdec 與 vits.onnx 的目錄
sampling_rate = 32000

[models.custom.hutao]
model_version = "v2"
model_dir = "models/hutao"
sampling_rate = 32000
```

#### 執行期參數表 (`[runtime]`)

| 欄位名稱 | 型別 | 預設值 | 說明 |
| :--- | :---: | :---: | :--- |
| `device` | string | `"cuda"` | 推論硬體裝置：`"cuda"`（NVIDIA GPU 加速）或 `"cpu"`。若選 `"cuda"` 且 GPU 不可用，伺服器將明確報錯終止，不會靜默退回 CPU。 |
| `cuda_device_id` | integer | `0` | 指定使用的 NVIDIA GPU 卡號（從 0 開始）。 |
| `cuda_lib_dir` | string | `""` | 指定 CUDA 13 核心動態庫（`libcudart.so.13`, `libcublas.so.13` 等）所在目錄。若已在系統動態庫搜尋路徑中可留空。 |
| `cudnn_lib_dir` | string | `""` | 指定 cuDNN 動態庫（`libcudnn.so.9` 等）所在目錄。若已在系統動態庫搜尋路徑中可留空。 |
| `intra_threads` | integer | `4` | 每個 ONNX 推論運算元內部之執行緒並行數（Intra-op threads）。 |
| `inter_threads` | integer | `2` | ONNX 計算圖運算元之間之並行調度執行緒數（Inter-op threads）。 |

#### 模型參數表 (`[models]`)

| 欄位名稱 | 型別 | 預設值 | 說明 |
| :--- | :---: | :---: | :--- |
| `default_version` | string | `"v2"` | 預設底模版本（`"v1"`, `"v2"`, `"v2Pro"`, `"v2ProPlus"`, `"v3"`, `"v4"`）。 |
| `enabled_base_versions` | array | `[]` | 啟用的官方底模清單（如 `["v2"]` 或 `["all"]`）。**留空時僅載入 `default_version`，大幅節省 VRAM**。 |
| `cnhubert_path` | string | `"models/chinese-hubert-base/cnhubert.onnx"` | CNHuBERT 語音特徵抽取模型路徑。 |
| `bert_path` | string | `"models/chinese-roberta-wwm-ext-large/bert.onnx"` | Chinese RoBERTa 文本語義特徵模型路徑。 |
| `speaker_path` | string | `"models/sv.onnx"` | ERes2NetV2 聲紋特徵抽取模型路徑（V2Pro/V2ProPlus 必要）。 |

### 4.2 音色預設 (`voices.toml`)

定義預設音色（`default`）與自訂角色微調音色（`sandrone` 等）對應之參考音訊、提示詞與文字切分策略：

```toml
# ==========================================
# 預設通用音色 (Default)
# ==========================================
[voices.default]
ref_audio_path = "voices/default/ref.wav"
prompt_text = "先帝创业未半而中道崩殂，今天下三分，益州疲弊。"
prompt_lang = "zh"
text_lang = "zh"
model_version = "v2"
text_split_method = "cut5"      # 👈 cut0(不切), cut1(湊4句), cut2(湊50字), cut3(按。切), cut4(按.切), cut5(按標點切)
fragment_interval = 0.2         # 句間自然停頓時間 (秒)
top_k = 15
top_p = 1.0
temperature = 1.0
repetition_penalty = 1.35

# ==========================================
# 自訂角色微調音色 (Sandrone)
# ==========================================
[voices.sandrone]
ref_audio_path = "voices/sandrone/ref.wav"
prompt_text = "我是「木偶」桑多涅。不必多言，做好你分内的事情即可。"
prompt_lang = "zh"
text_lang = "zh"
model_version = "v2ProPlus"
text_split_method = "cut5"      # 👈 對話推薦 (按所有標點符號切分，首字延遲最低)
fragment_interval = 0.2
top_k = 15
top_p = 1.0
temperature = 1.0
repetition_penalty = 1.35
```

#### 切分策略詳細說明 (`text_split_method`)
- **`cut5` (預設推薦)**：按所有標點符號切分（`，` `。` `？` `！` `、` `…` 等），首字延遲最低，適合聊天對話與即時串流。
- **`cut1`**：湊四句一切（以標點切分後每 4 句一組合成），兼顧上下文連貫性。
- **`cut2`**：湊 50 字一切（累計到約 50 字才切分，最後短句自動向前合併），故事朗讀、有聲書推薦。
- **`cut3`**：僅按中文句號 `。` 切分。
- **`cut4`**：僅按英文句號 `.` 切分（自動保護如 `3.14` 小數點不被截斷）。
- **`cut0`**：不切分（整段文字一次性合成，僅適合短句 < 25 字）。

---

## 5. API 調用範例

### 5.1 使用 cURL

#### 調用自訂微調模型 (以 Sandrone 為例)
在 `model` 欄位直接指定您在 `config.toml` 註冊的微調模型名稱：
```bash
curl http://localhost:9880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sandrone",
    "voice": "sandrone",
    "input": "這是使用我親自訓練的 Sandrone 專屬微調模型進行的高品質語音合成。",
    "response_format": "mp3"
  }' \
  --output sandrone.mp3
```

#### 基本合成 (調用官方通用底模)
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
  --output output.mp3
```

#### 變更語速與輸出格式 (WAV 格式, 1.25 倍速)
```bash
curl http://localhost:9880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-sovits-v2",
    "input": "測試變速合成功能，語速加快一點二五倍。",
    "voice": "sandrone",
    "response_format": "wav",
    "speed": 1.25
  }' \
  --output output.wav
```

#### 使用動態 Voice Object（零樣本聲音複製）
若客戶端不想依賴伺服器預設音色，可直接在 `voice` 傳入自訂參數物件：
```bash
curl http://localhost:9880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-sovits-v2",
    "input": "這是一個使用即時傳入參考音訊進行複製的語音。",
    "voice": {
      "ref_audio_path": "/path/to/custom_ref.wav",
      "prompt_text": "我是自訂參考音訊的提示文本",
      "prompt_lang": "zh",
      "text_lang": "zh",
      "model_version": "v2",
      "top_k": 15,
      "temperature": 1.0
    },
    "response_format": "wav"
  }' \
  --output custom_voice.wav
```

---

### 5.2 使用 Python OpenAI 官方 SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9880/v1",
    api_key="your-api-key" # 若未設定可隨意填寫
)

response = client.audio.speech.create(
    model="gpt-sovits-v2",
    voice="default",
    input="歡迎使用純 Rust 高效能 GPT-SoVITS 推論引擎！",
    response_format="mp3",
    speed=1.0
)

# 儲存音訊檔案
response.stream_to_file("speech.mp3")
print("音訊生成完成：speech.mp3")
```

---

### 5.3 使用 Node.js / TypeScript OpenAI SDK

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
  await fs.promises.writeFile("speech.mp3", buffer);
  console.log("音訊生成完成：speech.mp3");
}

main();
```

---

## 6. API 規格與參數參照表

### `POST /v1/audio/speech` 與 `POST /audio/speech`

| 欄位名稱 | 型別 | 必填 | 說明 | 範例 / 預設值 |
| :--- | :--- | :---: | :--- | :--- |
| `model` | string | 是 | 模型識別碼（如 `gpt-sovits-v1` ~ `v4`, `gpt-4o-mini-tts`, `tts-1`） | `"gpt-sovits-v2"` |
| `input` | string | 是 | 欲合成之文字（長度 1 ~ 4096 字符） | `"你好，世界！"` |
| `voice` | string \| object | 是 | 音色名稱（字串）或動態音色設定物件 | `"default"` |
| `instructions`| string | 否 | 風格、情感或發音指示 | `"自然清晰"` |
| `response_format` | string | 否 | 音訊格式：`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` | 預設 `"mp3"` |
| `speed` | number | 否 | 語速縮放比例（範圍 `0.25` ~ `4.0`） | 預設 `1.0` |
| `stream_format`| string | 否 | 串流模式：`audio`（二進位串流）或 `sse`（事件串流） | 預設 `"audio"` |

---

## 7. 互動式 API 文件與 OpenAPI 規格 (Interactive API Docs)

伺服器內建現代化互動式 API 測試與文件頁面，啟動後可直接於瀏覽器開啟：

* **Scalar 互動文件介面**：`http://localhost:9880/docs`
* **Swagger UI 介面**：`http://localhost:9880/swagger-ui`
* **OpenAPI 3.1.0 JSON 規格**：`http://localhost:9880/openapi.json`（亦存於專案根目錄 `openapi.json`，支援直接匯入 Postman 或 OpenAPI Generator）

---

## 8. 錯誤回應格式與 HTTP 狀態碼

所有錯誤皆回傳標準 OpenAI 格式：

```json
{
  "error": {
    "message": "描述性錯誤訊息",
    "type": "invalid_request_error",
    "param": "voice",
    "code": "invalid_voice"
  }
}
```

| HTTP 代碼 | 說明 |
| :---: | :--- |
| `400` | 請求參數無效（如 input 為空、超出 4096 字符、speed 不在 0.25~4.0 範圍）。 |
| `401` | Bearer Token 缺失或驗證失敗（`invalid_api_key`）。 |
| `404` | 找不到指定的模型或音色名稱。 |
| `429` | 伺服器並發超過上限（`rate_limit_exceeded`）。 |
| `500` | 推論核心處理發生錯誤。 |

---

## 9. Docker 與 Docker Compose 容器化部署

### 9.1 使用 Docker Compose 一鍵啟動 (推薦 GPU 部署)

專案預設之 Docker 映像檔基於 `nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04` 構建，內建完整的 CUDA 13 與 cuDNN 運行時，無需在本機安裝額外的 CUDA 動態庫。

```bash
# 前置條件：確保宿主機已安裝 NVIDIA Driver 與 nvidia-container-toolkit
# 驗證 Docker 可存取 GPU
docker run --rm --gpus all nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 nvidia-smi

# 1. 建立掛載目錄並準備模型
mkdir -p models voices

# 2. 建置並背景啟動服務 (自動啟用 GPU 加速)
docker compose up -d --build

# 3. 檢查容器執行狀態與日誌
docker compose logs -f
```

### 9.2 從 GitHub Container Registry (ghcr.io) 拉取映像檔

```bash
# 拉取最新預建置映像檔
docker pull ghcr.io/earthlyeric6/gptsovits-rs:latest

# 直接運行容器 (傳遞 GPU 裝置)
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

### 9.3 自行建置 Docker 映像檔

```bash
docker build -t gptsovits-rs:latest .
```

---

## 10. 常見問題與排錯指南 (Troubleshooting)

### Q1: 啟動時提示 `libcudart.so` / `libcublasLt.so` cannot open shared object file 怎麼辦？
* **原因**：本機 ONNX Runtime CUDA 發行包鏈接至 CUDA 核心動態庫，而宿主機環境缺少對應的 CUDA 函式庫或未加入搜尋路徑。
* **處置方案**：
  1. **（推薦）使用 Docker Compose 部署**：執行 `docker compose up -d --build`，容器內建完整的 CUDA 12.8 / 13 + cuDNN 環境，開箱即用。
  2. **手動指定路徑**：可在 `config.toml` 中配置 `[runtime] cuda_lib_dir = "/opt/cuda/targets/x86_64-linux/lib"` 或指定 `ORT_CUDA_LIB_DIR`。
  3. **切換為純 CPU 模式**：在 `config.toml` 中將 `[runtime] device = "cpu"` 即可於純 CPU 模式下順暢運行。

### Q2: 推論時出現 `CUDA error cudaErrorNoKernelImageForDevice: no kernel image is available`
* **原因**：微軟官方預編譯的 ONNX Runtime 發行包內建之 CUDA kernel 僅支援至 `sm_75` (Turing/RTX 20)、`sm_80/86` (Ampere/RTX 30)、`sm_89/90` (Ada Lovelace/RTX 40/Hopper)。在最新的 RTX 5000 系列（Blackwell 架構，`sm_120`）若無對應 kernel，CUDA 會報無可用 Kernel。
* **處置方案**：
  1. **透過 GitHub Actions / Docker 自動構建**：本專案 CI 已整合包含 `sm_120` 支援的 CUDA 12.8 ONNX Runtime 自動構建與 Docker 映像檔發布。
  2. **暫時切換為 CPU 模式**：在 `config.toml` 設定 `[runtime] device = "cpu"` 並調大 `intra_threads` 運行。

### Q2: 如何確認模型是否有真正使用 GPU 進行推論？
啟動伺服器後發送合成請求，並在終端機執行：
```bash
watch -n 1 nvidia-smi
```
在推論合成期間，應可觀察到 `gptsovits-rs` 程序佔用 GPU 顯存（約 1.5GB ~ 3GB，視模型而定），且 GPU-Util 數值有所跳動。若日誌顯示 `ONNX Runtime CUDAExecutionProvider registered` 則代表 GPU 加速已成功啟用。

### Q3: V2Pro / V2ProPlus 模型合成報錯 `V2Pro/V2ProPlus VITS requires a speaker embedding`
* **原因**：V2Pro / V2ProPlus 架構需要 `sv.onnx` 聲紋特徵抽取器，且 `vits.onnx` 圖需暴露 `sv_emb`（形狀 `[1, 20480]`）輸入。
* **處置方案**：請確認 `models/sv.onnx` 存在，並使用 `uv run tools/patch_vits_sv_input.py models/your_model/vits.onnx` 修補該模型。

### Q4: 合成請求延遲較高，可能的原因是什麼？
1. **未開啟 GPU 加速**：請確認 `config.toml` 中 `device = "cuda"` 且容器已正確掛載 GPU。
2. **文本切分過多**：預設的 `cut5` 切分策略會為每個標點符號建立獨立的自回歸解碼段。如為大篇幅朗讀，可選用 `cut2`（湊 50 字切分）減少解碼次數。
3. **音訊編碼格式**：預設輸出 `mp3` 會呼叫系統 `ffmpeg` 轉碼；若追求極致低延遲，可傳入 `"response_format": "wav"` 或 `"pcm"` 走純 Rust 直出編碼。
