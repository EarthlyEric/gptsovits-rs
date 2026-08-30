# GPT-SoVITS Rust 推論器與 OpenAI API 伺服器 - Agent 維護手冊

本檔案為 AI Agent 與開發維護者提供系統架構細節、慣例規範與操作說明。

---

## 1. 系統概觀

`gptsovits-rs` 是一個使用純 Rust 開發的高效能語音合成（TTS）伺服器，旨在實現以下核心功能：
1. **完全無 Python 執行期依賴（Zero Python Runtime Dependency）**：推論服務可作為獨立編譯的二進位執行檔運行，透過 `ort`（ONNX Runtime 2.x C API）直接調用 CPU / CUDA 硬體加速。
2. **GPT-SoVITS 全版本支援**：支援 `v1`, `v2`, `v2Pro`, `v2ProPlus`, `v3`, `v4` 六種模型架構。
3. **OpenAI TTS API 規範完全相容**：提供 `/audio/speech` 與 `/v1/audio/speech` 端點，完全符合 OpenAPI 3.1.0 規範，支援標準 Voice、自訂動態 Voice 物件、Bearer Token 認證、MP3/OPUS/AAC/FLAC/WAV/PCM 音訊串流與 SSE 串流。

---

## 2. 核心規範與設計約定

### 2.1 錯誤處理與 OpenAI API 對齊
- 所有 HTTP 回應的錯誤一律採用標準 `ErrorResponse` JSON 格式：
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
- HTTP 狀態碼對齊：
  - `400`: 參數缺失、無效的 speed (< 0.25 或 > 4.0)、無效的 response_format 或文本長度超過 4096 字符。
  - `401`: 缺少或無效的 `Authorization: Bearer <token>`。
  - `404`: 找不到指定的 `model` 或 `voice`。
  - `429`: 伺服器並發限制或速率超限。
  - `500`: 推論內部發生錯誤。

### 2.2 模型版本與取樣率
- **V1**: 符號集 `symbols_v1` (322 符號)，輸出 32,000 Hz。
- **V2 / V2Pro / V2ProPlus**: 符號集 `symbols_v2` (797 符號)，輸出 32,000 Hz。
- **V3**: 符號集 `symbols_v2` (797 符號)，輸出 24,000 Hz。
- **V4**: 符號集 `symbols_v2` (797 符號)，輸出 48,000 Hz。

### 2.3 語音與音色解析規則
- `voice` 欄位支援兩種格式：
  1. **字串格式**（如 `"default"`, `"sandrone"`）：從 `voices.toml` 中查找預先設定好的 `ref_audio_path`、`prompt_text`、`prompt_lang` 與 `model_version`。
  2. **自訂物件格式**（如 `{"ref_audio_path": "...", "prompt_text": "...", "prompt_lang": "zh", "text_lang": "zh"}`）：動態傳入參考音訊與提示詞進行零樣本複製（Zero-Shot Cloning）。

---

## 3. 開發與維護指令

### 3.1 編譯與代碼檢查
```bash
# 檢查語法與型別
cargo check

# 執行單元測試
cargo test

# 發布版本編譯
cargo build --release
```

### 3.2 官方預訓練底模下載（使用 uv）
在全新機器上一鍵自動下載官方權重：
```bash
# 下載全部底模
uv run tools/download_models.py

# 使用國內加速鏡像
uv run tools/download_models.py --source hf-mirror

# 僅下載指定版本並自動轉換 ONNX
uv run tools/download_models.py --version v2 --export-onnx
```

### 3.3 離線模型匯出（使用 uv 或 Python 從 PyTorch 權重轉為 ONNX）
使用 `uv` 可以在任何環境免手動安裝依賴直接執行匯出：
```bash
# 使用 uv 一鍵隔離執行
uv run tools/onnx_exporter.py \
    --gpt-path "GPT_weights/your_model.ckpt" \
    --sovits-path "SoVITS_weights/your_model.pth" \
    --version v2 \
    --output-dir "models"

# 或直接使用 Python 執行
python tools/onnx_exporter.py \
    --gpt-path "GPT_weights/your_model.ckpt" \
    --sovits-path "SoVITS_weights/your_model.pth" \
    --version v2 \
    --output-dir "models"
```

### 3.4 伺服器啟動
```bash
# 使用預設設定啟動
./target/release/gptsovits-rs

# 指定設定檔路徑與監聽位址
./target/release/gptsovits-rs -c config.toml -a 0.0.0.0 -p 9880
```

### 3.5 Docker 容器化部署
```bash
# 使用 Docker Compose 啟動
docker compose up -d

# 自行構建 Docker 映像檔
docker build -t gptsovits-rs:latest .
```

### 3.6 CI/CD 自動化與 GitHub Packages
- GitHub Actions 工作流位於 `.github/workflows/build.yml`。
- 在推動 Release Tag (`v*`) 或合併至 `main`/`master` 分支時，將自動執行測試、編譯多平台二進位檔並將 Docker 映像檔自動發布至 GitHub Container Registry (`ghcr.io/earthlyeric6/gptsovits-rs`)。
