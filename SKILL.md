---
name: gptsovits-setup
description: Use when setting up gptsovits-rs from scratch, installing Rust/uv/Docker prerequisites, downloading GPT-SoVITS checkpoints, exporting ONNX models, configuring config.toml and voices.toml, verifying sv.onnx speaker embeddings, or deploying with Docker Compose and testing /health or the TTS API.
---

# gptsovits-rs Environment Setup Guide

This guide walks you through setting up `gptsovits-rs` (a pure Rust GPT-SoVITS inference engine with an OpenAI-compatible TTS API server) on a fresh machine: prerequisites, model preparation, configuration, and deployment.

## 1. Project Overview

- **Runtime**: single native binary; no Python / PyTorch required at runtime.
- **Supported versions**: v1, v2, v2Pro, v2ProPlus, v3, v4.
- **v2Pro / v2ProPlus requirement**: an ERes2NetV2 speaker model produces `sv_emb` (`[1, 20480]`) fed into VITS.

## 2. Prerequisites

```bash
# Rust 1.75+ (includes cargo)
rustc --version

# uv (PEP 723 script runner, used for model download/export)
uv --version

# Docker / Docker Compose & NVIDIA Container Toolkit (for GPU container deployment)
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 nvidia-smi

# Audio encoding dependency (ffmpeg; bundled in the container image)
ffmpeg -version
```

Install any missing tool before continuing.

## 3. Get the code and build

```bash
git clone https://github.com/earthlyeric6/gptsovits-rs.git
cd gptsovits-rs

# Type / syntax check
cargo check

# Unit and API tests
cargo test

# Release build
cargo build --release
```

The binary is produced at `target/release/gptsovits-rs`.

## 4. Model download and ONNX export

### 4.1 One-click download of official base models (recommended)

```bash
# Download all official base models (HuggingFace)
uv run tools/download_models.py

# Use a mirror (for mainland China networks)
uv run tools/download_models.py --source hf-mirror
uv run tools/download_models.py --source modelscope

# Download a specific version and auto-export ONNX
uv run tools/download_models.py --version v2 --export-onnx
uv run tools/download_models.py --version sandrone --export-onnx
```

Supported `--version` values: `all`, `base`, `v1`, `v2`, `v2pro`, `v2proplus`, `v3`, `v4`, `sandrone`.

### 4.2 Export ONNX from existing PyTorch checkpoints

`tools/onnx_exporter.py` automatically locates the GPT-SoVITS upstream source (in order: the `GPT_SOVITS_UPSTREAM` environment variable, `./GPT-SoVITS`, `./GPT-SoVITS-src`, a sibling `gptsovits_upstream`, `/tmp/gpt-sovits-upstream`).

```bash
# v2 example
uv run tools/onnx_exporter.py \
    --gpt-path "GPT_weights/your_gpt_model.ckpt" \
    --sovits-path "SoVITS_weights/your_sovits_model.pth" \
    --version v2 \
    --output-dir "models"

# v2ProPlus example
uv run tools/onnx_exporter.py \
    --gpt-path "GPT_weights_v2ProPlus/your_model.ckpt" \
    --sovits-path "SoVITS_weights_v2ProPlus/your_model.pth" \
    --version v2ProPlus \
    --output-dir "models"
```

Exporting `v2Pro` / `v2ProPlus` also produces a shared `models/sv.onnx` (ERes2NetV2 speaker model) under `--output-dir`; use `--sv-path` to point at a specific checkpoint.

### 4.3 Patch an old VITS graph

If you already have an older V2Pro/V2ProPlus `vits.onnx` (with a hardcoded zero-valued SV constant), retrofit it to expose an `sv_emb` input:

```bash
uv run tools/patch_vits_sv_input.py models/v2ProPlus/vits.onnx
```

## 5. Verify the models

After export, confirm the required files and tensor shapes:

```bash
ls -la models/
ls -la models/sv.onnx models/v2ProPlus/vits.onnx models/sandrone/vits.onnx 2>/dev/null
```

**The V2Pro / V2ProPlus VITS graph MUST have an `sv_emb` input of shape `[1, 20480]`**:

```bash
uv run --with onnx python - <<'PY'
import onnx
m = onnx.load("models/v2ProPlus/vits.onnx", load_external_data=False)
for i in m.graph.input:
    print(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
PY
```

You should see `sv_emb ['1', '20480']`. `models/sv.onnx` takes `fbank [batch, frames, 80]` and outputs `sv_emb [1, 20480]`.

## 6. Configuration

### 6.1 `config.toml`

Confirm the `[models]` section (including `speaker_path`) and per-version model paths:

```toml
[models]
default_version = "v2"
cnhubert_path = "models/chinese-hubert-base/cnhubert.onnx"
bert_path = "models/chinese-roberta-wwm-ext-large/bert.onnx"
bert_tokenizer_path = "models/chinese-roberta-wwm-ext-large/tokenizer.json"
speaker_path = "models/sv.onnx"
```

Custom characters (e.g. sandrone) live under `[models.custom.<name>]` with `model_version = "v2ProPlus"` and `model_dir = "models/sandrone"`.

### 6.2 `voices.toml`

Each voice (`default`, `sandrone`, ...) needs `ref_audio_path`, `prompt_text`, `prompt_lang`, `text_lang`, and `model_version`.

Reference audio must be 3–10 seconds; the server returns 400 `missing_reference_audio` for empty reference audio.

## 7. Start the server

```bash
# Run directly
./target/release/gptsovits-rs -c config.toml

# Or via Docker Compose (CUDA 13 + cuDNN image; requires NVIDIA Container Toolkit)
mkdir -p models voices
docker compose up -d --build
docker compose logs -f
```

The server listens on `0.0.0.0:9880` by default.

## 8. Verify the deployment

```bash
# Health check
curl -fsS http://localhost:9880/health

# TTS synthesis (base model)
curl http://localhost:9880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-sovits-v2",
    "voice": "default",
    "input": "Hello, this is a speech synthesis test.",
    "response_format": "mp3"
  }' \
  --output speech.mp3

# Sandrone custom model
curl http://localhost:9880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sandrone",
    "voice": "sandrone",
    "input": "This speech was generated with the Sandrone model.",
    "response_format": "mp3"
  }' \
  --output sandrone.mp3
```

Other endpoints: `/v1/models`, `/v1/voices`, `/docs`, `/swagger-ui`, `/openapi.json`.

## 9. Git exclusion rules

Large models and audio files must not be committed (`.gitignore` already covers `/models/`, `/voices/*`, `*.onnx`, `*.wav`, `*.mp3`, `*.ckpt`, `*.pth`, `*.bin`, `GPT-SoVITS`, etc.). Inspect what would be committed:

```bash
git status --short
```

Do not run `git commit` / `git push` unless the user explicitly asks.

## 10. Troubleshooting

- **`libcublasLt.so / libcudart.so: cannot open shared object file` (local run)**: the compiled `ort` 2.x CUDA package links against CUDA 12/13. Use `docker compose up -d --build` (which embeds CUDA 12.8 + cuDNN), set `[runtime] cuda_lib_dir` to point at your CUDA libraries, or set `device = "cpu"` in `config.toml`.
- **`cudaErrorNoKernelImageForDevice` (RTX 5000 / Blackwell)**: prebuilt upstream ONNX Runtime CUDA packages support `sm_75` through `sm_90` (RTX 20/30/40, A100/H100). For RTX 5000 series, build with CUDA 12.8 + `sm_120` via GitHub Actions CI / Docker, or set `[runtime] device = "cpu"` in `config.toml`.
- **`sv_emb` missing / wrong shape**: the VITS graph has no speaker input; re-export or run `tools/patch_vits_sv_input.py`.
- **Model load warnings**: `T2S/VITS/CFM/CNHuBERT/RoBERTa ONNX model file not found` means the file at that path is missing; check `config.toml` paths and the model directory.
- **`/health` fails**: confirm the container is running and `docker compose ps` shows healthy.
- **Ignored integration tests**: tests in `tests/api_integration_tests.rs` that need external ONNX assets are `#[ignore]`d by default; run them with `cargo test -- --ignored` once the models are ready.
