# ==============================================================================
# Stage 1: Build binary using Ubuntu 24.04 (Noble) with glibc 2.39+ and GCC 14
# ==============================================================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src/gptsovits-rs

# Install system dependencies required for native C/C++ build bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install official stable Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy dependency manifests
COPY Cargo.toml Cargo.lock ./

# Copy source tree and included assets
COPY src ./src
COPY tests ./tests
COPY assets ./assets

# Build release binary
RUN cargo build --release

# ==============================================================================
# Stage 2: CUDA 12.8 + cuDNN 9 runtime image
# ==============================================================================
# CUDA 12.8 provides native hardware support for RTX 50 series (Blackwell sm_120)
# as well as RTX 40 (sm_89), RTX 30 (sm_80/86), and RTX 20 (sm_75).
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 AS runtime

LABEL org.opencontainers.image.title="gptsovits-rs"
LABEL org.opencontainers.image.description="Pure Rust Inference Engine & OpenAI-compatible TTS Server for GPT-SoVITS"
LABEL org.opencontainers.image.authors="earthlyeric6"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime utilities: ffmpeg for audio transcoding, ca-certificates,
# curl for healthcheck, libssl3, and libgomp1.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    curl \
    libssl3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Make CUDA/cuDNN and custom ONNX Runtime provider libraries discoverable
ENV RUST_LOG="info,gptsovits_rs=info,tower_http=info,axum=info" \
    NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
    CUDA_HOME="/usr/local/cuda" \
    ORT_CUDA_LIB_DIR="/usr/local/cuda/lib64" \
    ORT_CUDNN_LIB_DIR="/usr/lib/x86_64-linux-gnu" \
    ORT_DYLIB_PATH="/usr/local/lib/libonnxruntime.so" \
    LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/bin"

# Create persistent mount directories
RUN mkdir -p /app/models /app/voices

# Copy binary from builder
COPY --from=builder /usr/src/gptsovits-rs/target/release/gptsovits-rs /usr/local/bin/gptsovits-rs

# Copy ONNX runtime dynamic libraries if produced during build
COPY --from=builder /usr/src/gptsovits-rs/target/release/libonnxruntime* /usr/local/lib/

# Copy custom ONNX runtime libraries if provided in build context (e.g. from GitHub Actions CI)
COPY ort_libs* /usr/local/lib/

# Copy default configurations
COPY config.toml /app/config.toml
COPY voices.toml /app/voices.toml

# Expose default TTS service port
EXPOSE 9880

# Configure container health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9880/health || exit 1

# Start the OpenAI TTS API Server
ENTRYPOINT ["/usr/local/bin/gptsovits-rs"]
CMD ["-c", "/app/config.toml"]
