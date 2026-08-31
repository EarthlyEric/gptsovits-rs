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
# Stage 2: CUDA 13 + cuDNN runtime image
# ==============================================================================
# ort's CUDA-enabled prebuilt distribution currently targets CUDA 13. Keep the
# runtime image on the same major CUDA version as the provider binary.
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 AS runtime

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

# Make CUDA/cuDNN and the ONNX Runtime provider libraries discoverable by both
# the dynamic linker and the explicit preload performed by the Rust binary.
ENV CUDA_HOME=/usr/local/cuda \
    ORT_CUDA_LIB_DIR=/usr/local/cuda/lib64 \
    ORT_CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib:/usr/local/bin

# Create persistent mount directories
RUN mkdir -p /app/models /app/voices

# Copy binary from builder
COPY --from=builder /usr/src/gptsovits-rs/target/release/gptsovits-rs /usr/local/bin/gptsovits-rs

# copy-dylibs emits these provider libraries next to the release binary. They
# are not part of the statically linked core ONNX Runtime library.
COPY --from=builder /usr/src/gptsovits-rs/target/release/libonnxruntime_providers_*.so /usr/local/bin/

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
