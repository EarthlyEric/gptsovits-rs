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
# Stage 2: Minimal runtime image based on Ubuntu 24.04
# ==============================================================================
FROM ubuntu:24.04 AS runtime

LABEL org.opencontainers.image.title="gptsovits-rs"
LABEL org.opencontainers.image.description="Pure Rust Inference Engine & OpenAI-compatible TTS Server for GPT-SoVITS"
LABEL org.opencontainers.image.authors="earthlyeric6"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime utilities: ffmpeg for audio transcoding, ca-certificates, curl for healthcheck, libssl3, libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    curl \
    libssl3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create persistent mount directories
RUN mkdir -p /app/models /app/voices

# Copy binary from builder
COPY --from=builder /usr/src/gptsovits-rs/target/release/gptsovits-rs /usr/local/bin/gptsovits-rs

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
