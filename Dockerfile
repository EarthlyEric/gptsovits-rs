# ==============================================================================
# Stage 1: Build binary using official Rust toolchain
# ==============================================================================
FROM rust:1.80-bullseye AS builder

WORKDIR /usr/src/gptsovits-rs

# Install system dependencies required for native C/C++ build bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    cmake \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests
COPY Cargo.toml Cargo.lock ./

# Copy source tree and included assets
COPY src ./src
COPY tests ./tests
COPY assets ./assets

# Build release binary
RUN cargo build --release

# ==============================================================================
# Stage 2: Minimal runtime image
# ==============================================================================
FROM debian:bullseye-slim AS runtime

LABEL org.opencontainers.image.title="gptsovits-rs"
LABEL org.opencontainers.image.description="Pure Rust Inference Engine & OpenAI-compatible TTS Server for GPT-SoVITS"
LABEL org.opencontainers.image.authors="earthlyeric6"
LABEL org.opencontainers.image.licenses="MIT"

# Install runtime utilities: ffmpeg for audio transcoding, ca-certificates, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    curl \
    libssl1.1 \
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
