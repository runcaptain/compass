# CPU-only Compass build. For the GPU variant see Dockerfile.gpu.
#
# Multi-stage build: compile in a full Rust environment, copy the binary
# to a minimal Debian runtime. Final image is ~120MB instead of ~2GB.

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM rust:latest AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifests first so dependency resolution is cached.
COPY Cargo.toml Cargo.lock* ./
COPY crates/compass/Cargo.toml crates/compass/
COPY crates/compass-index-api/Cargo.toml crates/compass-index-api/
COPY crates/compass-vector-gpu/Cargo.toml crates/compass-vector-gpu/

# Stub source so dependency build is cached. The GPU crate is in workspace
# members; without --exclude its deps would also resolve. We compile only the
# default member chain.
RUN mkdir -p crates/compass/src crates/compass-index-api/src \
    && echo "fn main() {}" > crates/compass/src/main.rs \
    && echo "" > crates/compass-index-api/src/lib.rs \
    && cargo build --release -p compass --no-default-features 2>/dev/null || true

# Now bring in the real source and rebuild everything that changed.
COPY crates/ crates/
RUN touch crates/compass-index-api/src/lib.rs crates/compass/src/main.rs \
    && cargo build --release -p compass

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/compass /app/compass
RUN mkdir -p /app/data

ENV PORT=4001
ENV DATA_DIR=/app/data
EXPOSE 4001

CMD ["/app/compass"]
