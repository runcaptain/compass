# CPU-only Compass build. For the GPU variant see Dockerfile.gpu.
#
# Multi-stage build: compile in a full Rust environment, copy the binary
# to a minimal Debian runtime. Final image is ~120MB instead of ~2GB.

# ── Stage 1: Build ────────────────────────────────────────────────────────────
# Pin builder toolchain so deploys are reproducible and a compromised
# rust:latest tag can't silently land in our image.
FROM rust:1.88-bookworm AS builder

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
    curl \
    && rm -rf /var/lib/apt/lists/*
# Note: `curl` stays because the predeploy job at `scripts/download-models.sh`
# fetches BGE-small weights from HuggingFace via curl. Removing it breaks
# every cold deploy.

# Run the binary as an unprivileged user. The persistent disk mount path
# under $DATA_DIR is chown'd at deploy time; this image only
# needs to own /app and read its own scripts.
RUN groupadd --system --gid 10001 compass \
 && useradd  --system --uid 10001 --gid compass --no-create-home --shell /usr/sbin/nologin compass

WORKDIR /app

COPY --from=builder /app/target/release/compass /app/compass
COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh && mkdir -p /app/data && chown -R compass:compass /app

USER compass

ENV PORT=4001
# Default data directory. Override DATA_DIR to point at your persistent
# volume mount in production (e.g. a mounted disk or network volume).
ENV DATA_DIR=/app/data
EXPOSE 4001

CMD ["/app/compass"]
