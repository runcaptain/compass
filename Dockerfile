# Multi-stage build: compile in a full Rust environment, then copy the single binary
# to a minimal runtime image. Final image is ~100MB instead of ~2GB.

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM rust:1.82-bookworm AS builder

WORKDIR /app

# Install system dependencies needed by Tantivy and USearch
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first (Docker layer caching: dependencies only rebuild when Cargo.toml changes)
COPY Cargo.toml Cargo.lock* ./

# Create a dummy main.rs to pre-build dependencies
RUN mkdir -p src && echo "fn main() {}" > src/main.rs
RUN cargo build --release 2>/dev/null || true

# Now copy the real source code and build
COPY src/ src/
RUN cargo build --release

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /app/target/release/compass /app/compass

# Create data directory
RUN mkdir -p /app/data

# Default port
ENV PORT=4001
ENV DATA_DIR=/app/data

EXPOSE 4001

CMD ["/app/compass"]
