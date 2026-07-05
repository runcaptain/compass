# Contributing to Compass

Thanks for your interest. Compass is the embedded search engine behind [Captain](https://runcaptain.com); we develop it primarily for our own product but we're happy to take outside contributions that align with the roadmap.

## Quick start

```bash
git clone https://github.com/runcaptain/compass
cd compass
cargo build --release           # CPU-only, ~2 minutes on a fresh checkout
cargo run --release             # serves on http://localhost:4001
```

## Workspace layout

Compass is a Cargo workspace. Useful invocations:

Prerequisites on Linux: `cmake`, `pkg-config`, `libssl-dev` (what CI
installs). On Windows there is a known linker clash between `esaxx-rs` and
`cxx` — use `cargo check` locally and run builds/tests in Docker or WSL.

```bash
cargo build                                                # default member (`compass`)
cargo test --workspace --exclude compass-vector-gpu       # all tests CI runs
cargo test -p compass --features object-storage           # + S3/GCS backend tests
cargo clippy --workspace --exclude compass-vector-gpu --all-targets -- -D warnings
cargo fmt --all --check                                    # CI requires a clean diff
```

`compass-vector-gpu` is a standalone experimental crate (cuVS; needs CUDA
12+, CMake, a long first build) that is NOT wired into the engine yet —
every CI job excludes it, and so should you unless you're working on it.
Tests against real object storage skip cleanly unless `COMPASS_TEST_S3_BUCKET`
is set (CI runs them against MinIO).

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module map and where to put new code.

## What we accept

**In scope:**
- Bug fixes with a regression test.
- Performance improvements with a before/after benchmark using `cargo bench`.
- New vector backends that implement `compass_index_api::VectorIndex`.
- Documentation improvements, especially examples.

**Out of scope (for now):**
- New embedding model integrations (we plug into HuggingFace TEI / vLLM via the `embed_endpoint` config; pull requests adding new in-process embedders need a strong motivation).
- Consensus/quorum replication. Compass scales out via object storage as the source of truth (stateless writers + serving nodes + serve-from-storage cold reads); PRs should build on that model, not introduce node-to-node coordination.

If you're not sure, open an issue first and ask.

## Pull request checklist

- [ ] `cargo fmt --all` clean.
- [ ] `cargo clippy --workspace -- -D warnings` clean.
- [ ] `cargo test --workspace --exclude compass-vector-gpu` green.
- [ ] `CHANGELOG.md` updated under the `[Unreleased]` section.
- [ ] Public API changes have rustdoc comments.
- [ ] Behavior changes have a test that would have caught the regression.

## Commit format

We use [Conventional Commits](https://www.conventionalcommits.org). Common prefixes:

```
feat: add cuVS GPU backend
fix(rebuild): handle empty vector spaces without panicking
perf(search): batch HNSW lookups in hybrid mode
docs(architecture): clarify rebuild flow
chore: bump tokio to 1.40
```

This drives auto-CHANGELOG generation and informs release decisions. Don't agonize over the prefix; reviewers will fix it on merge if needed.

## Reporting bugs

Open an issue with:
1. Compass version (`compass --version` or git SHA).
2. OS and architecture.
3. Minimal repro: ideally a `curl` sequence against a fresh data directory.
4. Expected vs actual behavior.
5. Relevant logs (run with `RUST_LOG=compass=debug`).

## Reporting security issues

Don't open a public issue. Email `support@runcaptain.com` with the details. We'll acknowledge within two business days.

## Code of conduct

Be respectful. Disagree on technical merits, not on people. Reviewers are expected to focus on code quality; contributors are expected to take feedback in good faith.
