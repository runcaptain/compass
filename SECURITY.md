# Security Policy

## Reporting a Vulnerability

We take the security of Compass seriously. If you discover a security vulnerability, please email founders@runcaptain.com with the following information:

1. **Description** of the vulnerability
2. **Steps to reproduce** (if applicable)
3. **Affected versions** (or commit SHA if pre-release)
4. **Potential impact** (severity assessment)
5. **Proposed fix** (if you have one)

We will acknowledge your report within **two business days** and work with you to understand and resolve the issue. Please do not publicly disclose the vulnerability until we have released a fix.

## Security Best Practices for Users

### Data Protection

- Compass stores all data locally on disk. Ensure your DATA_DIR is protected with appropriate file system permissions.
- For on-premises deployments, restrict network access to the Compass HTTP API (default port 4001) using firewalls or reverse proxies.
- Use TLS/SSL when deploying Compass behind a reverse proxy (nginx, HAProxy, etc.).

### Dependency Updates

- Regularly run `cargo audit` to check for known vulnerabilities in dependencies.
- Monitor GitHub's security advisories for the Compass repository.
- Keep your Rust toolchain up to date.

### Model Weights

- Compass downloads model weights on first run (e.g., BGE-small via Hugging Face Hub).
- Verify downloaded files match expected checksums when possible.
- For air-gapped deployments, pre-download and verify model weights before use.

## Supported Versions

Security updates are provided for the current and previous minor versions. Check the CHANGELOG.md for version information.

## Acknowledgments

We appreciate the security research community's efforts in identifying and responsibly disclosing vulnerabilities. We will acknowledge security researchers in our release notes when appropriate.
