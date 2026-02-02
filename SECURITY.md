# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |
| < 1.0   | No        |

## Reporting a Vulnerability

If you find a security issue, please report it privately rather than opening a public issue.

**Email:** security@dlorp.dev

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (if you have them)

## Scope

r³LAY runs locally by default. The main areas of concern:

- **Local file access** - r³LAY indexes files you point it at. It doesn't reach outside the project directory unless explicitly configured.
- **Network endpoints** - Optional connections to Ollama, SearXNG, or other local services. No data leaves your machine unless you configure external endpoints.
- **Model loading** - GGUF/MLX models are loaded from paths you specify. Don't load models from untrusted sources.

## Out of Scope

- Vulnerabilities in third-party dependencies (report those upstream, but let me know if it affects r³LAY)
- Issues requiring physical access to your machine
- Social engineering

## Security Practices

- No telemetry, no analytics, no cloud sync
- All data stays in your project directory under `.r3lay/`
- API endpoints (when enabled) bind to localhost by default# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |
| < 1.0   | No        |

## Reporting a Vulnerability

If you find a security issue, please report it privately rather than opening a public issue.

**Email:** security@dlorp.dev

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (if you have them)

## Scope

r³LAY runs locally by default. The main areas of concern:

- **Local file access** - r³LAY indexes files you point it at. It doesn't reach outside the project directory unless explicitly configured.
- **Network endpoints** - Optional connections to Ollama, SearXNG, or other local services. No data leaves your machine unless you configure external endpoints.
- **Model loading** - GGUF/MLX models are loaded from paths you specify. Don't load models from untrusted sources.

## Out of Scope

- Vulnerabilities in third-party dependencies (report those upstream, but let me know if it affects r³LAY)
- Issues requiring physical access to your machine
- Social engineering

## Security Practices

- No telemetry, no analytics, no cloud sync
- All data stays in your project directory under `.r3lay/`
- API endpoints (when enabled) bind to localhost by default
