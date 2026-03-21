# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in LizyML, please report it responsibly:

1. **Do NOT** open a public issue
2. Use [GitHub Security Advisories](https://github.com/nbx-liz/LizyML/security/advisories/new) to report privately
3. Include steps to reproduce, impact assessment, and any suggested fixes

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest release | Yes |
| Previous minor | Best effort |
| Older versions | No |

## Security Considerations

LizyML processes user-provided data and configuration. Key security areas:

- **Deserialization**: Model artifacts use `joblib`. Only load artifacts from trusted sources.
- **Code generation**: `export_code()` generates executable Python scripts. Review generated code before running in production.
- **Dependencies**: We pin minimum versions and use Dependabot for automated updates.
