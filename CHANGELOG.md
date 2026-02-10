# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes._

## [0.7.0] - 2026-02-10

### Added
- Maintenance commands fully wired and functional (log, due, history, mileage)
- Natural language input support for maintenance logging
- Configurable intent routing (local/OpenClaw/auto)
- LLM conversational feedback integration
- GGUF model auto-discovery for local backends
- OpenClaw HTTP API backend documented
- vLLM backend support documented
- Command documentation expanded from 7 to 21 commands

### Changed
- Enhanced maintenance tracking with natural language parsing
- Improved backend configuration flexibility
- Expanded documentation coverage

## [0.6.1] - 2025-02-03

### Docs
- Updated CHANGELOG.md to document v0.6.0 changes

## [0.6.0] - 2025-02-03

### Added
- Version badge to README
- Test coverage for UI widgets (GarageHeader, ResponsePane, SessionPanel, Splash)

### Security
- Secure logging configuration to prevent sensitive data exposure

## [0.5.0] - 2025-02-03

### Added
- History CLI command (`/history`) for viewing maintenance history
- Comprehensive test suite for app module
- Test coverage for core init, backends, intent parser, and settings panel
- Test coverage for axioms and signals modules
- GarageHeader widget integrated into main screen
- Model swap via conversation interface
- Log and Due tabs in Phase 2 layout

### Changed
- Code cleanup and quality improvements with ruff
- Documentation accuracy improvements
- README streamlined with improved structure

### Fixed
- Permission error handling in `detect_format`
- Duplicate TestHistory class in test_cli.py
- Em-dash and backronym text issues

### Security
- Secure logging configuration to prevent sensitive data exposure
- Path validation to prevent path traversal in file attachments

## [0.4.0] - 2025-01-30

### Added
- Phase 2 UI panels (Log, Due tabs)
- OpenClaw backend integration
- vLLM backend support
- SearXNG module for web search
- Session persistence UI
- Axiom enhancements for knowledge validation

### Changed
- Migrated to Textual 0.47.0+
- Improved intent parsing logic

### Fixed
- Various type checking import issues
- Signal test stability improvements

[Unreleased]: https://github.com/dlorp/r3LAY/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/dlorp/r3LAY/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/dlorp/r3LAY/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/dlorp/r3LAY/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/dlorp/r3LAY/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/dlorp/r3LAY/releases/tag/v0.4.0
