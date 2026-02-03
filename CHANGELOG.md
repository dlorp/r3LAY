# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.5.0]: https://github.com/dlorp/r3LAY/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/dlorp/r3LAY/releases/tag/v0.4.0
