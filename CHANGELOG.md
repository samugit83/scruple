# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The CLI's `--json` output is a public, versioned interface (plan §13.1).
Breaking it is a breaking change.

## [Unreleased]

### Added
- Initial implementation of the v1 scope: statistics layer, codebook and corpus
  loading with a frozen three-way split, pluggable backends, caching engine,
  blind gold collection, Mondrian threshold selection, review, export and the
  validation report.
