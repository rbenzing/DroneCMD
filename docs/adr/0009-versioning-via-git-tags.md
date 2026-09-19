# ADR-0009: Versioning derived from git tags

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

Hardcoded version strings drift from reality and invite merge conflicts and
"forgot to bump" mistakes. The project needs a single, unambiguous source of
version truth wired to releases.

## Decision

Versions are **derived from git tags** by `setuptools_scm`. There is no hardcoded
version string:

- `__init__.__version__` and `constants.FRAMEWORK_VERSION` resolve from installed
  package metadata (falling back to the scm-written `_version.py`, then
  `"0.0.0+unknown"`); `_version.py` is git-ignored.
- To cut a release: `git tag vX.Y.Z && git push origin vX.Y.Z`. Pushing a `v*`
  tag triggers `.github/workflows/release.yml`, which builds the sdist + wheel
  (pinned to the tag) and publishes a GitHub Release.
- Tags are semver; the tag **is** the version.

## Consequences

### Positive
- One source of truth; releasing is one tag push; no bump commits.

### Negative / trade-offs
- Contributors must know not to reintroduce a static `version =`/`__version__ =` or bump2version.

### Neutral / notes
- v0.3.0 (this milestone) shipped P1–P3d: OFDM full chain + channel coding through BCH.
