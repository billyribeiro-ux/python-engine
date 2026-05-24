# Publishing to PyPI and private indexes

You've built a package. Now you need to ship it — to PyPI for open source, or to a private index for internal use. This chapter is the working workflow plus the modern OIDC pattern that eliminates long-lived API tokens.

## The build artifacts

A Python package distributes as either:

- **Source distribution (`.tar.gz` / `sdist`)** — raw source; users build on install. Needed if the package has compiled extensions.
- **Wheel (`.whl`)** — pre-built; users just unpack. Faster install; preferred when possible.

```bash
pip install build
python -m build
ls dist/
# mypackage-0.1.0-py3-none-any.whl
# mypackage-0.1.0.tar.gz
```

For pure-Python packages, the single wheel works everywhere. For C / Rust extensions, build wheels per OS / arch / Python version (use `cibuildwheel` in CI).

## Uploading to PyPI — the classic way

```bash
pip install twine
twine upload dist/*
# Prompts for username = __token__ and password = your PyPI API token
```

Store the token in `~/.pypirc`:

```ini
[pypi]
  username = __token__
  password = pypi-AgEIcHl...
```

Test the upload on TestPyPI first:

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ mypackage
```

If it installs cleanly from TestPyPI, you're safe to ship to real PyPI.

## OIDC — trusted publishers (the modern way)

PyPI supports OpenID Connect via GitHub Actions / GitLab CI. **No long-lived tokens stored anywhere.** Set up:

1. On PyPI, go to your project → Settings → Publishing → "Add a new publisher" → GitHub.
2. Fill in: owner, repository, workflow filename, environment (optional).
3. Save.

In `.github/workflows/publish.yml`:

```yaml
name: Publish

on:
  push:
    tags: ["v*"]

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write    # required for OIDC

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Build
        run: |
          python -m pip install --upgrade build
          python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

Push a tag (`git tag v0.1.0 && git push --tags`) → CI builds and publishes. No secrets in the repo, no tokens on contributors' machines.

This is the recommended pattern in 2026. Set it up once; never think about credentials again.

## Versioning

Semantic versioning is the convention:

- `MAJOR.MINOR.PATCH` (e.g., `2.4.1`).
- `MAJOR` bump → breaking API change.
- `MINOR` bump → backwards-compatible feature addition.
- `PATCH` bump → backwards-compatible bug fix.

For pre-release: `1.0.0a1` (alpha), `1.0.0b2` (beta), `1.0.0rc1` (release candidate), `1.0.0.dev3` (dev).

## Tagging and changelog

```bash
# Tag a release
git tag v0.2.0 -m "Release 0.2.0"
git push --tags

# CI picks up the tag and publishes
```

Maintain a `CHANGELOG.md` in [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
# Changelog

## [0.2.0] - 2024-11-15
### Added
- Support for vol surface fitting via SVI.
### Changed
- `Feed.bars()` now returns a tz-aware DataFrame.
### Fixed
- Edge case in implied vol solver near vega=0.

## [0.1.0] - 2024-10-01
### Added
- Initial release.
```

`towncrier` automates this from per-PR `changes/` fragments. Worth it past about 10 contributors.

## Private indexes

For internal packages (proprietary trading code, internal tooling), don't push to PyPI. Use a private index:

| Tool | Hosted by you | Notes |
|---|---|---|
| **devpi** | self-host | classic; full PyPI mirror + private repos |
| **Pypiserver** | self-host | minimal; fine for small teams |
| **AWS CodeArtifact** | AWS | IAM-integrated; pay-per-GB |
| **GCP Artifact Registry** | GCP | similar |
| **GitHub Packages** | GitHub | requires `~/.pypirc` setup |
| **Cloudsmith / JFrog / Artifactory** | SaaS | enterprise-grade |

Publishing to an internal index:

```toml
# ~/.pypirc
[distutils]
index-servers =
    pypi
    private

[private]
repository = https://pypi.internal.example.com/
username = __token__
password = ${PRIVATE_PYPI_TOKEN}
```

```bash
twine upload --repository private dist/*
```

Installing from it:

```bash
pip install --index-url https://pypi.internal.example.com/simple/ mypackage
# Or in requirements.txt:
# --index-url https://pypi.internal.example.com/simple/
# mypackage>=0.1.0
```

For mixed (some public PyPI, some private): `--extra-index-url`.

## Pre-publication checklist

Before pushing v1.0.0:

- [ ] `pyproject.toml` has correct name, description, classifiers, license.
- [ ] README has install instructions and at least one usage example.
- [ ] CHANGELOG documents the release.
- [ ] Tests pass on all supported Python versions (matrix CI).
- [ ] Coverage is above your project's floor.
- [ ] `python -m build` produces a wheel + sdist without warnings.
- [ ] `twine check dist/*` passes (validates the package metadata).
- [ ] Test-install on TestPyPI; verify it imports cleanly.
- [ ] Tag the commit with the version.

## Build wheels for multiple platforms

For pure-Python: one wheel works everywhere (`py3-none-any.whl`).

For C / Rust extensions: build per platform. `cibuildwheel` automates this in CI:

```yaml
# .github/workflows/build-wheels.yml
- uses: pypa/cibuildwheel@v2.20.0
  env:
    CIBW_BUILD: "cp311-* cp312-* cp313-*"
    CIBW_ARCHS_LINUX: "x86_64 aarch64"
    CIBW_ARCHS_MACOS: "x86_64 arm64"
    CIBW_ARCHS_WINDOWS: "AMD64"
```

Produces ~30 wheels covering Python 3.11-3.13 on Linux x86_64 + arm64 + macOS Intel + Apple Silicon + Windows.

## Naming and squatting

PyPI is first-come-first-served on package names. Reserve your name early. If your project is "the thing for X," secure `the-thing-for-x` on PyPI even before you ship — publishing an empty 0.0.1 with a "coming soon" README is fine.

## Pitfalls

!!! warning "Uploading the same version twice"
    PyPI rejects re-uploads of the same version. If you find a bug in `0.1.0` after release, you must publish `0.1.1` — even if the bug is trivial. Tag versions deliberately.

!!! warning "Forgetting to bump version"
    Build → upload → "file already exists." Bump in `pyproject.toml` (or by git tag if dynamic), tag, push.

!!! warning "Including secrets in the package"
    `python -m build` includes everything in your repo. Audit before pushing — a `.env` file in the wheel is the kind of mistake that ends careers. Use `.gitignore` + `MANIFEST.in` / `hatch.build.exclude`.

!!! warning "Missing `LICENSE` file"
    PyPI accepts but tools complain. Always commit `LICENSE`.

## Bottom line

For publishing:

- **`python -m build`** for the artifacts.
- **OIDC trusted publishers** for CI-driven release (no tokens).
- **Tag releases** in git; CI builds + publishes.
- **TestPyPI dry-run** before real PyPI.
- **Private indexes** for proprietary packages (codeartifact / artifactory / devpi).
- **CHANGELOG + semver** for honest version communication.

## End of Module 23

You now have the testing + packaging + distribution toolkit. The next module covers documents, PDFs, images, and OCR — the things you'll generate and process when integrating with non-technical stakeholders.

Continue to **[Module 24 — Documents, PDF, images, OCR](../24-documents-and-media/index.md)**.
