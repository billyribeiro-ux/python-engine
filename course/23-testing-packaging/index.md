# Module 23 — Testing, packaging, distribution

Writing code that works is half the job. Proving it keeps working, packaging it so others (or future you) can install it, and distributing it via PyPI / a private index is the other half. This module is the modern Python toolkit for both.

Pages:

1. **[pytest fundamentals](01-pytest-fundamentals.md)** — assertions, parametrize, markers, project layout.
2. **[Fixtures and dependency injection](02-fixtures.md)** — scopes, factories, conftest.py.
3. **[Mocking and patching](03-mocking.md)** — `unittest.mock`, `pytest-mock`, side_effect, autospec.
4. **[Property-based testing with Hypothesis](04-hypothesis.md)** — strategies, shrinking, stateful tests.
5. **[Coverage and mutation testing](05-coverage-and-mutation.md)** — coverage.py, mutmut, what coverage actually means.
6. **[pyproject.toml and modern build systems](06-pyproject.md)** — PEP 517/621, hatch, setuptools, flit.
7. **[Dependency management — uv, pip-tools, poetry](07-dependency-management.md)** — lockfiles, resolution, dev vs runtime.
8. **[Publishing to PyPI and private indexes](08-publishing.md)** — twine, OIDC trusted publishers, internal repositories.

Start with **[pytest fundamentals](01-pytest-fundamentals.md)**.
