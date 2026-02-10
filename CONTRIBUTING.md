# Contributing to GroundCheck

## Quick Start

```bash
git clone https://github.com/blockhead22/GroundCheck.git
cd GroundCheck
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Rules

1. **Zero runtime dependencies.** If your change adds a `pip install` requirement, it belongs behind an optional extra (like `[neural]` or `[mcp]`), not in core.
2. **Tests required.** Every PR needs tests. Run `pytest` before submitting.
3. **Type hints required.** This is a `py.typed` package.

## Running Tests

```bash
pytest                     # all tests
pytest -x                  # stop on first failure
pytest -k "test_verify"    # run specific tests
pytest --cov=groundcheck   # with coverage
```

## Reporting Issues

Open an issue at https://github.com/blockhead22/GroundCheck/issues with:
- Python version
- Minimal reproduction code
- Expected vs actual output
