# Contributing

AI Insar is an MIT-licensed SLC workbench. Issues and pull requests are
welcome on [github.com/sajadsarlak1361/AI-Insar](https://github.com/sajadsarlak1361/AI-Insar).

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ai-insar check
```

`pytest` must stay green. It covers the interferogram sign
(`later * conj(earlier)`), LOS sign (subsidence is negative velocity),
unwrap and SBAS, the demo bowl, the CLI, an offscreen click of the seven
stage buttons, and the markdown links in the docs.

## Conventions to keep

- Primary input remains a `complex64` stack shaped `(dates, height, width)`.
- Do not flip the phase equation in `ai_insar.conventions` without updating
  the tests, the Help text, and [docs/pipeline.md](docs/pipeline.md).
- The displacement mask is a statistical test. Do not describe it as a
  trained model.
- Do not document commands that `ai-insar --help` does not actually have.
  [docs/cli.md](docs/cli.md) is written from that help and from `cli.py`.
- The app is not a replacement for ISCE, MintPy, SNAP, or SNAPHU. New
  features should say which part of that gap they still leave open
  ([docs/limitations.md](docs/limitations.md)).

## Docs

User-facing pages are `README.md`, `WINDOWS.md`, and `docs/`. `tests/test_docs.py`
fails if a relative link in those files points at a missing path. The
Windows zip (`ai-insar package-windows`) includes `docs/` and
`CONTRIBUTING.md`; if you add a new top-level guide, add it to
`_TOP_FILES` or `_TOP_DIRS` in `ai_insar.packaging` or the zip will omit it.

## License

By contributing, you agree that your changes are released under the
[MIT license](LICENSE).
