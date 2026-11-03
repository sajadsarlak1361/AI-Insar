# AI Insar on Windows

This page is the Windows path for the same app described in the
[README](README.md). You need **Python 3.10 or newer** and, the first time,
a network connection so pip can download NumPy, SciPy, and PySide6.

The launcher does not bundle those libraries. It creates a virtual environment
and installs the local project.

## 1. Install Python

1. Download Python from [python.org](https://www.python.org/downloads/windows/).
2. Run the installer and enable **Add python.exe to PATH**.
3. Open a new **Command Prompt** and check:

```bat
python --version
```

You should see Python 3.10 or newer. If `python` is not recognized, the
launcher will stop and wait: it prints that message and `pause`s so the
window does not vanish.

## 2. Get the source

Either clone the repository, or unzip a source archive (see
[Build the Windows zip](#build-the-windows-zip) below). The folder you use must
contain `run_windows.bat`, `pyproject.toml`, and `src\`.

## 3. Start the app

Double-click `run_windows.bat`, or from Command Prompt in that folder:

```bat
run_windows.bat
```

The script always `cd`s to its own directory (`%~dp0`), then does only this:

1. `where python`. If that fails, it tells you to install Python 3.10+ and pauses.
2. If `.venv\Scripts\python.exe` does not exist, `python -m venv .venv`.
3. `call .venv\Scripts\activate.bat`.
4. `python -m pip install --upgrade pip`.
5. `pip install -e .` (the runtime package, not the pytest extra).
6. `python -m ai_insar`, which opens the GUI because there is no subcommand.

Leave the console open while the window is running. The first launch is the
slow one. Later launches reuse `.venv` and only re-run pip if you start the
batch file again (pip will see that the editable install is already there).

If install or the GUI returns a non-zero status, the batch file `pause`s.
If the window closes before you can read it, run `run_windows.bat` from
`cmd` instead of double-clicking.

After the venv exists you can also activate it yourself and use the
installed command:

```bat
.venv\Scripts\activate
ai-insar
```

`ai-insar` and `ai-insar-gui` are the same entry point. The batch file calls
`python -m ai_insar`, which does not depend on those Scripts launchers.

## 4. Headless check, demo products, and your own SLC

These commands do not open a window. Run them from the project folder with
the venv interpreter:

```bat
.venv\Scripts\python -m ai_insar check
.venv\Scripts\python -m ai_insar run --output products
.venv\Scripts\python -m ai_insar run --slc C:\data\slc.npy --output products
.venv\Scripts\python -m ai_insar run --slc C:\data\slc.npy --meta C:\data\slc.json --output products
```

`check` processes the built-in synthetic C-band subsidence bowl and prints
`PASS` or `FAIL`. Exit code 0 means pass. The stack is `complex64` with
shape `(dates, height, width)`. Interferograms are later × conjugate(earlier).
LOS velocity is negative when the ground moves away from the radar.

`--output` is required for `run`. Without `--slc`, `run` uses the built-in
bowl. `--meta` is read only when `--slc` is set. If `slc.json` sits next to
`slc.npy` and shares the stem, you can omit `--meta`. The sidecar format is
in [docs/data-format.md](docs/data-format.md).

`products\` then contains `velocity_m_yr.npy`, `mask.npy`, `los_m.npy`,
`products.json`, and the other arrays listed in that guide.

## Build the Windows zip

The zip is not committed. Building it does not need PySide6; it only
copies source files. From the repo root, before the package is installed:

```bat
python scripts\make_windows_zip.py
```

That script puts `src` on `sys.path` and calls `ai_insar.packaging`. After
`pip install -e .`, either of these does the same thing:

```bat
python -m ai_insar.packaging
ai-insar package-windows
```

All three write `dist/ai-insar-windows.zip`. The archive’s top folder is
`ai-insar\`. Unzip it and double-click `ai-insar\run_windows.bat`.

The zip contains:

- `README.md`, `WINDOWS.md`, `CONTRIBUTING.md`, `LICENSE`, `pyproject.toml`
- `run_windows.bat`
- `docs\` (the guides linked from the README)
- `src\`, `tests\`, `scripts\`

It does **not** contain `.venv`, `.git`, or third-party wheels. The machine
that unzips it still needs Python, network, and `run_windows.bat` to install
NumPy, SciPy, and PySide6.

## If something fails

| What you see | What it means |
| --- | --- |
| `Python was not found on PATH` | The installer option **Add python.exe to PATH** was off, or the prompt was opened before install. The batch file pauses. |
| `Failed to create the virtual environment` | `python -m venv .venv` returned an error. The batch file pauses. Check that `python --version` works in that same prompt. |
| `Install failed` | pip could not download or build dependencies. Read the pip log above the pause. A network block is the usual cause. |
| The GUI starts and the console stays open | That is expected. Closing the console stops the app. |
| `check` prints `FAIL` | The synthetic self-test missed its thresholds. The exit code is 1. This does not mean a user SLC was processed. |

Processing limits (no geocoding, no orbits, no weather model) are the same
on Windows as everywhere else: [docs/limitations.md](docs/limitations.md).
