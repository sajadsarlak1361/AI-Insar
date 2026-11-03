# AI Insar on Windows

AI Insar is a Python desktop app. The Windows launcher creates a virtual
environment, installs the package, and opens the GUI. You need Python 3.10 or
newer and a network connection the first time, so pip can download NumPy,
SciPy, and PySide6.

## Run from a source checkout

1. Install Python from [python.org](https://www.python.org/downloads/windows/).
   Enable **Add python.exe to PATH**.
2. Unzip the project, or clone the repository.
3. Double-click `run_windows.bat` in the project folder.

The script creates `.venv` next to itself, runs `pip install -e .`, then
starts `python -m ai_insar`. Leave the console window open while the GUI is
running. The first launch is the slow one; later launches reuse `.venv`.

If the window closes immediately, open `cmd` in the project folder and run
`run_windows.bat` so you can read the error. The usual cause is `python` not
being on `PATH`.

## Headless check and your own SLC

From the same folder, after the batch file has created the venv:

```bat
.venv\Scripts\python -m ai_insar check
.venv\Scripts\python -m ai_insar run --demo --output products
.venv\Scripts\python -m ai_insar run --slc C:\data\slc.npy --output products
```

`check` processes the built-in synthetic C-band subsidence bowl and prints
`PASS` or `FAIL`. It does not open a window. `slc.npy` must be `complex64`
with shape `(dates, height, width)`. An optional `slc.json` beside it may set
`wavelength_m` and `dates`. See the README for the interferogram convention
and the limitations.

## Source zip

A Windows source zip is not committed (dependencies are not frozen into it).
Build it from any checkout:

```bat
python -m ai_insar.packaging
```

or, after `pip install -e .`:

```bat
ai-insar package-windows
```

Both write `dist/ai-insar-windows.zip`. Unzip that archive and double-click
`run_windows.bat` inside the `ai-insar` folder. You can also run
`scripts\make_windows_zip.py` before the package is installed; it adds the
source tree to `sys.path` itself.

The zip contains the MIT license, this file, the README, the batch file, and
the Python package. It does not contain a virtual environment.
