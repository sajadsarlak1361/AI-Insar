# Command line

The console script is `ai-insar` (`ai_insar.cli:main`). `python -m ai_insar`
is the same function. `ai-insar-gui` is a second installed name for that
function, aimed at a windowed launcher. With no subcommand, all of them open
the GUI.

```text
ai-insar [-h] [--version] {gui,check,run,package-windows} ...
```

`ai-insar --help` describes the tool as an SLC-first workbench and states
that interferograms are later × conjugate(earlier). `ai-insar --version`
prints `ai-insar 0.1.0`.

## `gui` (default)

```bash
ai-insar
ai-insar gui
python -m ai_insar
```

Opens the PySide6 window. If Qt cannot be imported, the process prints

```text
The desktop GUI could not be imported. Install PySide6 (pip install PySide6) and retry.
```

to stderr and returns 1. This subcommand has no other flags.

## `check`

```bash
ai-insar check
python -m ai_insar check
```

Builds the default synthetic stack (`generate_demo_stack`), runs every
processing stage, and compares the velocity and mask to the known bowl.
No flags. No window. Exit 0 on `PASS`, exit 1 on `FAIL`.

The pass rule in `recovery_metrics` is all of:

- the 3×3 mean around the strongest true pixel has the same sign as the truth
- relative error of that 3×3 mean ≤ 0.40
- correlation of the velocity maps ≥ 0.80
- recall on the core (`|truth| ≥ 0.5 × peak`) ≥ 0.70
- false-positive rate outside the bowl (`|truth| < 0.15 × peak`) ≤ 0.15

The headline “truth” is that 3×3 mean, not the single-pixel value logged
earlier. A user SLC is not scored; `check` always uses the built-in bowl.

Example from version 0.1.0, seed 7 (the text `check` prints):

```text
SLC stack (8, 72, 72) complex64, wavelength 0.055466 m, sensor synthetic C-band.
Synthetic C-band SLC stack, 8 dates every 36 days, Gaussian subsidence bowl sigma=11 px, peak LOS rate -0.2200 m/yr (negative = away from the radar), seed=7.
Synthetic truth peak LOS velocity -0.2195 m/yr at row 35, col 35 (negative = away from the radar).
Interferograms: 18 pairs (max date span 3, temporal baselines 36–108 days). Each is later × conjugate(earlier).
First pair dates 0→1 at the peak pixel: deformation phase 4.90 rad (0.78 fringes, positive = subsidence), wrapped SLC interferogram -0.95 rad.
Goldstein filter: alpha=0.50, window=32. Wrapped residues after filtering: 0 (per pair min 0, max 0).
Coherence: 5×5 window on each SLC pair. Median 0.886, 5th percentile 0.628.
Unwrapped filtered phase with quality-guided integration and removed a border plane (4 px). Finite pixels: 100.0%.
SBAS: displacement at each date (first date = 0) and a linear LOS velocity. Velocity min -0.2253, max 0.0059 m/yr. Negative velocity is motion away from the radar.
At the synthetic peak pixel (row 35, col 35): estimated -0.2236 m/yr, truth -0.2195 m/yr.
Displacement mask: 990 pixels (|z| ≥ 3.5 and |velocity| ≥ 28% of the coherent peak, coherence ≥ 0.3).
Peak LOS velocity -0.2225 m/yr (truth -0.2183, relative error 1.92%).
Velocity correlation 0.999. Core recall 1.000. Outside false-positive rate 0.000. Mask pixels 990.
PASS
```

The wrapped phase on the first pair is negative (−0.95 rad) even though the
unwrapped deformation is positive (4.90 rad, 0.78 fringe). That is the wrap,
not a sign bug.

## `run`

```text
ai-insar run --output OUTPUT [--demo | --slc SLC] [--meta META]
```

`--output` is required. It is the directory for the `.npy` products and
`products.json` ([data-format.md](data-format.md)).

| Flag | Effect |
| --- | --- |
| `--output DIR` | Created if needed. Required. |
| `--demo` | Use `generate_demo_stack()`. Mutually exclusive with `--slc`. |
| `--slc PATH` | Load a complex `.npy`. Mutually exclusive with `--demo`. |
| `--meta PATH` | Sidecar JSON. Used only when `--slc` is set. If omitted, `<slc-stem>.json` is used when it exists. |

If neither `--demo` nor `--slc` is passed, the command still uses the
built-in stack and prints `Using the built-in synthetic C-band SLC stack.`
`--demo` with `--meta` is accepted by argparse and the meta path is ignored,
because the demo generator does not read a sidecar.

Exit code is 0 after a successful write. A bad SLC (real dtype, wrong
number of dates, non-finite values) raises and the process exits non-zero
with a traceback.

```bash
ai-insar run --output products
ai-insar run --demo --output products
ai-insar run --slc slc.npy --output products
ai-insar run --slc slc.npy --meta meta.json --output products
```

## `package-windows`

```bash
ai-insar package-windows
python -m ai_insar.packaging
python scripts/make_windows_zip.py
```

No flags. Prints the path of the zip it wrote, normally
`dist/ai-insar-windows.zip`, and exits 0. The archive layout is described in
[WINDOWS.md](../WINDOWS.md). `package-windows` needs to find `pyproject.toml`
and `run_windows.bat` (the project root, or the current directory).

## Help for one command

```bash
ai-insar run --help
ai-insar check --help
```

`check` and `package-windows` only add `-h`.
