"""Download a Sentinel-1 precise or restituted orbit that covers one acquisition.

The files come from the public ESA STEP orbit archive. A file is accepted
only when its name is the right platform and its validity window contains
the acquisition, and the EOF state vectors actually cover that time.
"""

from __future__ import annotations

import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ai_insar.pipeline.geometry import Orbit
from ai_insar.pipeline.s1_annotation import read_eof_orbit

_LIST_URL = "https://step.esa.int/auxdata/orbits/Sentinel-1/{kind}/{platform}/{year}/{month:02d}/"
_VALIDITY = re.compile(r"_V(\d{8}T\d{6})_(\d{8}T\d{6})")
_HREF = re.compile(r'href="([^"]+\.EOF(?:\.zip)?)"', re.IGNORECASE)


def download_orbit(
    when: np.datetime64,
    platform: str,
    cache: Path | str,
    *,
    kind: str = "POEORB",
) -> tuple[Orbit, Path]:
    """Download and validate one orbit file. ``kind`` is ``POEORB`` or ``RESORB``."""
    folder = Path(cache)
    folder.mkdir(parents=True, exist_ok=True)
    platform = platform.strip().upper()
    if platform not in {"S1A", "S1B", "S1C", "S1D"}:
        raise RuntimeError(f"{platform} is not a Sentinel-1 platform for orbit download.")
    kind = kind.strip().upper()
    if kind not in {"POEORB", "RESORB"}:
        raise RuntimeError(f"{kind} is not POEORB or RESORB.")
    moment = _as_datetime(when)
    listing = _listing(kind, platform, moment)
    name = select_orbit_name(listing, moment, platform)
    if name is None:
        raise RuntimeError(
            f"No {platform} {kind} file in the ESA listing covers {moment.isoformat()}. "
            "State vectors were not changed."
        )
    destination = folder / Path(name).name
    if not destination.is_file():
        url = _LIST_URL.format(kind=kind, platform=platform, year=moment.year, month=moment.month) + Path(name).name
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, temporary)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download {url}: {exc}") from exc
        temporary.replace(destination)
    orbit = read_eof_orbit(destination)
    validate_orbit(orbit, when, platform=platform, name=destination.name)
    return orbit, destination


def select_orbit_name(names: list[str], when: datetime, platform: str) -> str | None:
    """Pick the file whose validity window contains ``when`` and whose name starts with the platform."""
    platform = platform.upper()
    matches = []
    for name in names:
        base = Path(name).name
        if not base.upper().startswith(platform):
            continue
        found = _VALIDITY.search(base)
        if found is None:
            continue
        start = _stamp(found.group(1))
        stop = _stamp(found.group(2))
        if start <= when <= stop:
            matches.append(base)
    if not matches:
        return None
    precise = [name for name in matches if "POEORB" in name.upper()]
    return sorted(precise or matches)[-1]


def validate_orbit(orbit: Orbit, when: np.datetime64, *, platform: str, name: str) -> None:
    """Reject an EOF that does not cover the acquisition or is not a low-Earth orbit."""
    token = platform.upper()
    upper = name.upper()
    if any(code in upper for code in ("S1A", "S1B", "S1C", "S1D")) and token not in upper:
        raise RuntimeError(f"{name} is not a {platform} orbit file.")
    if when < orbit.time[0] or when > orbit.time[-1]:
        raise RuntimeError(
            f"{name} state vectors run {orbit.time[0]} to {orbit.time[-1]} and do not cover {when}."
        )
    _position, velocity = orbit.interpolate(when, 3)
    speed = float(np.linalg.norm(velocity))
    if not 7000.0 < speed < 8000.0:
        raise RuntimeError(f"{name} speed at {when} is {speed:.1f} m/s, not a Sentinel-1 orbit.")


def _listing(kind: str, platform: str, moment: datetime) -> list[str]:
    url = _LIST_URL.format(kind=kind, platform=platform, year=moment.year, month=moment.month)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"Could not list {url}: {exc}") from exc
    return _HREF.findall(text)


def _stamp(text: str) -> datetime:
    return datetime.strptime(text, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _as_datetime(when: np.datetime64) -> datetime:
    seconds = (np.datetime64(when, "ns") - np.datetime64("1970-01-01T00:00:00", "ns")) / np.timedelta64(1, "s")
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc)
