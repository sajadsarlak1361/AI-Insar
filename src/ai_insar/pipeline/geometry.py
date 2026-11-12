"""WGS84 geometry and polynomial orbit interpolation.

The equations are the usual radar ones: geodetic conversion, a polynomial
through nearby orbit state vectors, zero-Doppler time, and right-looking
ellipsoid geolocation. No SNAP or GMTSAR source is included.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B_M = WGS84_A_M * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
SPEED_OF_LIGHT_M_S = 299_792_458.0


def parse_utc(text: str) -> np.datetime64:
    """Parse an ISO time, including the ``UTC=`` prefix used in orbit files."""
    value = text.strip()
    if value.upper().startswith("UTC="):
        value = value[4:]
    value = value.replace("Z", "")
    if "." in value:
        head, frac = value.split(".", 1)
        frac = "".join(ch for ch in frac if ch.isdigit())[:9].ljust(9, "0")
        value = f"{head}.{frac}"
    return np.datetime64(value)


def seconds_since(times: np.ndarray, epoch: np.datetime64) -> np.ndarray:
    delta = np.asarray(times, dtype="datetime64[ns]") - np.datetime64(epoch, "ns")
    return delta.astype(np.float64) * 1e-9


def geodetic_to_ecef(lat_rad: np.ndarray, lon_rad: np.ndarray, height_m: np.ndarray) -> np.ndarray:
    """Convert geodetic latitude, longitude, and height to ECEF metres."""
    lat = np.asarray(lat_rad, dtype=np.float64)
    lon = np.asarray(lon_rad, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    normal = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (normal + height) * cos_lat * np.cos(lon)
    y = (normal + height) * cos_lat * np.sin(lon)
    z = (normal * (1.0 - WGS84_E2) + height) * sin_lat
    return np.stack((x, y, z), axis=-1)


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bowring's method. Returns latitude, longitude in radians and height in metres."""
    point = np.asarray(xyz, dtype=np.float64)
    x = point[..., 0]
    y = point[..., 1]
    z = point[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = np.sin(lat)
        normal = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        height = p / np.cos(lat) - normal
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * normal / (normal + height)))
    sin_lat = np.sin(lat)
    normal = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    height = p / np.cos(lat) - normal
    return lat, lon, height


@dataclass
class Orbit:
    """Earth-fixed state vectors. Times increase."""

    time: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype="datetime64[ns]")
        self.position_m = np.asarray(self.position_m, dtype=np.float64)
        self.velocity_m_s = np.asarray(self.velocity_m_s, dtype=np.float64)
        if self.time.shape[0] < 2:
            raise ValueError("an orbit needs at least two state vectors")
        if self.position_m.shape != (self.time.shape[0], 3) or self.velocity_m_s.shape != self.position_m.shape:
            raise ValueError("orbit position and velocity must be (n, 3)")
        order = np.argsort(self.time, kind="mergesort")
        self.time = self.time[order]
        self.position_m = self.position_m[order]
        self.velocity_m_s = self.velocity_m_s[order]
        self._epoch = self.time[0]
        self._seconds = seconds_since(self.time, self._epoch)
        self._epoch_ns = np.datetime64(self._epoch, "ns").astype(np.int64)

    def seconds(self, when: np.datetime64 | np.ndarray) -> np.ndarray:
        return seconds_since(np.asarray(when), self._epoch)

    def interpolate(
        self,
        when: np.datetime64 | np.ndarray,
        degree: int = 3,
        states: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate position and velocity with a polynomial of ``degree``.

        ``states`` is the number of surrounding vectors in the fit. The
        default, ``degree + 1``, reproduces a node up to rounding. SNAP's
        flat-earth orbit uses 8 vectors and a cubic least-squares fit.
        """
        degree = int(degree)
        if degree < 1:
            raise ValueError("polynomial degree must be at least 1")
        needed = degree + 1 if states is None else int(states)
        if needed < degree + 1:
            raise ValueError("an orbit fit needs at least degree + 1 state vectors")
        if self.time.shape[0] < needed:
            raise ValueError(f"orbit has {self.time.shape[0]} vectors, need {needed}")
        times = np.atleast_1d(np.asarray(when, dtype="datetime64[ns]"))
        seconds = np.clip(self.seconds(times), float(self._seconds[0]), float(self._seconds[-1]))
        position, velocity = self._evaluate(seconds, degree, needed)
        if np.asarray(when).shape == ():
            return position[0], velocity[0]
        return position, velocity

    def nearest_window(self, when: np.datetime64, count: int = 8) -> Orbit:
        """The ``count`` state vectors SNAP keeps around one burst centre.

        The closest vector is index ``minIdx``. The window is ``minIdx-3``
        through ``minIdx+4`` when that fits, the same ends as
        ``getAdjacentOrbitStateVectors``.
        """
        count = int(count)
        if count < 2:
            raise ValueError("an orbit window needs at least two state vectors")
        count = min(count, int(self.time.shape[0]))
        seconds = float(np.asarray(self.seconds(np.datetime64(when, "ns")), dtype=np.float64).reshape(-1)[0])
        index = int(np.argmin(np.abs(self._seconds - seconds)))
        last = int(self.time.shape[0]) - 1
        if index < 3:
            start = 0
        elif index > last - 4:
            start = max(last + 1 - count, 0)
        else:
            start = index - 3
        stop = min(start + count, int(self.time.shape[0]))
        start = max(stop - count, 0)
        return Orbit(self.time[start:stop], self.position_m[start:stop], self.velocity_m_s[start:stop])

    def _polynomials(self, degree: int, needed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cubic (or ``degree``) coefficients for every orbit window.

        Each window is the same ``degree + 1`` nodes ``interpolate`` used to
        fit one query at a time. Fitting them once keeps a million queries on
        those same coefficients.
        """
        cache = getattr(self, "_poly_cache", None)
        if cache is None:
            cache = {}
            self._poly_cache = cache
        found = cache.get((degree, needed))
        if found is not None:
            return found
        nodes = self._seconds
        windows = nodes.shape[0] - needed + 1
        position = np.empty((windows, degree + 1, 3), dtype=np.float64)
        velocity = np.empty_like(position)
        origins = np.empty(windows, dtype=np.float64)
        for start in range(windows):
            sample_t = nodes[start : start + needed]
            origin = float(sample_t[needed // 2])
            origins[start] = origin
            abscissa = sample_t - origin
            for axis in range(3):
                position[start, :, axis] = np.polyfit(
                    abscissa, self.position_m[start : start + needed, axis], degree
                )
                velocity[start, :, axis] = np.polyfit(
                    abscissa, self.velocity_m_s[start : start + needed, axis], degree
                )
        cache[(degree, needed)] = (origins, position, velocity)
        return cache[(degree, needed)]

    def _evaluate(self, seconds: np.ndarray, degree: int, needed: int) -> tuple[np.ndarray, np.ndarray]:
        origins, position_coeff, velocity_coeff = self._polynomials(degree, needed)
        nodes = self._seconds
        center = np.searchsorted(nodes, seconds)
        start = np.clip(center - needed // 2, 0, nodes.shape[0] - needed).astype(np.int64)
        position = np.empty((seconds.shape[0], 3), dtype=np.float64)
        velocity = np.empty_like(position)
        for window in np.unique(start):
            chosen = start == window
            query = seconds[chosen] - origins[window]
            for axis in range(3):
                position[chosen, axis] = np.polyval(position_coeff[window, :, axis], query)
                velocity[chosen, axis] = np.polyval(velocity_coeff[window, :, axis], query)
        return position, velocity

    def zero_doppler_times(
        self,
        xyz_m: np.ndarray,
        guess: np.ndarray,
        degree: int = 3,
        states: int | None = None,
    ) -> np.ndarray:
        """Zero-Doppler time of many targets.

        The iteration is the scalar solver: clip to the interior nodes, query
        the orbit at the nearest nanosecond, and stop a target when its step
        is below 1e-12 s or the Doppler rate vanishes. Targets do not share
        a step.
        """
        points = np.asarray(xyz_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("zero-Doppler targets must have shape (n, 3)")
        guesses = np.atleast_1d(np.asarray(guess, dtype="datetime64[ns]"))
        if guesses.shape[0] != points.shape[0]:
            raise ValueError("zero-Doppler needs one guess time per target")
        degree = int(degree)
        needed = degree + 1 if states is None else int(states)
        if needed > self.time.shape[0]:
            needed = int(self.time.shape[0])
        low = float(self._seconds[min(degree, needed - 1)])
        high = float(self._seconds[-min(degree, needed - 1) - 1])
        raw = self.seconds(guesses)
        if np.any(raw < float(self._seconds[0]) - 1.0) or np.any(raw > float(self._seconds[-1]) + 1.0):
            raise ValueError(
                "zero-Doppler guess is outside this orbit file. "
                "Use an azimuth time from the same acquisition."
            )
        second = np.clip(raw, low, high)
        done = np.zeros(points.shape[0], dtype=bool)
        for _ in range(20):
            if bool(np.all(done)):
                break
            second = np.where(done, second, np.clip(second, low, high))
            query = self._epoch_ns + np.rint(second * 1e9).astype(np.int64)
            later = self._epoch_ns + np.rint((second + 1e-4) * 1e9).astype(np.int64)
            pos, vel = self.interpolate(query.astype("datetime64[ns]"), degree, states=needed)
            pos2, vel2 = self.interpolate(later.astype("datetime64[ns]"), degree, states=needed)
            doppler = np.sum((points - pos) * vel, axis=1)
            doppler2 = np.sum((points - pos2) * vel2, axis=1)
            rate = (doppler2 - doppler) / 1e-4
            small_rate = np.abs(rate) < 1e-8
            step = np.where(small_rate, 0.0, doppler / np.where(small_rate, 1.0, rate))
            updated = np.where(small_rate, second, second - step)
            second = np.where(done, second, updated)
            done = done | small_rate | (np.abs(step) < 1e-12)
        rounded = self._epoch_ns + np.rint(second * 1e9).astype(np.int64)
        return rounded.astype("datetime64[ns]")

    def zero_doppler_time(self, xyz_m: np.ndarray, guess: np.datetime64, degree: int = 3) -> np.datetime64:
        """Azimuth time at which the target has zero Doppler."""
        point = np.asarray(xyz_m, dtype=np.float64)
        low = float(self._seconds[degree])
        high = float(self._seconds[-degree - 1])
        raw = float(self.seconds(guess))
        if raw < float(self._seconds[0]) - 1.0 or raw > float(self._seconds[-1]) + 1.0:
            raise ValueError(
                "zero-Doppler guess is outside this orbit file. "
                "Use an azimuth time from the same acquisition."
            )
        guess_s = float(np.clip(raw, low, high))
        second = guess_s
        for _ in range(20):
            second = float(np.clip(second, low, high))
            when = self._epoch + np.timedelta64(int(round(second * 1e9)), "ns")
            pos, vel = self.interpolate(when, degree)
            relative = point - pos
            doppler = float(np.dot(relative, vel))
            later = self._epoch + np.timedelta64(int(round((second + 1e-4) * 1e9)), "ns")
            pos2, vel2 = self.interpolate(later, degree)
            doppler2 = float(np.dot(point - pos2, vel2))
            rate = (doppler2 - doppler) / 1e-4
            if abs(rate) < 1e-8:
                break
            step = doppler / rate
            second -= step
            # 1 ns is one orbit-time quantum. Stopping earlier leaves a line error
            # that the ESD unit-phasor sum turns into a block-shift bias.
            if abs(step) < 1e-12:
                break
        return self._epoch + np.timedelta64(int(round(second * 1e9)), "ns")


def right_looking_ground_point(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    slant_range_m: float,
    height_m: float,
) -> np.ndarray:
    """Ground point at ``slant_range_m`` on the right-looking zero-Doppler side.

    The Earth is the WGS84 ellipsoid expanded by ``height_m`` through a local
    spherical radius. The look side is ``velocity × nadir``, which is the
    right-looking direction. Several samples share one call.
    """
    sat = np.asarray(position_m, dtype=np.float64)
    single = sat.ndim == 1
    points = right_looking_ground_points(
        sat.reshape(1, 3) if single else sat,
        np.asarray(velocity_m_s, dtype=np.float64).reshape(1, 3) if single else np.asarray(velocity_m_s, dtype=np.float64),
        np.asarray(slant_range_m, dtype=np.float64).reshape(1),
        np.asarray(height_m, dtype=np.float64).reshape(1),
    )
    return points[0] if single else points


def right_looking_ground_points(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    slant_range_m: np.ndarray,
    height_m: np.ndarray,
) -> np.ndarray:
    """Same ground intersection as :func:`right_looking_ground_point`, for many samples."""
    sat = np.asarray(position_m, dtype=np.float64)
    vel = np.asarray(velocity_m_s, dtype=np.float64)
    slant = np.asarray(slant_range_m, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    radius_sat = np.linalg.norm(sat, axis=1)
    earth_radius = _local_earth_radii(sat) + height
    cos_nadir = (radius_sat**2 + slant**2 - earth_radius**2) / (2.0 * radius_sat * slant)
    nadir = np.arccos(np.clip(cos_nadir, -1.0, 1.0))
    speed = np.linalg.norm(vel, axis=1)
    vhat = vel / speed[:, None]
    down = -sat / radius_sat[:, None]
    down = down - np.sum(down * vhat, axis=1)[:, None] * vhat
    down = down / np.linalg.norm(down, axis=1)[:, None]
    # Sentinel-1 is right-looking. forward × down points to the left of the
    # ground track in this frame, so the right-looking side is the opposite.
    across = np.cross(down, vhat)
    across = across / np.linalg.norm(across, axis=1)[:, None]
    look = np.cos(nadir)[:, None] * down + np.sin(nadir)[:, None] * across
    return sat + look * slant[:, None]


def _local_earth_radius(sat: np.ndarray) -> float:
    """Radius of the WGS84 ellipsoid along the geocentric satellite direction."""
    return float(_local_earth_radii(np.asarray(sat, dtype=np.float64).reshape(1, 3))[0])


def _local_earth_radii(sat: np.ndarray) -> np.ndarray:
    """Radius of the WGS84 ellipsoid along each geocentric satellite direction."""
    direction = sat / np.linalg.norm(sat, axis=1)[:, None]
    factor = (direction[:, 0] / WGS84_A_M) ** 2 + (direction[:, 1] / WGS84_A_M) ** 2 + (direction[:, 2] / WGS84_B_M) ** 2
    return 1.0 / np.sqrt(factor)


def ground_point_at_height(
    orbit: Orbit,
    azimuth_time: np.datetime64,
    slant_range_m: float,
    height_m: float,
    *,
    degree: int = 3,
) -> np.ndarray:
    """Geolocate one radar sample onto a surface of constant ellipsoidal height."""
    single = np.asarray(azimuth_time).shape == ()
    points = ground_points_at_height(
        orbit,
        np.atleast_1d(np.asarray(azimuth_time, dtype="datetime64[ns]")),
        np.atleast_1d(np.asarray(slant_range_m, dtype=np.float64)),
        np.atleast_1d(np.asarray(height_m, dtype=np.float64)),
        degree=degree,
    )
    return points[0] if single else points


def ground_points_at_height(
    orbit: Orbit,
    azimuth_time: np.ndarray,
    slant_range_m: np.ndarray,
    height_m: np.ndarray,
    *,
    degree: int = 3,
    states: int | None = None,
) -> np.ndarray:
    """Geolocate many radar samples onto ellipsoidal heights.

    ``height_m`` is either one height or one height per sample. The four
    corrections are the scalar loop, applied independently to each sample.
    """
    position, velocity = orbit.interpolate(azimuth_time, degree, states=states)
    slant = np.asarray(slant_range_m, dtype=np.float64)
    height = np.broadcast_to(np.asarray(height_m, dtype=np.float64), slant.shape).copy()
    correction = np.zeros(slant.shape[0], dtype=np.float64)
    point = right_looking_ground_points(position, velocity, slant, height)
    for _ in range(12):
        _lat, _lon, got = ecef_to_geodetic(point)
        error = height - got
        if float(np.max(np.abs(error))) < 1e-3:
            break
        correction = correction + error
        point = right_looking_ground_points(position, velocity, slant, height + correction)
    return point
