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

    def seconds(self, when: np.datetime64 | np.ndarray) -> np.ndarray:
        return seconds_since(np.asarray(when), self._epoch)

    def interpolate(self, when: np.datetime64 | np.ndarray, degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate position and velocity with a polynomial of ``degree``.

        The fit uses ``degree + 1`` surrounding state vectors, so a state
        vector that is one of those nodes is reproduced up to rounding.
        """
        degree = int(degree)
        if degree < 1:
            raise ValueError("polynomial degree must be at least 1")
        needed = degree + 1
        if self.time.shape[0] < needed:
            raise ValueError(f"orbit has {self.time.shape[0]} vectors, need {needed}")
        times = np.atleast_1d(np.asarray(when, dtype="datetime64[ns]"))
        seconds = np.clip(self.seconds(times), float(self._seconds[0]), float(self._seconds[-1]))
        position = np.empty((times.shape[0], 3), dtype=np.float64)
        velocity = np.empty_like(position)
        nodes = self._seconds
        for index, second in enumerate(seconds):
            center = int(np.searchsorted(nodes, second))
            start = int(np.clip(center - needed // 2, 0, nodes.shape[0] - needed))
            sample_t = nodes[start : start + needed]
            origin = float(sample_t[needed // 2])
            abscissa = sample_t - origin
            query = float(second) - origin
            for axis in range(3):
                position[index, axis] = np.polyval(
                    np.polyfit(abscissa, self.position_m[start : start + needed, axis], degree),
                    query,
                )
                velocity[index, axis] = np.polyval(
                    np.polyfit(abscissa, self.velocity_m_s[start : start + needed, axis], degree),
                    query,
                )
        if np.asarray(when).shape == ():
            return position[0], velocity[0]
        return position, velocity

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
    right-looking direction.
    """
    sat = np.asarray(position_m, dtype=np.float64)
    vel = np.asarray(velocity_m_s, dtype=np.float64)
    radius_sat = float(np.linalg.norm(sat))
    lat = np.arcsin(np.clip(sat[2] / radius_sat, -1.0, 1.0))
    earth_radius = _local_earth_radius(sat) + float(height_m)
    cos_nadir = (radius_sat**2 + slant_range_m**2 - earth_radius**2) / (2.0 * radius_sat * slant_range_m)
    cos_nadir = float(np.clip(cos_nadir, -1.0, 1.0))
    nadir = float(np.arccos(cos_nadir))
    vhat = vel / np.linalg.norm(vel)
    down = -sat / radius_sat
    down = down - np.dot(down, vhat) * vhat
    down = down / np.linalg.norm(down)
    # Sentinel-1 is right-looking. forward × down points to the left of the
    # ground track in this frame, so the right-looking side is the opposite.
    across = np.cross(down, vhat)
    across = across / np.linalg.norm(across)
    look = np.cos(nadir) * down + np.sin(nadir) * across
    return sat + look * float(slant_range_m)


def _local_earth_radius(sat: np.ndarray) -> float:
    """Radius of the WGS84 ellipsoid along the geocentric satellite direction."""
    direction = sat / np.linalg.norm(sat)
    factor = (direction[0] / WGS84_A_M) ** 2 + (direction[1] / WGS84_A_M) ** 2 + (direction[2] / WGS84_B_M) ** 2
    return float(1.0 / np.sqrt(factor))


def ground_point_at_height(
    orbit: Orbit,
    azimuth_time: np.datetime64,
    slant_range_m: float,
    height_m: float,
    *,
    degree: int = 3,
) -> np.ndarray:
    """Geolocate one radar sample onto a surface of constant ellipsoidal height."""
    position, velocity = orbit.interpolate(azimuth_time, degree)
    correction = 0.0
    point = right_looking_ground_point(position, velocity, slant_range_m, height_m)
    for _ in range(4):
        _lat, _lon, got = ecef_to_geodetic(point)
        correction += float(height_m - got)
        point = right_looking_ground_point(position, velocity, slant_range_m, height_m + correction)
    return point
