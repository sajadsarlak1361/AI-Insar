"""ASF search parsing and local SLC import. No network and no secrets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai_insar.pipeline.interferogram import form_interferograms
from ai_insar.sentinel import (
    credentials_summary,
    date_from_text,
    download_sentinel1_slc,
    earthdata_credentials,
    import_slc_paths,
    parse_asf_payload,
    search_sentinel1_slc,
)

_HIT = {
    "granuleName": "S1A_IW_SLC__1SDV_20240118T141632_20240118T141659_052162_064E27_074A",
    "startTime": "2024-01-18T14:16:32Z",
    "platform": "Sentinel-1A",
    "flightDirection": "DESCENDING",
    "polarization": "VV+VH",
    "beamModeType": "IW",
    "sizeMB": 7391.5,
    "downloadUrl": "https://datapool.asf.alaska.edu/SLC/SA/example.zip",
    "processingLevel": "SLC",
}


def test_parse_asf_json_list_and_geojson():
    granules = parse_asf_payload([_HIT, {"processingLevel": "GRD", "granuleName": "skip", "downloadUrl": "https://x"}])
    assert len(granules) == 1
    assert granules[0].granule_name.startswith("S1A_IW_SLC")
    assert granules[0].size_mb == pytest.approx(7391.5)
    assert granules[0].date == np.datetime64("2024-01-18", "D")
    geo = {"features": [{"properties": {**_HIT, "sceneName": _HIT["granuleName"], "url": _HIT["downloadUrl"]}}]}
    geo["features"][0]["properties"].pop("granuleName")
    geo["features"][0]["properties"].pop("downloadUrl")
    again = parse_asf_payload(geo)
    assert again[0].download_url.endswith("example.zip")
    assert again[0].granule_name.startswith("S1A_IW_SLC")


def test_search_uses_public_endpoint_without_credentials(monkeypatch):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    seen = {}

    class _Response:
        def read(self):
            return json.dumps([_HIT]).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def opener(request, timeout=0):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        assert "Authorization" not in {key.lower(): value for key, value in request.header_items()}
        return _Response()

    granules = search_sentinel1_slc(
        west=-122.5,
        south=37.7,
        east=-122.3,
        north=37.9,
        start="2024-01-01",
        end="2024-01-20",
        max_results=1,
        urlopen=opener,
    )
    assert "processingLevel=SLC" in seen["url"]
    assert "platform=Sentinel-1" in seen["url"]
    assert granules[0].platform == "Sentinel-1A"
    with pytest.raises(ValueError, match="longitude"):
        search_sentinel1_slc(
            west=10, south=0, east=0, north=1, start="2024-01-01", end="2024-01-02", urlopen=opener
        )


def test_credentials_summary_does_not_echo_password(monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "earth-user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "super-secret-value")
    found = earthdata_credentials()
    assert found is not None
    assert found[0] == "earth-user"
    assert found[2] == "environment"
    text = credentials_summary()
    assert "earth-user" in text
    assert "super-secret-value" not in text
    assert "super-secret-value" not in json.dumps({"text": text})


def test_download_refuses_empty_login(tmp_path):
    from ai_insar.sentinel import SentinelGranule

    granule = SentinelGranule(
        granule_name="S1A_example",
        start_time="2024-01-18T00:00:00Z",
        platform="Sentinel-1A",
        flight_direction="ASCENDING",
        polarization="VV",
        beam_mode="IW",
        size_mb=10.0,
        download_url="https://example.invalid/x.zip",
    )
    with pytest.raises(RuntimeError, match="Earthdata"):
        download_sentinel1_slc([granule], tmp_path, username="", password="")


def test_download_calls_asf_search_without_logging_password(monkeypatch, tmp_path):
    from ai_insar.sentinel import SentinelGranule

    calls = {}

    class Session:
        def auth_with_creds(self, username, password):
            calls["username"] = username
            calls["password"] = password

    class Results:
        def __len__(self):
            return 1

        def download(self, path, session=None):
            calls["path"] = path
            calls["session"] = session
            (Path(path) / "S1A_example.zip").write_bytes(b"PK")

    class FakeAsf:
        ASFSession = Session

        @staticmethod
        def granule_search(granule_list):
            calls["names"] = list(granule_list)
            return Results()

    monkeypatch.setattr("ai_insar.sentinel._load_asf_search", lambda: FakeAsf)
    granule = SentinelGranule(
        granule_name="S1A_example",
        start_time="2024-01-18T00:00:00Z",
        platform="Sentinel-1A",
        flight_direction="ASCENDING",
        polarization="VV",
        beam_mode="IW",
        size_mb=10.0,
        download_url="https://example.invalid/x.zip",
    )
    saved = download_sentinel1_slc(
        [granule], tmp_path, username="earth-user", password="super-secret-value"
    )
    assert calls["username"] == "earth-user"
    assert calls["names"] == ["S1A_example"]
    assert saved == [tmp_path / "S1A_example.zip"]
    assert "super-secret-value" not in str(saved)


def test_import_npy_planes_and_run_interferogram(tmp_path):
    earlier = np.full((6, 5), 1 + 0j, dtype=np.complex64)
    later = np.full((6, 5), 0 + 1j, dtype=np.complex64)
    early_path = tmp_path / "s1a-iw1-slc-vv-20200108t000000.npy"
    late_path = tmp_path / "s1a-iw1-slc-vv-20200213t000000.npy"
    np.save(early_path, earlier)
    np.save(late_path, later)
    stack = import_slc_paths([late_path, early_path], row=1, col=1, height=4, width=3)
    assert stack.data.shape == (2, 4, 3)
    assert stack.data.dtype == np.complex64
    assert stack.dates[0] == np.datetime64("2020-01-08", "D")
    ifg = form_interferograms(stack.data, [(0, 1)])
    assert np.allclose(ifg[0], stack.data[1] * np.conj(stack.data[0]))


def test_import_safe_folder_prefers_iw1_vv(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    folder = tmp_path / "S1A_IW_SLC__1SDV_20200301T000000.SAFE" / "measurement"
    folder.mkdir(parents=True)
    base = (np.arange(12, dtype=np.float32).reshape(3, 4) + 1j).astype(np.complex64)
    vh = folder / "s1a-iw1-slc-vh-20200301t000000-001.tiff"
    vv = folder / "s1a-iw1-slc-vv-20200301t000000-001.tiff"
    iw2 = folder / "s1a-iw2-slc-vv-20200301t000000-001.tiff"
    tifffile.imwrite(vh, base * 0)
    tifffile.imwrite(vv, base)
    tifffile.imwrite(iw2, base * 5)
    second = tmp_path / "S1A_IW_SLC__1SDV_20200313T000000.SAFE" / "measurement"
    second.mkdir(parents=True)
    tifffile.imwrite(second / "s1a-iw1-slc-vv-20200313t000000-001.tiff", base + 2)
    stack = import_slc_paths([folder.parent, second.parent], row=0, col=0, height=256, width=256)
    assert stack.data.shape == (2, 3, 4)
    assert stack.dates[0] == np.datetime64("2020-03-01", "D")
    assert np.allclose(stack.data[0], base)
    assert date_from_text(vv.name) == np.datetime64("2020-03-01", "D")


def test_import_iq_tiff_crop():
    tifffile = pytest.importorskip("tifffile")
    path = Path("/tmp/ai-insar-iq-slc.tif")
    iq = np.zeros((5, 6, 2), np.int16)
    iq[..., 0] = 3
    iq[..., 1] = -4
    tifffile.imwrite(path, iq, photometric="minisblack", planarconfig="contig")
    from ai_insar.sentinel import read_tiff_window

    crop = read_tiff_window(path, row=1, col=2, height=2, width=2)
    assert crop.shape == (2, 2)
    assert crop.dtype == np.complex64
    assert np.allclose(crop, 3 - 4j)
