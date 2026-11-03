"""Sentinel-1 SAFE open: folder, manifest.safe, and zip, without SNAP code."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tifffile")

from ai_insar.conventions import C_BAND_WAVELENGTH_M, SPEED_OF_LIGHT_M_S
from ai_insar.safe import (
    format_product_status,
    measurement_storage,
    preview_amplitude,
    read_annotation,
    read_window,
    resolve_safe,
    select_measurement,
)
from ai_insar.sentinel import discover_slc_rasters, import_slc_paths

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "safe"
SAFE_NAME = "S1B_IW_SLC__1SDV_20210401T052622_20210401T052650_026269_032297_EFA4"
SAFE = FIXTURE / f"{SAFE_NAME}.SAFE"
ZIP = FIXTURE / f"{SAFE_NAME}.zip"
RADAR_HZ = 5.405000454334350e9


def _identity(product):
    return (
        product.product_id,
        product.platform,
        product.mode,
        product.product_type,
        product.polarisations,
        product.swaths,
        product.start_utc,
        product.stop_utc,
        product.absolute_orbit,
        product.relative_orbit,
        product.orbit_pass,
        product.datatake_id,
        product.footprint,
        tuple((item.href, item.annotation_href, item.swath, item.polarisation) for item in product.measurements),
        product.quicklook_href,
    )


def test_folder_manifest_and_zip_resolve_to_the_same_product():
    folder = resolve_safe(SAFE)
    manifest = resolve_safe(SAFE / "manifest.safe")
    archive = resolve_safe(ZIP)
    assert _identity(folder) == _identity(manifest) == _identity(archive)
    assert folder.root == manifest.root == SAFE.resolve()
    assert not folder.is_zip and not manifest.is_zip
    assert archive.is_zip and archive.root == ZIP.resolve()
    assert folder.platform == "S1B"
    assert folder.mode == "IW"
    assert folder.product_type == "SLC"
    assert folder.polarisations == ("VV", "VH")
    assert folder.swaths == ("IW1", "IW2", "IW3")
    assert folder.start_utc.startswith("2021-04-01T05:26:22")
    assert folder.stop_utc.startswith("2021-04-01T05:26:50")
    assert folder.absolute_orbit == 26269
    assert folder.relative_orbit == 168
    assert folder.orbit_pass == "DESCENDING"
    assert folder.datatake_id == "205463"
    assert folder.footprint[0] == pytest.approx((45.526531, 11.986685))
    assert len(folder.footprint) == 4
    bands = {(item.swath, item.polarisation) for item in folder.measurements}
    assert bands == {("IW1", "VV"), ("IW1", "VH"), ("IW2", "VH")}
    assert folder.level == "L1"
    assert archive.archive_prefix == f"{SAFE_NAME}.SAFE/"
    assert folder.measurements[0].annotation_ready
    assert select_measurement(folder, "IW1", "VV").lines == 13509
    assert select_measurement(folder, "IW1", "VV").samples == 21632
    assert "IW1–3" in format_product_status(folder)
    assert "Orbit 26269 (168)" in format_product_status(folder)


def test_annotation_and_complex64_window_match_across_containers():
    folder = resolve_safe(SAFE)
    archive = resolve_safe(ZIP)
    meas = select_measurement(folder, "IW1", "VV")
    zip_meas = select_measurement(archive, "IW1", "VV")
    read_annotation(folder, meas)
    assert meas.lines == 13509
    assert meas.samples == 21632
    assert len(meas.bursts) == 9
    assert meas.bursts[0].azimuth_time.startswith("2021-04-01T05:26:24")
    assert meas.azimuth_time_interval == pytest.approx(2.055556299999998e-3)
    assert meas.range_pixel_spacing == pytest.approx(2.329562)
    assert meas.radar_frequency == pytest.approx(RADAR_HZ)
    assert meas.slant_range_time == pytest.approx(5.343035814454385e-3)
    assert meas.incidence == pytest.approx(33.87494380774521)
    window = read_window(folder, meas, row=1, col=2, height=2, width=2)
    zipped = read_window(archive, zip_meas, row=1, col=2, height=2, width=2)
    assert window.shape == (2, 2)
    assert window.dtype == np.complex64
    expected = np.array([[2 + 5j, 2 + 6j], [3 + 5j, 3 + 6j]], dtype=np.complex64)
    assert np.allclose(window, expected)
    assert np.allclose(zipped, expected)
    _path, offset, size = measurement_storage(archive, zip_meas)
    assert offset is not None and size is not None and offset > 0


def test_stored_zip_window_does_not_extract(monkeypatch):
    def fail_extract(*_args, **_kwargs):
        raise AssertionError("STORED zip entry was extracted")

    monkeypatch.setattr("ai_insar.safe._cache_zip_member", fail_extract)
    product = resolve_safe(ZIP)
    window = read_window(product, select_measurement(product, "IW1", "VV"), 0, 0, 1, 1)
    assert window.dtype == np.complex64
    assert np.allclose(window, np.array([[1 + 3j]], dtype=np.complex64))


def test_deflated_zip_extracts_one_tiff(tmp_path: Path):
    dest = tmp_path / f"{SAFE_NAME}.zip"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in SAFE.rglob("*"):
            if path.is_file():
                archive.write(path, f"{SAFE_NAME}.SAFE/{path.relative_to(SAFE).as_posix()}")
    product = resolve_safe(dest)
    measurement = select_measurement(product, "IW2", "VH")
    _path, offset, _size = measurement_storage(product, measurement)
    assert offset is None
    window = read_window(product, measurement, 0, 0, 1, 1)
    assert np.allclose(window, np.array([[0 + 7j]], dtype=np.complex64))


def test_preview_uses_quicklook_or_decimated_amplitude():
    product = resolve_safe(SAFE)
    quick = preview_amplitude(product)
    assert quick.shape == (5, 7)
    assert quick.dtype == np.float32
    assert np.allclose(quick, np.arange(35, dtype=np.float32).reshape(5, 7))
    band = preview_amplitude(product, select_measurement(product, "IW1", "VH"))
    assert band.shape == (8, 6)
    assert band[0, 0] == pytest.approx(abs(10 + 0j))


def test_import_uses_manifest_time_and_radar_frequency():
    stack = import_slc_paths([ZIP], row=0, col=0, height=2, width=3)
    assert stack.data.shape == (1, 2, 3)
    assert stack.data.dtype == np.complex64
    assert stack.dates[0] == np.datetime64("2021-04-01", "D")
    expected = SPEED_OF_LIGHT_M_S / RADAR_HZ
    assert stack.wavelength_m == pytest.approx(expected, rel=1e-12)
    assert stack.wavelength_m != pytest.approx(C_BAND_WAVELENGTH_M, abs=1e-12)
    assert "IW1 VV" in stack.description
    raster = discover_slc_rasters([SAFE])
    assert raster == [SAFE / "measurement" / "s1b-iw1-slc-vv-20210401t052624-20210401t052649-026269-032297-004.tiff"]


def test_open_does_not_read_tiff_headers(monkeypatch):
    import tifffile

    calls: list[object] = []
    real = tifffile.TiffFile

    def wrapped(*args, **kwargs):
        calls.append(args[:1])
        return real(*args, **kwargs)

    monkeypatch.setattr(tifffile, "TiffFile", wrapped)
    product = resolve_safe(ZIP)
    assert calls == []
    assert product.level == "L1"
    assert select_measurement(product, "IW1", "VV").samples == 21632


def test_subsampled_window_and_packed_int32():
    product = resolve_safe(SAFE)
    measurement = select_measurement(product, "IW1", "VV")
    full = read_window(product, measurement, 0, 0, 8, 6)
    stepped = read_window(product, measurement, 0, 0, 8, 6, y_step=2, x_step=2)
    assert stepped.dtype == np.complex64
    assert np.allclose(stepped, full[::2, ::2])
    from ai_insar.sentinel import to_complex64

    real = np.array([[1, -2], [4, 5]], dtype=np.int16)
    imag = np.array([[3, -4], [6, -7]], dtype=np.int16)
    packed = (imag.astype(np.int32) << np.int32(16)) | (real.astype(np.int32) & np.int32(0xFFFF))
    assert np.allclose(to_complex64(packed), real.astype(np.float32) + 1j * imag.astype(np.float32))


def test_reject_etad_cog_level_and_incomplete_products(tmp_path: Path):
    etad = tmp_path / "S1A_IW_ETA__1SDH_20200101T000000_20200101T000030_000001_00000B_AAAA.SAFE"
    etad.mkdir()
    (etad / "manifest.safe").write_text("<xfdu/>", encoding="utf-8")
    with pytest.raises(ValueError, match="ETAD"):
        resolve_safe(etad)
    two = tmp_path / "s1_two.zip"
    with zipfile.ZipFile(two, "w") as archive:
        archive.writestr("A.SAFE/manifest.safe", "<a/>")
        archive.writestr("B.SAFE/manifest.safe", "<b/>")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_safe(two)
    plain = tmp_path / "download.zip"
    with zipfile.ZipFile(plain, "w") as archive:
        archive.writestr(f"{SAFE_NAME}.SAFE/manifest.safe", "<a/>")
    with pytest.raises(ValueError, match="start with s1"):
        resolve_safe(plain)
    cog = tmp_path / "S1A_IW_SLC__1SDV_20200101T000000_20200101T000030_000001_00000B_AAAA_COG.zip"
    cog.write_bytes(b"PK")
    with pytest.raises(ValueError, match="COG"):
        resolve_safe(cog)
    level2 = tmp_path / "s1_ocn.zip"
    with zipfile.ZipFile(level2, "w") as archive:
        archive.writestr("X.SAFE/manifest.safe", "<a/>")
        archive.writestr("X.SAFE/measurement/ocn.nc", b"netcdf")
    with pytest.raises(ValueError, match="Level-2"):
        resolve_safe(level2)
    bare = tmp_path / SAFE.name
    bare.mkdir()
    shutil.copy(SAFE / "manifest.safe", bare / "manifest.safe")
    with pytest.raises(ValueError, match="incomplete"):
        resolve_safe(bare)


def test_namespace_agnostic_manifest_date_beats_the_filename(tmp_path: Path):
    import tifffile

    product_id = "S1C_IW_SLC__1SDV_20200505T000000_20200505T000030_000010_00000A_ABCD"
    folder = tmp_path / f"{product_id}.SAFE"
    (folder / "measurement").mkdir(parents=True)
    (folder / "annotation").mkdir()
    href = "s1c-iw1-slc-vv-20200505t000000-001"
    array = np.full((4, 4), 2 - 1j, dtype=np.complex64)
    tifffile.imwrite(folder / "measurement" / f"{href}.tiff", array)
    (folder / "annotation" / f"{href}.xml").write_text(
        """<product xmlns="urn:example:s1c-ipf">
          <adsHeader><swath>IW1</swath><polarisation>VV</polarisation></adsHeader>
          <generalAnnotation><productInformation>
            <radarFrequency>5.400e9</radarFrequency>
          </productInformation></generalAnnotation>
          <imageAnnotation><imageInformation>
            <numberOfSamples>4</numberOfSamples>
            <numberOfLines>4</numberOfLines>
            <azimuthTimeInterval>0.002</azimuthTimeInterval>
            <rangePixelSpacing>2.3</rangePixelSpacing>
            <slantRangeTime>0.006</slantRangeTime>
            <incidenceAngleMidSwath>36</incidenceAngleMidSwath>
          </imageInformation></imageAnnotation>
          <swathTiming><burstList count="1">
            <burst><azimuthTime>2020-06-15T12:00:00.000000</azimuthTime></burst>
          </burstList></swathTiming>
        </product>""",
        encoding="utf-8",
    )
    (folder / "manifest.safe").write_text(
        f"""<xfdu:XFDU xmlns:xfdu="urn:example:s1c-ipf" xmlns:safe="urn:example:s1c-safe">
          <informationPackageMap>
            <xfdu:contentUnit unitType="SAFE Archive Information Package">
              <xfdu:contentUnit unitType="Measurement Data Unit" dmdID="ann1">
                <dataObjectPointer dataObjectID="meas1"/>
              </xfdu:contentUnit>
            </xfdu:contentUnit>
          </informationPackageMap>
          <metadataSection>
            <metadataObject ID="ann1"><dataObjectPointer dataObjectID="annfile"/></metadataObject>
            <metadataObject ID="platform"><metadataWrap><xmlData>
              <safe:platform>
                <safe:familyName>SENTINEL-1</safe:familyName>
                <safe:number>C</safe:number>
                <safe:instrument><safe:extension><instrumentMode>
                  <mode>IW</mode><swath>IW1</swath>
                </instrumentMode></safe:extension></safe:instrument>
              </safe:platform>
            </xmlData></metadataWrap></metadataObject>
            <metadataObject ID="generalProductInformation"><metadataWrap><xmlData>
              <standAloneProductInformation>
                <productType>SLC</productType>
                <transmitterReceiverPolarisation>VV</transmitterReceiverPolarisation>
                <missionDataTakeID>10</missionDataTakeID>
              </standAloneProductInformation>
            </xmlData></metadataWrap></metadataObject>
            <metadataObject ID="acquisitionPeriod"><metadataWrap><xmlData>
              <acquisitionPeriod>
                <startTime>2020-06-15T12:00:00.000000</startTime>
                <stopTime>2020-06-15T12:00:30.000000</stopTime>
              </acquisitionPeriod>
            </xmlData></metadataWrap></metadataObject>
            <metadataObject ID="measurementOrbitReference"><metadataWrap><xmlData>
              <orbitReference>
                <orbitNumber type="start">10</orbitNumber>
                <relativeOrbitNumber type="start">7</relativeOrbitNumber>
                <pass>ASCENDING</pass>
              </orbitReference>
            </xmlData></metadataWrap></metadataObject>
            <metadataObject ID="measurementFrameSet"><metadataWrap><xmlData>
              <frameSet><frame><footPrint><coordinates>1,2 3,4</coordinates></footPrint></frame></frameSet>
            </xmlData></metadataWrap></metadataObject>
          </metadataSection>
          <dataObjectSection>
            <dataObject ID="meas1" repID="s1Level1MeasurementSchema">
              <byteStream><fileLocation href="./measurement/{href}.tiff"/></byteStream>
            </dataObject>
            <dataObject ID="annfile" repID="s1Level1ProductSchema">
              <byteStream><fileLocation href="./annotation/{href}.xml"/></byteStream>
            </dataObject>
          </dataObjectSection>
        </xfdu:XFDU>""",
        encoding="utf-8",
    )
    product = resolve_safe(folder)
    assert product.platform == "S1C"
    assert product.orbit_pass == "ASCENDING"
    measurement = select_measurement(product, "IW1", "VV")
    read_annotation(product, measurement)
    assert measurement.lines == 4
    assert len(measurement.bursts) == 1
    stack = import_slc_paths([folder / "manifest.safe"], row=0, col=0, height=4, width=4)
    assert stack.dates[0] == np.datetime64("2020-06-15", "D")
    assert stack.wavelength_m == pytest.approx(SPEED_OF_LIGHT_M_S / 5.4e9, rel=1e-12)
    assert np.allclose(stack.data[0], array)


def test_filename_band_is_explicit_not_a_vv_substring(tmp_path: Path):
    import tifffile

    folder = tmp_path / "loose.SAFE" / "measurement"
    folder.mkdir(parents=True)
    base = np.ones((3, 3), dtype=np.complex64)
    tifffile.imwrite(folder / "avv-notes.tif", base * 9)
    tifffile.imwrite(folder / "s1a-iw2-slc-vv-20200101t000000-001.tiff", base)
    with pytest.raises(ValueError, match="IW1 VV"):
        import_slc_paths([folder.parent])
    stack = import_slc_paths([folder.parent], swath="IW2", pol="VV", row=0, col=0, height=3, width=3)
    assert np.allclose(stack.data[0], base)
