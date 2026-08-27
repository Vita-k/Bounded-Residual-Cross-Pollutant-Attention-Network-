# -*- coding: utf-8 -*-
"""
Sentinel-5P/TROPOMI Level-2 preprocessing for the SG-CPA-RNet air-quality study.

Purpose
-------
Download public Sentinel-5P Level-2 products for the station area and convert
irregular satellite overpasses into causal, model-ready regional atmospheric
context features.

Products used
-------------
CO  : L2__CO____ / carbonmonoxide_total_column
NO2 : L2__NO2___ / nitrogendioxide_tropospheric_column
SO2 : L2__SO2___ / sulfurdioxide_total_vertical_column

Data source
-----------
Public MEEO Sentinel-5P archive on AWS Open Data (anonymous S3 access):
    s3://meeo-s5p/RPRO/...
with OFFL fallback when RPRO is unavailable.

Scientific safeguards
---------------------
1. Sentinel-5P columns are used as regional atmospheric context, not as direct
   substitutes for surface concentrations measured at the station.
2. Product-specific QA filtering is applied before aggregation:
      CO/SO2: qa_value > 0.50
      NO2   : qa_value > 0.75
3. Only pixels inside SAT_RADIUS_KM around the station are used.
4. Alignment to the 30-min ground grid is strictly backward/causal:
   at model time t only the latest satellite observation tau <= t is available.
5. Satellite age and availability are retained explicitly.
6. Wind-aware weighting uses the transport direction (wind direction + 180°)
   because meteorological wind direction denotes where wind comes FROM.

Required packages
-----------------
pip install boto3 botocore h5py pandas numpy

Example
-------
python prepare_sentinel5p_tropomi.py \
    --ground-xlsx 2019-10-efir_1-3.xlsx \
    --out-dir results_s5p

The script writes:
    results_s5p/s5p_pixel_samples.csv
    results_s5p/s5p_overpass_features.csv
    results_s5p/s5p_aligned_30min.csv
    results_s5p/s5p_download_log.csv
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "boto3/botocore are required. Install with: pip install boto3 botocore"
    ) from exc

try:
    import h5py
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "h5py is required. Install with: pip install h5py"
    ) from exc



# -----------------------------------------------------------------------------
# Study configuration
# -----------------------------------------------------------------------------
STATION_LAT = 47.5839
STATION_LON = 34.3585
SAT_RADIUS_KM = 50.0
DISTANCE_DECAY_KM = 25.0
MAX_SAT_AGE_H = 36.0
RESAMPLE_MINUTES = 30

BUCKET = "meeo-s5p"
STREAMS = ("RPRO", "OFFL")  # historical reprocessed first, near-real-time fallback

PRODUCTS: Dict[str, Dict[str, object]] = {
    "CO": {
        "product": "L2__CO____",
        "variable": "carbonmonoxide_total_column",
        "qa_threshold": 0.50,
    },
    "NO2": {
        "product": "L2__NO2___",
        "variable": "nitrogendioxide_tropospheric_column",
        "qa_threshold": 0.75,
    },
    "SO2": {
        "product": "L2__SO2___",
        "variable": "sulfurdioxide_total_vertical_column",
        "qa_threshold": 0.50,
    },
}

# Sentinel-5P overpass is near early afternoon local solar time. This broad UTC
# window only prioritizes candidate orbit files; if no hit is found, the script
# falls back to all daily files.
CANDIDATE_UTC_HOURS = (7, 14)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def _pick_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for c in cols:
        lc = str(c).lower()
        if any(cand.lower() in lc for cand in candidates):
            return c
    return None


def load_ground_grid(path: Path) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """Load the station workbook and create the same 30-min causal time grid."""
    df = pd.read_excel(path)
    tcol = _pick_col(df.columns, ["date", "datetime", "timestamp", "time", "дата", "час"])
    if tcol is None:
        # Try first column as a datetime fallback.
        tcol = df.columns[0]
    ts = pd.to_datetime(df[tcol], errors="coerce", dayfirst=True)
    valid = ts.notna()
    df = df.loc[valid].copy()
    df.index = ts.loc[valid]
    df = df.sort_index()

    ws_col = _pick_col(df.columns, ["ws", "wind speed", "windspeed", "швидкість вітру"])
    wd_col = _pick_col(df.columns, ["wd", "wind direction", "winddirection", "напрямок вітру"])

    numeric = df.select_dtypes(include=[np.number]).columns
    grid = df[numeric].resample(f"{RESAMPLE_MINUTES}min").mean()
    return grid, ws_col if ws_col in grid.columns else None, wd_col if wd_col in grid.columns else None


def haversine_km(lat: np.ndarray, lon: np.ndarray,
                 lat0: float = STATION_LAT, lon0: float = STATION_LON) -> np.ndarray:
    r = 6371.0088
    lat1 = np.radians(lat.astype(float))
    lon1 = np.radians(lon.astype(float))
    lat2 = math.radians(lat0)
    lon2 = math.radians(lon0)
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat: np.ndarray, lon: np.ndarray,
                lat0: float = STATION_LAT, lon0: float = STATION_LON) -> np.ndarray:
    """Bearing from satellite pixel to station, degrees clockwise from north."""
    p1 = np.radians(lat.astype(float))
    p2 = math.radians(lat0)
    dl = np.radians(lon0 - lon.astype(float))
    y = np.sin(dl) * math.cos(p2)
    x = np.cos(p1) * math.sin(p2) - np.sin(p1) * math.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def angular_difference_deg(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def parse_scene_times(key: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    # Typical name contains ..._YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_...
    hits = re.findall(r"(20\d{6}T\d{6})", Path(key).name)
    if not hits:
        return None, None
    start = pd.to_datetime(hits[0], format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
    stop = pd.to_datetime(hits[1], format="%Y%m%dT%H%M%S", utc=True, errors="coerce") if len(hits) > 1 else start
    if pd.isna(start):
        return None, None
    return start.tz_convert(None), (stop.tz_convert(None) if stop is not None and not pd.isna(stop) else start.tz_convert(None))


def scene_midpoint(key: str) -> pd.Timestamp:
    a, b = parse_scene_times(key)
    if a is None:
        return pd.NaT
    if b is None:
        return a
    return a + (b - a) / 2


def _to_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2-D swath after squeeze, got shape {a.shape}")
    return a


def _read_h5_dataset(ds) -> np.ndarray:
    """Read a NetCDF4/HDF5 dataset and apply common fill/scale attributes."""
    a = np.asarray(ds[...], dtype=float)
    fill = ds.attrs.get("_FillValue", None)
    if fill is not None:
        try:
            fv = float(np.asarray(fill).reshape(-1)[0])
            a[np.isclose(a, fv, equal_nan=False)] = np.nan
        except Exception:
            pass
    missing = ds.attrs.get("missing_value", None)
    if missing is not None:
        try:
            mv = float(np.asarray(missing).reshape(-1)[0])
            a[np.isclose(a, mv, equal_nan=False)] = np.nan
        except Exception:
            pass
    scale = ds.attrs.get("scale_factor", 1.0)
    offset = ds.attrs.get("add_offset", 0.0)
    try:
        a = a * float(np.asarray(scale).reshape(-1)[0]) + float(np.asarray(offset).reshape(-1)[0])
    except Exception:
        pass
    return a


def _scanline_time_h5(product_group, valid_mask: np.ndarray, key: str) -> pd.Timestamp:
    """Best-effort AOI time; robust filename midpoint fallback."""
    try:
        if "time_utc" in product_group:
            raw = np.squeeze(product_group["time_utc"][...])
            ys = np.where(valid_mask)[0]
            if len(ys):
                scan_idx = int(np.median(ys))
                if raw.ndim == 1 and len(raw) > scan_idx:
                    val = raw[scan_idx]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="ignore")
                    t = pd.to_datetime(str(val), utc=True, errors="coerce")
                    if not pd.isna(t):
                        return t.tz_convert(None)
    except Exception:
        pass
    return scene_midpoint(key)


# -----------------------------------------------------------------------------
# S3 discovery and swath extraction
# -----------------------------------------------------------------------------
def anonymous_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_keys(s3, prefix: str) -> List[str]:
    keys: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj.get("Key", "")
            if key.endswith(".nc"):
                keys.append(key)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return sorted(keys)


def prioritize_daily_keys(keys: List[str]) -> List[str]:
    lo, hi = CANDIDATE_UTC_HOURS
    primary, other = [], []
    for k in keys:
        st, _ = parse_scene_times(k)
        if st is not None and lo <= st.hour < hi:
            primary.append(k)
        else:
            other.append(k)
    return primary + other


def extract_pixels_from_nc(
    nc_path: Path,
    key: str,
    pollutant: str,
    variable: str,
    qa_threshold: float,
    radius_km: float = SAT_RADIUS_KM,
) -> pd.DataFrame:
    with h5py.File(str(nc_path), "r") as root:
        if "PRODUCT" not in root:
            return pd.DataFrame()
        g = root["PRODUCT"]
        needed = ["latitude", "longitude", "qa_value", variable]
        if any(v not in g for v in needed):
            return pd.DataFrame()

        lat = _to_2d(_read_h5_dataset(g["latitude"]))
        lon = _to_2d(_read_h5_dataset(g["longitude"]))
        qa = _to_2d(_read_h5_dataset(g["qa_value"]))
        val = _to_2d(_read_h5_dataset(g[variable]))

        if not (lat.shape == lon.shape == qa.shape == val.shape):
            return pd.DataFrame()

        # Fast geographic prefilter before haversine distance.
        dlat = radius_km / 111.0 + 0.2
        dlon = radius_km / (111.0 * max(math.cos(math.radians(STATION_LAT)), 0.2)) + 0.2
        bbox = (
            np.isfinite(lat) & np.isfinite(lon) &
            (np.abs(lat - STATION_LAT) <= dlat) &
            (np.abs(lon - STATION_LON) <= dlon)
        )
        if not bbox.any():
            return pd.DataFrame()

        dist = np.full(lat.shape, np.nan, dtype=float)
        dist[bbox] = haversine_km(lat[bbox], lon[bbox])
        valid = (
            bbox & np.isfinite(val) & np.isfinite(qa) &
            (qa > qa_threshold) & np.isfinite(dist) & (dist <= radius_km)
        )
        if not valid.any():
            return pd.DataFrame()

        t = _scanline_time_h5(g, valid, key)
        lat_v, lon_v = lat[valid], lon[valid]
        out = pd.DataFrame({
            "timestamp": t,
            "product": pollutant,
            "value": val[valid],
            "qa": qa[valid],
            "lat": lat_v,
            "lon": lon_v,
            "distance_km": dist[valid],
            "bearing_to_station_deg": bearing_deg(lat_v, lon_v),
            "source_key": key,
        })
        return out


def fetch_pixels_for_period(
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_dir: Path,
    radius_km: float = SAT_RADIUS_KM,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pixel_csv = out_dir / "s5p_pixel_samples.csv"
    log_csv = out_dir / "s5p_download_log.csv"
    if pixel_csv.exists() and not overwrite:
        pixels = pd.read_csv(pixel_csv, parse_dates=["timestamp"])
        log = pd.read_csv(log_csv) if log_csv.exists() else pd.DataFrame()
        return pixels, log

    s3 = anonymous_s3_client()
    rows: List[pd.DataFrame] = []
    logs: List[dict] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="s5p_"))
    try:
        days = pd.date_range(start.normalize(), end.normalize(), freq="D")
        for day in days:
            for pol, cfg in PRODUCTS.items():
                product = str(cfg["product"])
                variable = str(cfg["variable"])
                qa_thr = float(cfg["qa_threshold"])
                found_for_day = False

                for stream in STREAMS:
                    prefix = f"{stream}/{product}/{day:%Y/%m/%d}/"
                    try:
                        keys = list_keys(s3, prefix)
                    except Exception as exc:
                        logs.append({
                            "date": day.date(), "product": pol, "stream": stream,
                            "key": "", "status": "list_error", "detail": str(exc)[:300]
                        })
                        continue
                    if not keys:
                        continue

                    for key in prioritize_daily_keys(keys):
                        local = tmp_dir / Path(key).name
                        try:
                            s3.download_file(BUCKET, key, str(local))
                            pix = extract_pixels_from_nc(
                                local, key, pol, variable, qa_thr, radius_km=radius_km
                            )
                            if not pix.empty:
                                rows.append(pix)
                                logs.append({
                                    "date": day.date(), "product": pol, "stream": stream,
                                    "key": key, "status": "used", "n_valid": len(pix)
                                })
                                found_for_day = True
                                # One orbit normally supplies the AOI on a given day/product.
                                break
                            else:
                                logs.append({
                                    "date": day.date(), "product": pol, "stream": stream,
                                    "key": key, "status": "no_valid_aoi", "n_valid": 0
                                })
                        except Exception as exc:
                            logs.append({
                                "date": day.date(), "product": pol, "stream": stream,
                                "key": key, "status": "read_error", "detail": str(exc)[:300]
                            })
                        finally:
                            try:
                                local.unlink(missing_ok=True)
                            except Exception:
                                pass
                    if found_for_day:
                        break
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    pixels = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["timestamp", "product", "value", "qa", "lat", "lon",
                 "distance_km", "bearing_to_station_deg", "source_key"]
    )
    if not pixels.empty:
        pixels["timestamp"] = pd.to_datetime(pixels["timestamp"], errors="coerce")
        pixels = pixels.dropna(subset=["timestamp"]).sort_values("timestamp")
    log = pd.DataFrame(logs)
    pixels.to_csv(pixel_csv, index=False)
    log.to_csv(log_csv, index=False)
    return pixels, log


# -----------------------------------------------------------------------------
# Spatial aggregation and causal alignment
# -----------------------------------------------------------------------------
def _nearest_prior_value(series: pd.Series, t: pd.Timestamp) -> float:
    s = series.loc[:t].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def aggregate_overpasses(
    pixels: pd.DataFrame,
    ground_grid: pd.DataFrame,
    ws_col: Optional[str],
    wd_col: Optional[str],
    distance_decay_km: float = DISTANCE_DECAY_KM,
) -> pd.DataFrame:
    if pixels.empty:
        return pd.DataFrame()

    px = pixels.copy()
    px["timestamp"] = pd.to_datetime(px["timestamp"])
    # Group files separately even when their timestamp is close.
    group_cols = ["product", "source_key", "timestamp"]
    out: List[dict] = []

    for (pol, key, t), g in px.groupby(group_cols, sort=True):
        v = g["value"].to_numpy(float)
        qa = g["qa"].to_numpy(float)
        d = g["distance_km"].to_numpy(float)
        bear = g["bearing_to_station_deg"].to_numpy(float)

        eps = 1e-6
        w_dist = qa * np.exp(-d / max(distance_decay_km, eps))
        dist_mean = float(np.average(v, weights=w_dist)) if np.sum(w_dist) > 0 else np.nan

        ws = _nearest_prior_value(ground_grid[ws_col], t) if ws_col else np.nan
        wd = _nearest_prior_value(ground_grid[wd_col], t) if wd_col else np.nan
        wind_mean = np.nan
        if np.isfinite(wd):
            transport_dir = (wd + 180.0) % 360.0
            diff = angular_difference_deg(bear, transport_dir)
            align = np.maximum(np.cos(np.radians(diff)), 0.0)
            # Keep a non-zero floor so the estimator is stable when no pixel is
            # perfectly upwind; aligned pixels still receive substantially more weight.
            w_wind = w_dist * (0.25 + 0.75 * align)
            if np.sum(w_wind) > 0:
                wind_mean = float(np.average(v, weights=w_wind))

        idx_near = int(np.nanargmin(d))
        out.append({
            "timestamp": pd.Timestamp(t),
            "product": pol,
            "mean": float(np.nanmean(v)),
            "median": float(np.nanmedian(v)),
            "std": float(np.nanstd(v)),
            "nearest": float(v[idx_near]),
            "distance_weighted": dist_mean,
            "wind_weighted": wind_mean,
            "n_valid": int(np.isfinite(v).sum()),
            "mean_qa": float(np.nanmean(qa)),
            "mean_distance_km": float(np.nanmean(d)),
            "wind_speed": ws,
            "wind_direction": wd,
            "source_key": key,
        })

    return pd.DataFrame(out).sort_values(["timestamp", "product"])


def align_overpasses_to_grid(
    overpasses: pd.DataFrame,
    grid_index: pd.DatetimeIndex,
    max_age_h: float = MAX_SAT_AGE_H,
) -> pd.DataFrame:
    base = pd.DataFrame({"timestamp": pd.DatetimeIndex(grid_index).sort_values()})
    result = base.copy()

    for pol in PRODUCTS:
        op = overpasses.loc[overpasses["product"] == pol].copy() if not overpasses.empty else pd.DataFrame()
        prefix = f"S5P_{pol}_"
        if op.empty:
            for c in ["mean", "median", "nearest", "distance_weighted", "wind_weighted",
                      "std", "n_valid", "mean_qa", "mean_distance_km", "age_h", "available"]:
                result[prefix + c] = np.nan if c != "available" else 0.0
            continue

        op = op.sort_values("timestamp")
        keep = ["timestamp", "mean", "median", "nearest", "distance_weighted", "wind_weighted",
                "std", "n_valid", "mean_qa", "mean_distance_km"]
        op2 = op[keep].copy()
        op2["satellite_time"] = op2["timestamp"]
        merged = pd.merge_asof(
            base.sort_values("timestamp"),
            op2.sort_values("timestamp"),
            on="timestamp", direction="backward", allow_exact_matches=True
        )
        age_h = (merged["timestamp"] - merged["satellite_time"]).dt.total_seconds() / 3600.0
        available = merged["satellite_time"].notna() & (age_h >= 0) & (age_h <= max_age_h)

        for c in keep[1:]:
            vals = merged[c].where(available, np.nan)
            result[prefix + c] = vals.to_numpy()
        result[prefix + "age_h"] = age_h.where(available, np.nan).to_numpy()
        result[prefix + "available"] = available.astype(float).to_numpy()

    result = result.set_index("timestamp")
    return result


def prepare_all(
    ground_xlsx: Path,
    out_dir: Path,
    overwrite: bool = False,
    radius_km: float = SAT_RADIUS_KM,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid, ws_col, wd_col = load_ground_grid(ground_xlsx)
    if grid.empty:
        raise RuntimeError("Ground workbook produced an empty time grid.")

    pixels, log = fetch_pixels_for_period(
        grid.index.min(), grid.index.max(), out_dir, radius_km=radius_km, overwrite=overwrite
    )
    overpasses = aggregate_overpasses(pixels, grid, ws_col, wd_col)
    aligned = align_overpasses_to_grid(overpasses, grid.index)

    overpasses.to_csv(out_dir / "s5p_overpass_features.csv", index=False)
    aligned.to_csv(out_dir / "s5p_aligned_30min.csv", index=True)

    # Compact QC summary for the paper/reviewer response.
    qc_rows = []
    for pol in PRODUCTS:
        p = pixels[pixels["product"] == pol] if not pixels.empty else pd.DataFrame()
        o = overpasses[overpasses["product"] == pol] if not overpasses.empty else pd.DataFrame()
        avail_col = f"S5P_{pol}_available"
        qc_rows.append({
            "product": pol,
            "valid_pixels": int(len(p)),
            "usable_overpasses": int(len(o)),
            "days_with_data": int(o["timestamp"].dt.date.nunique()) if len(o) else 0,
            "grid_availability_fraction": float(aligned[avail_col].mean()) if avail_col in aligned else 0.0,
            "median_age_h_when_available": float(aligned.loc[aligned[avail_col] > 0, f"S5P_{pol}_age_h"].median())
            if avail_col in aligned and (aligned[avail_col] > 0).any() else np.nan,
        })
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(out_dir / "s5p_qc_summary.csv", index=False)
    return pixels, overpasses, aligned, qc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-xlsx", type=Path, default=Path("2019-10-efir_1-3.xlsx"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_s5p"))
    ap.add_argument("--radius-km", type=float, default=SAT_RADIUS_KM)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.ground_xlsx.exists():
        raise FileNotFoundError(
            f"Ground workbook not found: {args.ground_xlsx}. "
            "Place 2019-10-efir_1-3.xlsx next to the script or pass --ground-xlsx."
        )

    pixels, overpasses, aligned, qc = prepare_all(
        args.ground_xlsx, args.out_dir, overwrite=args.overwrite, radius_km=args.radius_km
    )
    print("\nSentinel-5P preprocessing completed")
    print(f"  valid pixel samples : {len(pixels):,}")
    print(f"  usable overpasses   : {len(overpasses):,}")
    print(f"  aligned grid rows   : {len(aligned):,}")
    print("\nQC summary:")
    print(qc.to_string(index=False))
    print(f"\nOutputs: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
