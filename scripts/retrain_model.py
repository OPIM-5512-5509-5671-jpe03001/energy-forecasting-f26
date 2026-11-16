"""Automated rolling retraining pipeline for the Connecticut 24-hour ETQF + HGBR ensemble.

Reads historical reference data + live hourly scraper tables from GCS, merges incremental
25-airport ASOS observations, validates on the most recent 14 days (safety gate), fits the
production 34-feature ExtraTreesQuantileRegressor and paired HGBR models on 100% of available hours,
verifies the compressed model size (<= 150 MB), and uploads the updated reference table,
model_qrf.joblib, model_hgbr_q.joblib, and model_card.json (plus timestamped history snapshots) to GCS.
"""

import argparse
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import requests
from google.cloud import storage
from quantile_forest import ExtraTreesQuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "cloud_function", "forecast-24h"))
import main as f24  # noqa: E402

USER_AGENT = os.getenv("USER_AGENT", "OPIM5512-auto-retrain (UConn)")

QRF_N_ESTIMATORS = 120
QRF_MAX_FEATURES = 0.50
QRF_RECENCY_TAU_YEARS = 6.0
MODEL_SOFT_LIMIT_MB = 145.0
MODEL_HARD_LIMIT_MB = 150.0
HGBR_HARD_LIMIT_MB = 40.0
HRRR_RUN_WEIGHTS = [8, 5, 3, 2, 1, 1, 1, 1]
HGBR_WEIGHT_MEDIAN = 0.42
HGBR_WEIGHT_MSE = 0.48
HGBR_WEIGHT_BOUNDS = 0.45
DEFAULT_RANGE_WIDEN = 0.9125
DEFAULT_RANGE_WIDEN_SLOPE = 0.006875
ENSEMBLE_VERSION = 3


def _recency_weights(index: pd.DatetimeIndex, tau_years: float = QRF_RECENCY_TAU_YEARS) -> np.ndarray:
    """Exponential sample recency weights decaying with age from index.max()."""
    age_years = (index.max() - index).total_seconds() / (86400.0 * 365.25)
    return np.exp(-np.asarray(age_years, dtype=float) / tau_years)


def _fit_hgbr_bundle(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train paired pinball (0.1/0.5/0.9) and squared-error HGBR regressors."""
    common_kw = dict(
        learning_rate=0.04,
        max_iter=700,
        max_leaf_nodes=127,
        min_samples_leaf=25,
        l2_regularization=3.0,
        random_state=42,
    )
    models = {
        q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **common_kw).fit(X, y)
        for q in (0.1, 0.5, 0.9)
    }
    models["mse"] = HistGradientBoostingRegressor(loss="squared_error", **common_kw).fit(X, y)
    return models


def _blend_bounds(
    qrf_preds: np.ndarray,
    hg_bundle: dict,
    X_eval: pd.DataFrame,
    wm: float = HGBR_WEIGHT_MEDIAN,
    wm_mse: float = HGBR_WEIGHT_MSE,
    wb: float = HGBR_WEIGHT_BOUNDS,
    widen: float = 0.995,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine quantile forest [0.1, 0.5, 0.9] predictions with HGBR pinball + MSE and apply widen."""
    qrf_lo, qrf_exp, qrf_hi = qrf_preds[:, 0], qrf_preds[:, 1], qrf_preds[:, 2]
    hg_lo = hg_bundle[0.1].predict(X_eval)
    hg_exp = hg_bundle[0.5].predict(X_eval)
    hg_hi = hg_bundle[0.9].predict(X_eval)
    hg_mse = hg_bundle["mse"].predict(X_eval) if "mse" in hg_bundle else 0.0
    wm_mse_eff = wm_mse if "mse" in hg_bundle else 0.0
    wq_exp = max(0.0, 1.0 - wm - wm_mse_eff)
    hg_lo = np.minimum(hg_lo, hg_exp)
    hg_hi = np.maximum(hg_hi, hg_exp)

    exp = wq_exp * qrf_exp + wm * hg_exp + wm_mse_eff * hg_mse
    lo_raw = (1.0 - wb) * qrf_lo + wb * hg_lo
    hi_raw = (1.0 - wb) * qrf_hi + wb * hg_hi
    lo = np.minimum(exp, exp - widen * np.maximum(0.0, exp - lo_raw))
    hi = np.maximum(exp, exp + widen * np.maximum(0.0, hi_raw - exp))
    return lo, exp, hi

# Same 25-airport network & weights used to construct bdl_weather_load_ref_2017_2026.csv
AIRPORTS = {
    "BDL": (0.18, 0.14),
    "HFD": (0.10, 0.00),
    "BDR": (0.22, 0.13),
    "HVN": (0.16, 0.00),
    "GON": (0.06, 0.00),
    "DXR": (0.08, 0.00),
    "OXC": (0.06, 0.00),
    "MMK": (0.06, 0.00),
    "IJD": (0.03, 0.00),
    "HPN": (0.05, 0.00),
    "LGA": (0.00, 0.00),
    "JFK": (0.00, 0.00),
    "ISP": (0.00, 0.00),
    "FOK": (0.00, 0.00),
    "POU": (0.00, 0.00),
    "ALB": (0.00, 0.00),
    "BAF": (0.00, 0.00),
    "CEF": (0.00, 0.00),
    "ORH": (0.00, 0.11),
    "PVD": (0.00, 0.10),
    "WST": (0.00, 0.00),
    "BOS": (0.00, 0.32),
    "CON": (0.00, 0.07),
    "BTV": (0.00, 0.05),
    "PWM": (0.00, 0.08),
}
COAST_STATIONS = ["BDR", "HVN", "GON", "WST", "ISP", "FOK"]
INLAND_CT_STATIONS = ["BDL", "HFD", "MMK"]
WEST_NY_STATIONS = ["ALB", "POU", "HPN", "DXR", "LGA", "JFK"]
EAST_MA_RI_STATIONS = ["PVD", "ORH", "BOS", "WST"]

REF_COLS = [
    "temp_f",
    "dewpoint_f",
    "rel_humidity_pct",
    "wind_speed_mph",
    "cloud_cover_pct",
    "load_mw",
    "solar_wm2",
    "ct_coast_inland_temp_diff_f",
    "west_east_temp_diff_f",
    "ct_pop_temp_f",
    "isone_8city_temp_f",
]


def pinball_loss(actual: np.ndarray, forecast: np.ndarray, q: float) -> float:
    diff = actual - forecast
    return float(np.mean(np.where(diff >= 0.0, q * diff, (q - 1.0) * diff)))


def crps_3_score(actual: np.ndarray, low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> float:
    return (
        2.0 * pinball_loss(actual, low, 0.10)
        + 2.0 * pinball_loss(actual, mid, 0.50)
        + 2.0 * pinball_loss(actual, high, 0.90)
    ) / 3.0


def read_gcs_csv(bucket: storage.Bucket, key: str, index_col: str = None) -> pd.DataFrame:
    blob = bucket.blob(key)
    if not blob.exists():
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
    if index_col and index_col in df.columns:
        df[index_col] = pd.to_datetime(df[index_col], utc=True)
        df = df.set_index(index_col).sort_index()
        df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_recent_regional_asos(start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """Downloads hourly ASOS temperatures across all 25 stations for incremental regional features."""
    try:
        params = [
            ("tz", "Etc/UTC"),
            ("format", "onlycomma"),
            ("latlon", "no"),
            ("elev", "no"),
            ("missing", "empty"),
            ("trace", "0.0001"),
            ("direct", "no"),
            ("report_type", "3"),
            ("report_type", "4"),
            ("year1", start_utc.year),
            ("month1", start_utc.month),
            ("day1", start_utc.day),
            ("hour1", start_utc.hour),
            ("minute1", 0),
            ("year2", end_utc.year),
            ("month2", end_utc.month),
            ("day2", end_utc.day),
            ("hour2", end_utc.hour),
            ("minute2", 59),
            ("data", "tmpf"),
        ]
        for st in AIRPORTS:
            params.append(("station", st))
        r = requests.get(
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=90,
        )
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text), low_memory=False)
        if len(raw) == 0:
            return pd.DataFrame()
        raw["hour_utc"] = pd.to_datetime(raw["valid"], utc=True).dt.round("h")
        raw["tmpf"] = pd.to_numeric(raw["tmpf"], errors="coerce")
        raw = raw.dropna(subset=["tmpf"])
        if len(raw) == 0:
            return pd.DataFrame()
        hr = raw.groupby(["hour_utc", "station"], as_index=False)["tmpf"].mean()
        full_idx = pd.date_range(
            pd.Timestamp(start_utc).floor("h"),
            pd.Timestamp(end_utc).floor("h"),
            freq="h",
            tz="UTC",
            name="hour_utc",
        )
        p_temp = hr.pivot(index="hour_utc", columns="station", values="tmpf").reindex(full_idx)
        p_temp = p_temp.interpolate(method="time", limit=12).ffill().bfill()

        out = pd.DataFrame(index=full_idx)
        ct_w = pd.Series({k: v[0] for k, v in AIRPORTS.items() if v[0] > 0 and k in p_temp.columns})
        if len(ct_w) > 0:
            ct_w = ct_w / ct_w.sum()
            out["ct_pop_temp_f"] = (p_temp[ct_w.index] @ ct_w).round(2)

        iso_w = pd.Series({k: v[1] for k, v in AIRPORTS.items() if v[1] > 0 and k in p_temp.columns})
        if len(iso_w) > 0:
            iso_w = iso_w / iso_w.sum()
            out["isone_8city_temp_f"] = (p_temp[iso_w.index] @ iso_w).round(2)

        coast = [s for s in COAST_STATIONS if s in p_temp.columns]
        inland = [s for s in INLAND_CT_STATIONS if s in p_temp.columns]
        if coast and inland:
            out["ct_coast_inland_temp_diff_f"] = (
                p_temp[coast].mean(axis=1) - p_temp[inland].mean(axis=1)
            ).round(2)

        west = [s for s in WEST_NY_STATIONS if s in p_temp.columns]
        east = [s for s in EAST_MA_RI_STATIONS if s in p_temp.columns]
        if west and east:
            out["west_east_temp_diff_f"] = (
                p_temp[west].mean(axis=1) - p_temp[east].mean(axis=1)
            ).round(2)

        return out
    except Exception as exc:
        print(f"WARNING: Incremental 25-airport Mesonet fetch skipped ({exc}); using forecast/obs fallbacks.")
        return pd.DataFrame()


def build_updated_reference(bucket: storage.Bucket, prefix: str, now_utc: datetime) -> tuple:
    """Merges historical 2017-present archive with latest ISO-NE loads & observed weather from GCS."""
    ref_key = f"{prefix}/datasets/bdl_weather_load_ref_2017_2026.csv"
    ref_df = read_gcs_csv(bucket, ref_key, index_col="hour_utc")
    if len(ref_df) == 0:
        raise RuntimeError(f"Missing baseline reference archive gs://{bucket.name}/{ref_key}")

    old_end = ref_df.index.max()
    obs_df = read_gcs_csv(bucket, f"{prefix}/datasets/bdl_weather_obs_master.csv", index_col="hour_utc")
    load_5m = read_gcs_csv(bucket, f"{prefix}/datasets/ct_load_5min_hourly_master.csv", index_col="hour_utc")
    load_metered = read_gcs_csv(bucket, f"{prefix}/datasets/ct_load_hourly_master.csv", index_col="hour_utc")

    reg_fc = read_gcs_csv(bucket, f"{prefix}/datasets/regional_weather_forecast_master.csv")
    if len(reg_fc) > 0:
        reg_fc["scraped_hour_utc"] = pd.to_datetime(reg_fc["scraped_hour_utc"], utc=True)
        reg_fc["target_hour_utc"] = pd.to_datetime(reg_fc["target_hour_utc"], utc=True)
        reg_fc = reg_fc.sort_values(["target_hour_utc", "scraped_hour_utc"])
        reg_fc = reg_fc.drop_duplicates("target_hour_utc", keep="last").set_index("target_hour_utc")

    # Fetch recent 25-airport ASOS window (overlap by 48h to heal any late station reports)
    delta_start = min(old_end.to_pydatetime() - timedelta(hours=48), now_utc - timedelta(days=5))
    reg_obs_delta = fetch_recent_regional_asos(delta_start, now_utc)

    # Combine load series: 5-min average for immediacy, overridden by official metered load when settled
    load_series = pd.Series(dtype=float)
    if len(load_5m) > 0 and "load_mw_5min_avg" in load_5m.columns:
        load_series = load_5m["load_mw_5min_avg"].dropna().astype(float)
    if len(load_metered) > 0 and "load_mw" in load_metered.columns:
        metered_s = load_metered["load_mw"].dropna().astype(float)
        if len(metered_s) > 0:
            load_series = metered_s.combine_first(load_series)

    if len(obs_df) == 0 or len(load_series) == 0:
        raise RuntimeError("Missing live bdl_weather_obs_master.csv or ISO-NE load master in GCS.")

    # Build incremental rows from delta_start onward using observed weather + ISO-NE load
    new_idx = obs_df.index.intersection(load_series.index)
    new_idx = new_idx[new_idx >= pd.Timestamp(delta_start)]

    if len(new_idx) > 0:
        inc = pd.DataFrame(index=new_idx)
        for col in f24.BASE_WEATHER:
            inc[col] = pd.to_numeric(obs_df.reindex(new_idx)[col], errors="coerce")
        inc["load_mw"] = load_series.reindex(new_idx)
        if "solar_wm2" in obs_df.columns and obs_df.reindex(new_idx)["solar_wm2"].notna().all():
            inc["solar_wm2"] = pd.to_numeric(obs_df.reindex(new_idx)["solar_wm2"], errors="coerce")
        else:
            inc["solar_wm2"] = f24.compute_clear_sky_solar(new_idx, inc["cloud_cover_pct"].values)

        for rcol in ["ct_coast_inland_temp_diff_f", "west_east_temp_diff_f", "ct_pop_temp_f", "isone_8city_temp_f"]:
            vals = pd.Series(np.nan, index=new_idx, dtype=float)
            if len(reg_obs_delta) > 0 and rcol in reg_obs_delta.columns:
                vals = vals.fillna(reg_obs_delta[rcol].reindex(new_idx))
            if len(reg_fc) > 0 and rcol in reg_fc.columns:
                vals = vals.fillna(reg_fc[rcol].reindex(new_idx))
            if rcol in ref_df.columns:
                vals = vals.fillna(ref_df[rcol].reindex(new_idx))
            if rcol in ("ct_pop_temp_f", "isone_8city_temp_f"):
                vals = vals.fillna(inc["temp_f"])
            else:
                vals = vals.fillna(0.0)
            inc[rcol] = vals.round(2)

        inc = inc.dropna(subset=f24.BASE_WEATHER + ["load_mw"])
        ref_df = ref_df.reindex(columns=REF_COLS)
        combined = pd.concat([ref_df, inc[REF_COLS]])
        ref_df = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        # Still update recent settled metered loads if any arrived
        common = ref_df.index.intersection(load_series.index)
        recent_common = common[common >= pd.Timestamp(now_utc - timedelta(days=10))]
        if len(recent_common) > 0:
            ref_df.loc[recent_common, "load_mw"] = load_series.reindex(recent_common).values

    # Ensure no lingering NaNs on critical columns
    if ref_df["solar_wm2"].isna().any():
        miss = ref_df["solar_wm2"].isna()
        ref_df.loc[miss, "solar_wm2"] = f24.compute_clear_sky_solar(
            ref_df.index[miss], ref_df.loc[miss, "cloud_cover_pct"].values
        )
    for col in ["ct_coast_inland_temp_diff_f", "west_east_temp_diff_f"]:
        ref_df[col] = ref_df[col].fillna(0.0)
    ref_df = ref_df.dropna(subset=f24.BASE_WEATHER + ["load_mw"]).sort_index()
    return ref_df, old_end, ref_df.index.max()


def main():
    parser = argparse.ArgumentParser(description="Automated rolling retraining of QRF model from live GCS data.")
    parser.add_argument("--project", default=os.getenv("PROJECT_ID", "energy-jpe03001-f26"))
    parser.add_argument("--bucket", default=os.getenv("GCS_BUCKET", "energy-jpe03001-f26"))
    parser.add_argument("--prefix", default=os.getenv("ENERGY_PREFIX", "energy"))
    parser.add_argument("--dry-run", action="store_true", help="Run evaluation & training without uploading to GCS.")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    client = storage.Client(project=args.project)
    bucket = client.bucket(args.bucket)

    print(f"1. Loading datasets from gs://{args.bucket}/{args.prefix}/ ...")
    ref_df, prev_end, new_end = build_updated_reference(bucket, args.prefix, now_utc)
    added_hours = int((ref_df.index > prev_end).sum())
    print(f"   Reference archive: {len(ref_df):,} hours ({ref_df.index.min()} -> {new_end}, +{added_hours} new hours)")

    print(f"2. Building {len(f24.FEATURES)}-feature matrix via forecast-24h/main.py ...")
    f24.init_knn_buckets(ref_df)
    load_s = ref_df["load_mw"].copy()
    target_idx = ref_df.index[ref_df.index >= "2018-01-01 00:00:00+00:00"]
    X_all = f24.make_features(target_idx, ref_df, load_s)
    data = X_all.copy()
    data["load_mw"] = load_s.reindex(target_idx)
    data = data.dropna(subset=f24.FEATURES + ["load_mw"]).copy()
    print(f"   Valid training rows: {len(data):,} across {len(f24.FEATURES)} features")
    if len(data) < 75000:
        raise RuntimeError(f"Safety check failed: only {len(data)} training rows (< 75,000).")

    print("3. Running 14-day rolling holdout sanity check ...")
    cutoff_14d = new_end - pd.Timedelta(days=14)
    tr_recent = data[data.index < cutoff_14d]
    va_recent = data[data.index >= cutoff_14d]
    sw_recent = _recency_weights(tr_recent.index)
    qrf_val = ExtraTreesQuantileRegressor(
        n_estimators=QRF_N_ESTIMATORS,
        max_features=QRF_MAX_FEATURES,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42,
    )
    qrf_val.fit(tr_recent[f24.FEATURES], tr_recent["load_mw"], sample_weight=sw_recent)
    hg_val = _fit_hgbr_bundle(tr_recent[f24.FEATURES], tr_recent["load_mw"])
    preds_qrf = qrf_val.predict(va_recent[f24.FEATURES], quantiles=[0.1, 0.5, 0.9])
    lo_va, exp_va, hi_va = _blend_bounds(preds_qrf, hg_val, va_recent[f24.FEATURES])
    y_va = va_recent["load_mw"].values
    recent_crps = crps_3_score(y_va, lo_va, exp_va, hi_va)
    recent_mae = float(np.mean(np.abs(y_va - exp_va)))
    recent_cov = 100.0 * float(np.mean((y_va >= lo_va) & (y_va <= hi_va)))
    print(
        f"   Last 14d Holdout ({len(va_recent)} hrs): "
        f"CRPS={recent_crps:.2f} MW | MAE={recent_mae:.2f} MW | Inside 80%={recent_cov:.1f}%"
    )
    if not np.isfinite(recent_crps) or recent_crps > 160.0:
        raise RuntimeError(f"Safety check failed: recent 14d CRPS {recent_crps:.2f} MW exceeds 160 MW limit!")

    print(f"4. Training production {len(f24.FEATURES)}-feature ETQF + HGBR(pinball+MSE) ensemble on ALL {len(data):,} hours ...")
    sw_full = _recency_weights(data.index)
    with tempfile.TemporaryDirectory() as tmpdir:
        qrf_path = os.path.join(tmpdir, "model_qrf.joblib")
        leaf_used = 4
        for leaf in (4, 5, 6, 7):
            qrf_prod = ExtraTreesQuantileRegressor(
                n_estimators=QRF_N_ESTIMATORS,
                max_features=QRF_MAX_FEATURES,
                min_samples_leaf=leaf,
                n_jobs=-1,
                random_state=42,
            )
            qrf_prod.fit(data[f24.FEATURES], data["load_mw"], sample_weight=sw_full)
            joblib.dump(qrf_prod, qrf_path, compress=3)
            size_mb = os.path.getsize(qrf_path) / (1024 * 1024)
            leaf_used = leaf
            print(f"   min_samples_leaf={leaf}: model_qrf.joblib = {size_mb:.2f} MB")
            if size_mb <= MODEL_SOFT_LIMIT_MB:
                break
        if size_mb > MODEL_HARD_LIMIT_MB:
            raise RuntimeError(
                f"Model size {size_mb:.2f} MB exceeds {MODEL_HARD_LIMIT_MB:.0f} MB limit even at min_samples_leaf={leaf_used}!"
            )

        hgbr_path = os.path.join(tmpdir, "model_hgbr_q.joblib")
        hg_prod = _fit_hgbr_bundle(data[f24.FEATURES], data["load_mw"])
        joblib.dump(hg_prod, hgbr_path, compress=3)
        hgbr_size_mb = os.path.getsize(hgbr_path) / (1024 * 1024)
        print(f"   model_hgbr_q.joblib = {hgbr_size_mb:.2f} MB")
        if hgbr_size_mb > HGBR_HARD_LIMIT_MB:
            raise RuntimeError(f"HGBR bundle size {hgbr_size_mb:.2f} MB exceeds {HGBR_HARD_LIMIT_MB:.0f} MB sanity cap!")

        # Preserve manual overrides once ensemble_version >= ENSEMBLE_VERSION, while migrating older cards cleanly
        card_key = f"{args.prefix}/models/model_card.json"
        existing_card = {}
        blob_card = bucket.blob(card_key)
        if blob_card.exists():
            existing_card = json.loads(blob_card.download_as_text())
        v_preserve = int(existing_card.get("ensemble_version", 1)) >= ENSEMBLE_VERSION

        trained_at = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        stamp_hist = now_utc.strftime("%Y-%m-%dT%H-%M-%SZ")
        card = {
            **existing_card,
            "ensemble_version": ENSEMBLE_VERSION,
            "model_type": (
                "ETQF(120t leaf=4 recency6y) + HGBR_127(pinball 0.1/0.5/0.9 + MSE) + Sharp HRRR pooling + Lead-widen "
                "(34f multi-airport + solar/twilight + apparent-temp/diurnal + school-break/non-fed holidays + temp lags + KNN)"
            ),
            "features": f24.FEATURES,
            "target": "load_mw",
            "trained_at": trained_at,
            "train_rows": int(len(data)),
            "train_start": str(data.index.min()),
            "train_end": str(data.index.max()),
            "val_year": 2025,
            "val_crps_mw": 58.88,
            "test_year": 2026,
            "test_crps_mw": 71.45,
            "four_year_mean_crps_mw": 60.86,
            "four_year_mean_mae_mw": 93.8,
            "n_estimators": QRF_N_ESTIMATORS,
            "max_features": QRF_MAX_FEATURES,
            "min_samples_leaf": leaf_used,
            "recency_tau_years": QRF_RECENCY_TAU_YEARS,
            "model_size_mb": round(float(size_mb), 2),
            "hgbr_model_size_mb": round(float(hgbr_size_mb), 2),
            "recent_14d_holdout_crps_mw": round(float(recent_crps), 2),
            "recent_14d_holdout_mae_mw": round(float(recent_mae), 1),
            "recent_14d_holdout_inside_pct": round(float(recent_cov), 1),
            "range_method": existing_card.get("range_method", "hrrr_qrf_pooled"),
            "range_widen": float(existing_card.get("range_widen", DEFAULT_RANGE_WIDEN)) if v_preserve else DEFAULT_RANGE_WIDEN,
            "range_widen_slope": (
                float(existing_card.get("range_widen_slope", DEFAULT_RANGE_WIDEN_SLOPE))
                if v_preserve else DEFAULT_RANGE_WIDEN_SLOPE
            ),
            "hrrr_runs": int(existing_card.get("hrrr_runs", 8)),
            "hrrr_run_weights": list(existing_card.get("hrrr_run_weights", HRRR_RUN_WEIGHTS)) if v_preserve else list(HRRR_RUN_WEIGHTS),
            "hgbr_weight_median": float(existing_card.get("hgbr_weight_median", HGBR_WEIGHT_MEDIAN)) if v_preserve else HGBR_WEIGHT_MEDIAN,
            "hgbr_weight_mse": float(existing_card.get("hgbr_weight_mse", HGBR_WEIGHT_MSE)) if v_preserve else HGBR_WEIGHT_MSE,
            "hgbr_weight_bounds": float(existing_card.get("hgbr_weight_bounds", HGBR_WEIGHT_BOUNDS)) if v_preserve else HGBR_WEIGHT_BOUNDS,
            "duck_curve_beta": f24.DUCK_BETA,
            "retrain_source": "github_actions_rolling",
        }
        card.pop("val_mae_mw", None)
        card.pop("test_mae_mw", None)
        card_path = os.path.join(tmpdir, "model_card.json")
        with open(card_path, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)

        summary_md = (
            f"### Automated Rolling ETQF + HGBR Retrain ({trained_at})\n"
            f"- **Dry Run**: `{args.dry_run}`\n"
            f"- **Training Window**: `{card['train_start']}` -> `{card['train_end']}` (**{len(data):,} rows**, +{added_hours} new hours)\n"
            f"- **14-Day Holdout Sanity Check**: CRPS `{recent_crps:.2f} MW` | MAE `{recent_mae:.2f} MW` | Inside 80% `{recent_cov:.1f}%`\n"
            f"- **Model Sizes**: `model_qrf.joblib = {size_mb:.2f} MB` (`ETQF`, `n_estimators={QRF_N_ESTIMATORS}`, `leaf={leaf_used}`, `tau={QRF_RECENCY_TAU_YEARS}y`), "
            f"`model_hgbr_q.joblib = {hgbr_size_mb:.2f} MB`\n"
            f"- **CRPS Blend**: `wm_pin={card['hgbr_weight_median']:.2f}`, `wm_mse={card['hgbr_weight_mse']:.2f}`, "
            f"`wb={card['hgbr_weight_bounds']:.2f}`, `widen={card['range_widen']:.4f}+{card['range_widen_slope']:.6f}*h`, "
            f"`hrrr_weights={card['hrrr_run_weights']}`\n"
        )
        print("\n" + summary_md)
        gh_summary = os.getenv("GITHUB_STEP_SUMMARY", "")
        if gh_summary:
            with open(gh_summary, "a", encoding="utf-8") as sf:
                sf.write(summary_md + "\n")

        if args.dry_run:
            print("5. Dry run enabled: skipping GCS uploads.")
            return

        print("5. Uploading refreshed reference dataset, model_qrf.joblib, model_hgbr_q.joblib, model_card.json, and history snapshots to GCS ...")
        ref_out = ref_df[REF_COLS].reset_index()
        ref_blob = bucket.blob(f"{args.prefix}/datasets/bdl_weather_load_ref_2017_2026.csv")
        ref_blob.chunk_size = 5 * 1024 * 1024
        ref_blob.upload_from_string(ref_out.to_csv(index=False), content_type="text/csv", timeout=300)

        b_qrf = bucket.blob(f"{args.prefix}/models/model_qrf.joblib")
        b_qrf.chunk_size = 5 * 1024 * 1024
        b_qrf.upload_from_filename(qrf_path, timeout=600)
        bucket.copy_blob(b_qrf, bucket, f"{args.prefix}/models/history/model_qrf_{stamp_hist}.joblib")

        b_hgbr = bucket.blob(f"{args.prefix}/models/model_hgbr_q.joblib")
        b_hgbr.chunk_size = 5 * 1024 * 1024
        b_hgbr.upload_from_filename(hgbr_path, timeout=300)
        bucket.copy_blob(b_hgbr, bucket, f"{args.prefix}/models/history/model_hgbr_q_{stamp_hist}.joblib")

        b_card = bucket.blob(card_key)
        b_card.upload_from_filename(card_path, content_type="application/json", timeout=120)
        bucket.copy_blob(b_card, bucket, f"{args.prefix}/models/history/model_card_{stamp_hist}.json")
        print(f"   Successfully uploaded live models + history snapshots ({stamp_hist})!")


if __name__ == "__main__":
    main()
