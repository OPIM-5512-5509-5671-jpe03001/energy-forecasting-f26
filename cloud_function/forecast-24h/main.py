# forecast-24h: YOUR model forecasts Connecticut electricity load for the next 24 hours, every hour.
#
# Every hour (:20, after the weather scrapers at :05 and the ISO-NE scraper at :15) this function:
#   1. loads your live model:          gs://<bucket>/energy/models/model.joblib (+ model_card.json)
#   2. picks the next 24 FULL hours:   if it runs at 2:20 p.m., it forecasts 3 p.m. today ... 2 p.m. tomorrow
#   3. builds the features with make_features(), the SAME code as the Lab 5 / Lab 6 notebook:
#        weather     -> the newest NWS + 25-airport multi-model forecast for each hour
#                       (bdl_weather_forecast_master.csv + regional_weather_forecast_master.csv)
#        duck curve  -> solar energy entering the CT grid (est_solar_gen_mw), observed vs. forecasted cloud cover
#                       (cloud_cover_obs_24h_pct, cloud_cover_diff_24h_pct), and 24h solar MW differential (solar_gen_diff_24h_mw)
#        load        -> Duck-Curve-adjusted 1D, 7D, 1M (same DOW), and 1Y (same DOW) load lags + KNN weather-analog load
#   4. adds a low and high value (an 80% range: 10th and 90th percentiles) if your model card says how (Lab 6):
#        "range_method": "quantile_models"  -> two more models, energy/models/model_low.joblib and model_high.joblib
#        "range_method": "offsets_by_hour"  -> "range_offsets_mw": {"low": [24 MW], "high": [24 MW]}, by Connecticut hour
#        "range_method": "hrrr_qrf_pooled"  -> Lab 6C: energy/models/model_qrf.joblib (a quantile forest) on EVERY recent HRRR
#                                              run that covers each hour (weather uncertainty), pooled into one low,
#                                              expected and high. HRRR runs come from Dr. Dave's class file (HRRR_RUNS_URL);
#                                              if it is missing or stale, the NWS/multi-model forecast is used instead
#        no "range_method" (Lab 5)          -> expected value only (the low and high columns stay empty)
#   5. writes 24 rows (target_hour_utc, predicted_low_mw, predicted_load_mw, predicted_high_mw, ...):
#        gs://<bucket>/energy/forecasts/latest.csv                       <- replaced every hour (Lab 6 reads this one)
#        gs://<bucket>/energy/forecasts/history/<issued time>.csv        <- one file per hour, never overwritten
#        gs://<bucket>/energy/datasets/forecasts_master.csv              <- every forecast you ever made, in one table
# No model yet? It politely does nothing ("status": "no_model"). No ISO-NE data yet? "status": "no_load_data".
# HTTP entrypoint: forecast_24h_http

import io
import json
import logging
import os
import traceback
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from google.cloud import storage
from pandas.tseries.holiday import USFederalHolidayCalendar

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
ENERGY_PREFIX = os.getenv("ENERGY_PREFIX", "energy")
HOURS_AHEAD = int(os.getenv("HOURS_AHEAD", "24"))

MODEL_KEY = f"{ENERGY_PREFIX}/models/model.joblib"
CARD_KEY = f"{ENERGY_PREFIX}/models/model_card.json"
LOW_MODEL_KEY = f"{ENERGY_PREFIX}/models/model_low.joblib"
HIGH_MODEL_KEY = f"{ENERGY_PREFIX}/models/model_high.joblib"
QRF_MODEL_KEY = f"{ENERGY_PREFIX}/models/model_qrf.joblib"
# Lab 6C: the class HRRR file, rewritten every hour at :05 by Dr. Dave's hrrr-bdl job (8 newest runs, 0-48 hours ahead)
HRRR_RUNS_URL = os.getenv("HRRR_RUNS_URL", "https://storage.googleapis.com/opim5512-leaderboard-f26/weather/bdl_hrrr_runs.csv")
HRRR_MAX_AGE_HOURS = 14        # newest run older than this = something is wrong upstream: use the NWS forecast instead
# 20 percentiles, each the middle of a 5% slice of the chances (2.5%, 7.5%, ..., 97.5%), same as the Lab 6C notebook
POOL_QUANTILES = [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, 0.675, 0.725,
                  0.775, 0.825, 0.875, 0.925, 0.975]
WEATHER_KEY = f"{ENERGY_PREFIX}/datasets/bdl_weather_forecast_master.csv"
REGIONAL_WEATHER_KEY = f"{ENERGY_PREFIX}/datasets/regional_weather_forecast_master.csv"
OBS_WEATHER_KEY = f"{ENERGY_PREFIX}/datasets/bdl_weather_obs_master.csv"
KNN_REF_KEY = f"{ENERGY_PREFIX}/datasets/bdl_weather_load_ref_2017_2026.csv"
LOAD_KEY = f"{ENERGY_PREFIX}/datasets/ct_load_5min_hourly_master.csv"
LATEST_KEY = f"{ENERGY_PREFIX}/forecasts/latest.csv"
HISTORY_PREFIX = f"{ENERGY_PREFIX}/forecasts/history"
MASTER_KEY = f"{ENERGY_PREFIX}/datasets/forecasts_master.csv"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

# ----------------------------------------------------------------------------------------------------------------
# make_features: copied word for word from the Lab 5 / Lab 6 notebook. Change it there, change it here.
# ----------------------------------------------------------------------------------------------------------------
BASE_WEATHER = ["temp_f", "dewpoint_f", "rel_humidity_pct", "wind_speed_mph", "cloud_cover_pct"]
WEATHER = ["temp_f", "dewpoint_f", "rel_humidity_pct", "wind_speed_mph", "cloud_cover_pct",
           "solar_wm2", "ct_coast_inland_temp_diff_f"]
TEMP_LAG_FEATURES = ["temp_lag_3h_f", "temp_lag_24h_f", "temp_mean_24h_f", "dewpoint_mean_24h_f"]
DUCK_CURVE_FEATURES = ["est_solar_gen_mw", "cloud_cover_obs_24h_pct", "cloud_cover_diff_24h_pct", "solar_gen_diff_24h_mw"]
KNN_EXTRA_COLS = ["west_east_temp_diff_f"]
CALENDAR = ["hour", "day_of_week", "month", "is_weekend", "is_holiday", "is_non_fed_holiday"]
LOAD_LAGS = ["load_yesterday_mw", "load_7d_mw", "load_1m_dow_mw", "load_1y_dow_mw"]
FEATURES = CALENDAR + WEATHER + TEMP_LAG_FEATURES + DUCK_CURVE_FEATURES + LOAD_LAGS + ["knn_weather_load_mw"]
DUCK_BETA = 0.55
HOLIDAYS = USFederalHolidayCalendar().holidays(start="2015-01-01", end="2030-12-31")


def _easter_sunday(year: int) -> date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _build_non_fed_holidays(start_year: int = 2015, end_year: int = 2031) -> set:
    """Connecticut commercial/school non-federal holiday weekdays with significant grid load drops."""
    fed = set(pd.DatetimeIndex(HOLIDAYS).date)
    dates = set()
    for y in range(start_year, end_year):
        nov1 = date(y, 11, 1)
        thanksgiving = nov1 + timedelta(days=((3 - nov1.weekday()) % 7) + 21)
        election_day = nov1 + timedelta(days=((0 - nov1.weekday()) % 7) + 1)
        good_friday = _easter_sunday(y) - timedelta(days=2)
        black_friday = thanksgiving + timedelta(days=1)
        for d in (good_friday, election_day, black_friday, date(y, 12, 24), date(y, 12, 31)):
            if d.weekday() < 5 and d not in fed:
                dates.add(d)
    return dates


NON_FED_HOLIDAYS = _build_non_fed_holidays()

# Empirical mean, std, and feature weights for 10-variable + seasonal KNN weather-analog matcher:
# [temp_f, dewpoint_f, rel_humidity_pct, wind_speed_mph, cloud_cover_pct, cdd, hdd,
#  solar_wm2, ct_coast_inland_temp_diff_f, west_east_temp_diff_f, sin_doy, cos_doy]
W_MEAN = np.array([52.75, 40.53, 66.46, 7.59, 61.98, 3.02, 12.03, 156.84, 0.71, 1.47], dtype=np.float32)
W_STD = np.array([18.92, 19.37, 20.49, 5.20, 35.00, 5.91, 13.54, 232.07, 3.79, 3.09], dtype=np.float32)
WEIGHTS = np.array([2.5, 1.5, 0.6, 0.7, 0.8, 2.0, 1.5, 0.9, 0.5, 0.4, 1.2, 1.2], dtype=np.float32)
BUCKETS = {}
LATEST_REGIONAL = None
OBS_CLOUD_SERIES = pd.Series(dtype=float)
OBS_SOLGEN_SERIES = pd.Series(dtype=float)
OBS_TEMP_SERIES = pd.Series(dtype=float)
OBS_DEWP_SERIES = pd.Series(dtype=float)


def compute_clear_sky_solar(hours_utc: pd.DatetimeIndex, cloud_pct: np.ndarray) -> np.ndarray:
    """Physical clear-sky solar irradiance at BDL (41.9389N, -72.6832W) attenuated by cloud cover."""
    hours_utc = pd.DatetimeIndex(hours_utc)
    lat_rad = np.deg2rad(41.9389)
    lon_deg = -72.6832
    doy = hours_utc.dayofyear.values
    hr_utc = hours_utc.hour.values - 0.5
    gamma = 2.0 * np.pi / 365.25 * (doy - 1 + (hr_utc - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    time_offset = eqtime + 4.0 * lon_deg
    tst = hr_utc * 60.0 + time_offset
    ha = np.deg2rad((tst / 4.0) - 180.0)
    sin_el = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    sin_el = np.maximum(0.0, sin_el)
    clear_sky = 1020.0 * (sin_el ** 1.15)
    cf = np.clip(np.nan_to_num(np.asarray(cloud_pct, dtype=float), nan=50.0) / 100.0, 0.0, 1.0)
    atten = 1.0 - 0.68 * (cf ** 2.0)
    return np.round(clear_sky * atten, 1)


def compute_solar_geometry_and_gen(hours_utc: pd.DatetimeIndex, solar_wm2: np.ndarray,
                                   cloud_pct: np.ndarray, temp_f: np.ndarray) -> tuple:
    """Computes (sin_sun_elev, est_solar_gen_mw) representing the estimated MW of solar PV
    entering the Connecticut grid during daytime hours, accounting for CT BTM PV capacity growth (2017->2026),
    shortwave irradiance (W/m2), cloud cover attenuation, and panel temperature derating."""
    hours_utc = pd.DatetimeIndex(hours_utc)
    lat_rad = np.deg2rad(41.9389)
    lon_deg = -72.6832
    doy = hours_utc.dayofyear.values
    hr_utc = hours_utc.hour.values - 0.5
    gamma = 2.0 * np.pi / 365.25 * (doy - 1 + (hr_utc - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    time_offset = eqtime + 4.0 * lon_deg
    tst = hr_utc * 60.0 + time_offset
    ha = np.deg2rad((tst / 4.0) - 180.0)
    sin_el = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    sin_el = np.maximum(0.0, sin_el)

    frac_year = hours_utc.year.values + (doy - 1.0) / 365.25
    years_since_2017 = np.clip(frac_year - 2017.0, 0.0, 12.0)
    ct_pv_cap_mw = 420.0 + 75.0 * years_since_2017 + 3.2 * (years_since_2017 ** 2)

    sol = np.clip(np.nan_to_num(np.asarray(solar_wm2, dtype=float), nan=0.0), 0.0, 1100.0)
    cf = np.clip(np.nan_to_num(np.asarray(cloud_pct, dtype=float), nan=50.0) / 100.0, 0.0, 1.0)
    direct_boost = 1.0 + 0.15 * (1.0 - cf) * np.clip(sin_el / 0.5, 0.0, 1.2)
    t_arr = np.nan_to_num(np.asarray(temp_f, dtype=float), nan=60.0)
    temp_derate = np.clip(1.0 - 0.0022 * np.maximum(0.0, t_arr - 70.0), 0.85, 1.05)

    est_gen_mw = ct_pv_cap_mw * (sol / 1000.0) * direct_boost * temp_derate
    est_gen_mw = np.where(sin_el > 0.01, est_gen_mw, 0.0)
    return np.round(sin_el, 4), np.round(est_gen_mw, 1)


def ensure_weather_columns(hours_utc: pd.DatetimeIndex, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Ensures solar_wm2, ct_coast_inland_temp_diff_f, and west_east_temp_diff_f exist on any weather table."""
    hours_utc = pd.DatetimeIndex(hours_utc)
    w = weather_df.reindex(hours_utc).copy()
    if LATEST_REGIONAL is not None and len(LATEST_REGIONAL) > 0:
        reg_sub = LATEST_REGIONAL.reindex(hours_utc)
        if "solar_wm2" not in w.columns or w["solar_wm2"].isna().any():
            sol_reg = reg_sub["bdl_solar_wm2"] if "bdl_solar_wm2" in reg_sub.columns else reg_sub.get("ct_pop_solar_wm2")
            if sol_reg is not None:
                w["solar_wm2"] = w["solar_wm2"].fillna(sol_reg) if "solar_wm2" in w.columns else sol_reg
        for col in ["ct_coast_inland_temp_diff_f", "west_east_temp_diff_f"]:
            if col not in w.columns or w[col].isna().any():
                if col in reg_sub.columns:
                    w[col] = w[col].fillna(reg_sub[col]) if col in w.columns else reg_sub[col]
    if "solar_wm2" not in w.columns:
        w["solar_wm2"] = compute_clear_sky_solar(hours_utc, w["cloud_cover_pct"].values)
    elif w["solar_wm2"].isna().any():
        miss = w["solar_wm2"].isna()
        w.loc[miss, "solar_wm2"] = compute_clear_sky_solar(hours_utc[miss], w.loc[miss, "cloud_cover_pct"].values)
    for col in ["ct_coast_inland_temp_diff_f", "west_east_temp_diff_f"]:
        if col not in w.columns:
            w[col] = 0.0
        else:
            w[col] = w[col].fillna(0.0)
    return w


def encode_weather_vec(hours_utc: pd.DatetimeIndex, weather_df: pd.DataFrame) -> np.ndarray:
    hours_utc = pd.DatetimeIndex(hours_utc)
    loc = hours_utc.tz_convert("America/New_York")
    doy_rad = 2.0 * np.pi * loc.dayofyear.values / 365.25
    w_df = ensure_weather_columns(hours_utc, weather_df)
    w_mat = w_df[BASE_WEATHER].values.astype(np.float32)
    temp = w_mat[:, 0:1]
    cdd = np.maximum(0.0, temp - 65.0)
    hdd = np.maximum(0.0, 60.0 - temp)
    extra = w_df[["solar_wm2", "ct_coast_inland_temp_diff_f", "west_east_temp_diff_f"]].values.astype(np.float32)
    w_ext = np.hstack([w_mat, cdd, hdd, extra])
    w_norm = (w_ext - W_MEAN) / W_STD
    feats = np.column_stack([w_norm, np.sin(doy_rad).astype(np.float32), np.cos(doy_rad).astype(np.float32)])
    return feats * WEIGHTS


def get_day_regime(loc: pd.DatetimeIndex) -> np.ndarray:
    is_fed_hol = loc.tz_localize(None).normalize().isin(HOLIDAYS)
    is_non_fed = np.array([d in NON_FED_HOLIDAYS for d in loc.date], dtype=bool)
    dow = loc.dayofweek.values
    return np.where(is_fed_hol | is_non_fed | (dow == 6), 2, np.where(dow == 5, 1, 0))


def init_knn_buckets(ref_df: pd.DataFrame):
    global BUCKETS, OBS_CLOUD_SERIES, OBS_SOLGEN_SERIES, OBS_TEMP_SERIES, OBS_DEWP_SERIES
    ref_df = ensure_weather_columns(ref_df.index, ref_df)
    _, est_gen = compute_solar_geometry_and_gen(
        ref_df.index, ref_df["solar_wm2"].values, ref_df["cloud_cover_pct"].values, ref_df["temp_f"].values
    )
    ref_df["est_solar_gen_mw"] = est_gen
    OBS_CLOUD_SERIES = ref_df["cloud_cover_pct"].sort_index()
    OBS_SOLGEN_SERIES = ref_df["est_solar_gen_mw"].sort_index()
    OBS_TEMP_SERIES = ref_df["temp_f"].sort_index()
    OBS_DEWP_SERIES = ref_df["dewpoint_f"].sort_index()

    ref_loc = pd.DatetimeIndex(ref_df.index).tz_convert("America/New_York")
    ref_h = ref_loc.hour.values
    ref_reg = get_day_regime(ref_loc)
    ref_vecs = encode_weather_vec(ref_df.index, ref_df)
    ref_loads = ref_df["load_mw"].values.astype(np.float32)
    ref_solgen = ref_df["est_solar_gen_mw"].values.astype(np.float32)
    ref_t_sec = pd.DatetimeIndex(ref_df.index).as_unit("s").asi8.astype(np.int64)
    BUCKETS = {}
    for h in range(24):
        for reg in (0, 1, 2):
            mask = (ref_h == h) & (ref_reg == reg)
            BUCKETS[(h, reg)] = (ref_t_sec[mask], ref_vecs[mask], ref_loads[mask], ref_solgen[mask])


def compute_knn_weather_load(hours_utc: pd.DatetimeIndex, weather_df: pd.DataFrame, q_solgen: np.ndarray = None,
                             k_neighbors: int = 15, max_years_back: float = 4.0, recency_tau_years: float = 2.5) -> np.ndarray:
    hours_utc = pd.DatetimeIndex(hours_utc)
    if not BUCKETS:
        return np.full(len(hours_utc), np.nan, dtype=np.float64)
    loc = hours_utc.tz_convert("America/New_York")
    h_arr = loc.hour.values
    reg_arr = get_day_regime(loc)
    w_ens = ensure_weather_columns(hours_utc, weather_df)
    if q_solgen is None:
        _, q_solgen = compute_solar_geometry_and_gen(
            hours_utc, w_ens["solar_wm2"].values, w_ens["cloud_cover_pct"].values, w_ens["temp_f"].values
        )
    q_solgen = np.asarray(q_solgen, dtype=np.float32)
    q_vecs = encode_weather_vec(hours_utc, w_ens)
    q_t_sec = hours_utc.as_unit("s").asi8.astype(np.int64)

    knn_mean = np.full(len(hours_utc), np.nan, dtype=np.float64)
    min_gap_sec = 48 * 3600
    max_gap_sec = int(max_years_back * 365.25 * 86400)
    tau_sec = recency_tau_years * 365.25 * 86400

    for h in range(24):
        for reg in (0, 1, 2):
            q_idx = np.where((h_arr == h) & (reg_arr == reg))[0]
            if len(q_idx) == 0:
                continue
            qv = q_vecs[q_idx]
            valid_q = ~np.isnan(qv).any(axis=1)
            if not valid_q.any():
                continue
            q_idx = q_idx[valid_q]
            qv = qv[valid_q]
            qt = q_t_sec[q_idx]
            qs = q_solgen[q_idx][:, None]

            bt, bv, bl, bs = BUCKETS[(h, reg)]
            q_norm2 = np.sum(qv * qv, axis=1, keepdims=True)
            b_norm2 = np.sum(bv * bv, axis=1, keepdims=True).T
            d2 = np.maximum(0.0, q_norm2 + b_norm2 - 2.0 * (qv @ bv.T))

            dt = qt[:, None] - bt[None, :]
            invalid = (dt < min_gap_sec) | (dt > max_gap_sec)
            d2[invalid] = np.inf

            k_eff = min(k_neighbors, bv.shape[0])
            part_idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
            row_idx = np.arange(len(q_idx))[:, None]
            top_d2 = d2[row_idx, part_idx]
            top_dt = dt[row_idx, part_idx]
            top_loads = bl[part_idx]
            top_sol = bs[part_idx]

            has_valid = np.isfinite(top_d2).any(axis=1)
            dists = np.sqrt(np.where(np.isfinite(top_d2), top_d2, 1e9))
            recency_w = np.exp(-np.clip(top_dt, 0, None) / tau_sec)
            w = np.where(np.isfinite(top_d2), recency_w / (dists + 0.15), 0.0)
            w_sum = np.sum(w, axis=1, keepdims=True)
            w_norm = np.where(w_sum > 0, w / w_sum, 0.0)

            # Duck-Curve-adjusted neighbor load: adjusts historical neighbor load for difference in CT solar PV MW
            adj_loads = top_loads + DUCK_BETA * (top_sol - qs)
            mean_val = np.sum(w_norm * adj_loads, axis=1)
            knn_mean[q_idx[has_valid]] = mean_val[has_valid]

    return knn_mean


def lagged_same_dow_utc(hours_utc: pd.DatetimeIndex, days: int = 0, months: int = 0, years: int = 0) -> pd.DatetimeIndex:
    """Leap-year-aware and DST-aware lag lookup in Connecticut time, snapped to the exact same day of the week."""
    local = pd.DatetimeIndex(hours_utc).tz_convert("America/New_York")
    local_naive = local.tz_localize(None)
    target_dow = local_naive.dayofweek

    if days != 0 and months == 0 and years == 0:
        shifted_naive = local_naive - pd.Timedelta(days=days)
    else:
        offset = pd.DateOffset(years=years, months=months, days=days)
        days_norm = local_naive.normalize()
        unique_days = days_norm.unique()
        day_map = {d: d - offset for d in unique_days}
        shifted_naive = days_norm.map(day_map) + pd.to_timedelta(local_naive.hour, unit="h")

    dow_diff = ((target_dow - shifted_naive.dayofweek + 3) % 7) - 3
    snapped_naive = shifted_naive + pd.to_timedelta(dow_diff, unit="D")
    snapped_local = snapped_naive.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward")
    return snapped_local.tz_convert("UTC")


def make_features(hours_utc, weather, load):
    """One row of model inputs per hour.
    hours_utc: the hours to describe (UTC).  weather: table indexed by UTC hour with the WEATHER columns.
    load: Connecticut load in MW, indexed by UTC hour (only hours that are already over)."""
    hours_utc = pd.DatetimeIndex(hours_utc)
    local = hours_utc.tz_convert("America/New_York")           # people use electricity on local time
    out = pd.DataFrame(index=hours_utc)
    out["hour"] = local.hour
    out["day_of_week"] = local.dayofweek                       # 0 = Monday ... 6 = Sunday
    out["month"] = local.month
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    out["is_holiday"] = local.tz_localize(None).normalize().isin(HOLIDAYS).astype(int)
    out["is_non_fed_holiday"] = np.array([1 if d in NON_FED_HOLIDAYS else 0 for d in local.date], dtype=int)

    w_ens = ensure_weather_columns(hours_utc, weather)
    for column in WEATHER + KNN_EXTRA_COLS:
        out[column] = w_ens[column].values

    # Causal temperature & dewpoint time-series lags: observed past weather stitched to current forecast trajectory
    w_clean = weather[~weather.index.duplicated(keep="last")].sort_index()
    full_grid = pd.date_range(hours_utc.min() - pd.Timedelta(hours=24), hours_utc.max(), freq="h", tz="UTC")
    obs_t = OBS_TEMP_SERIES[~OBS_TEMP_SERIES.index.duplicated(keep="last")] if len(OBS_TEMP_SERIES) > 0 else pd.Series(dtype=float)
    obs_d = OBS_DEWP_SERIES[~OBS_DEWP_SERIES.index.duplicated(keep="last")] if len(OBS_DEWP_SERIES) > 0 else pd.Series(dtype=float)
    w_t = pd.to_numeric(w_clean["temp_f"], errors="coerce").reindex(full_grid) if "temp_f" in w_clean.columns else pd.Series(dtype=float)
    w_d = pd.to_numeric(w_clean["dewpoint_f"], errors="coerce").reindex(full_grid) if "dewpoint_f" in w_clean.columns else pd.Series(dtype=float)
    s_t = obs_t.reindex(full_grid).combine_first(w_t).interpolate(method="time", limit=12).ffill().bfill()
    s_d = obs_d.reindex(full_grid).combine_first(w_d).interpolate(method="time", limit=12).ffill().bfill()
    out["temp_lag_3h_f"] = np.round(s_t.reindex(hours_utc - pd.Timedelta(hours=3)).values, 1)
    out["temp_lag_24h_f"] = np.round(s_t.reindex(hours_utc - pd.Timedelta(hours=24)).values, 1)
    out["temp_mean_24h_f"] = np.round(s_t.rolling(24, min_periods=1).mean().reindex(hours_utc).values, 1)
    out["dewpoint_mean_24h_f"] = np.round(s_d.rolling(24, min_periods=1).mean().reindex(hours_utc).values, 1)

    # Solar energy entering the CT grid (est_solar_gen_mw)
    _, est_gen = compute_solar_geometry_and_gen(
        hours_utc, out["solar_wm2"].values, out["cloud_cover_pct"].values, out["temp_f"].values
    )
    out["est_solar_gen_mw"] = est_gen

    # Observed cloud cover & solar generation 24h ago (fallback 25h ago if 24h not yet over)
    cloud_src = OBS_CLOUD_SERIES if len(OBS_CLOUD_SERIES) > 0 else w_ens["cloud_cover_pct"]
    solgen_src = OBS_SOLGEN_SERIES if len(OBS_SOLGEN_SERIES) > 0 else pd.Series(est_gen, index=hours_utc)

    cc_24 = cloud_src.reindex(hours_utc - pd.Timedelta(hours=24)).values
    cc_25 = cloud_src.reindex(hours_utc - pd.Timedelta(hours=25)).values
    cc_obs_24 = np.where(np.isnan(cc_24), cc_25, cc_24)
    cc_obs_24 = pd.Series(cc_obs_24, index=hours_utc).ffill().bfill().fillna(out["cloud_cover_pct"]).values
    out["cloud_cover_obs_24h_pct"] = np.round(cc_obs_24, 1)
    out["cloud_cover_diff_24h_pct"] = np.round(out["cloud_cover_pct"].values - out["cloud_cover_obs_24h_pct"].values, 1)

    sg_24 = solgen_src.reindex(hours_utc - pd.Timedelta(hours=24)).values
    sg_25 = solgen_src.reindex(hours_utc - pd.Timedelta(hours=25)).values
    sgen_24 = np.nan_to_num(np.where(np.isnan(sg_24), sg_25, sg_24), nan=0.0)
    out["solar_gen_diff_24h_mw"] = np.round(est_gen - sgen_24, 1)

    # 1D lag (Duck-Curve adjusted): same hour yesterday, adjusted for difference in solar PV entering the grid
    yesterday = load.reindex(hours_utc - pd.Timedelta(hours=24)).values
    hour_before = load.reindex(hours_utc - pd.Timedelta(hours=25)).values
    load_1d_raw = np.where(np.isnan(yesterday), hour_before, yesterday)
    out["load_yesterday_mw"] = np.round(load_1d_raw + DUCK_BETA * (sgen_24 - est_gen), 1)

    # 7D lag (Duck-Curve adjusted): same day of week 1 week ago
    idx_7d = lagged_same_dow_utc(hours_utc, days=7)
    sgen_7d = np.nan_to_num(solgen_src.reindex(idx_7d).values, nan=0.0)
    out["load_7d_mw"] = np.round(load.reindex(idx_7d).values + DUCK_BETA * (sgen_7d - est_gen), 1)

    # 1M lag (Duck-Curve adjusted): 1 calendar month prior, same day of week (leap-year aware)
    idx_1m = lagged_same_dow_utc(hours_utc, months=1)
    sgen_1m = np.nan_to_num(solgen_src.reindex(idx_1m).values, nan=0.0)
    out["load_1m_dow_mw"] = np.round(load.reindex(idx_1m).values + DUCK_BETA * (sgen_1m - est_gen), 1)

    # 1Y lag (Duck-Curve adjusted): 1 calendar year prior, same day of week (leap-year aware)
    idx_1y = lagged_same_dow_utc(hours_utc, years=1)
    sgen_1y = np.nan_to_num(solgen_src.reindex(idx_1y).values, nan=0.0)
    out["load_1y_dow_mw"] = np.round(load.reindex(idx_1y).values + DUCK_BETA * (sgen_1y - est_gen), 1)

    # KNN weather-analog load (Duck-Curve adjusted): weighted mean load of the 15 closest past weather-analog hours
    out["knn_weather_load_mw"] = np.round(
        compute_knn_weather_load(hours_utc, out[WEATHER + KNN_EXTRA_COLS], q_solgen=est_gen), 1
    )
    return out[FEATURES]
# ----------------------------------------------------------------------------------------------------------------


def read_csv(bucket, key):
    blob = bucket.blob(key)
    if not blob.exists():
        return None
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def latest_weather(forecasts: pd.DataFrame) -> pd.DataFrame:
    """For every target hour, the NEWEST forecast that covers it."""
    forecasts["scraped_hour_utc"] = pd.to_datetime(forecasts["scraped_hour_utc"], utc=True)
    forecasts["target_hour_utc"] = pd.to_datetime(forecasts["target_hour_utc"], utc=True)
    forecasts = forecasts.sort_values(["target_hour_utc", "scraped_hour_utc"])
    newest = forecasts.drop_duplicates("target_hour_utc", keep="last").set_index("target_hour_utc")
    return newest


def forecast_range(bucket, card, X, preds, targets):
    """The low and high ends of your 80% range for each hour, plus how they were made.
    Returns (low, high, method), or (None, None, "none") when your model card asks for no range."""
    method = str(card.get("range_method", "none"))
    if method == "quantile_models":
        if not bucket.blob(LOW_MODEL_KEY).exists() or not bucket.blob(HIGH_MODEL_KEY).exists():
            raise ValueError("Your model card says quantile_models, but model_low.joblib or model_high.joblib is missing.")
        low_model = joblib.load(io.BytesIO(bucket.blob(LOW_MODEL_KEY).download_as_bytes()))
        high_model = joblib.load(io.BytesIO(bucket.blob(HIGH_MODEL_KEY).download_as_bytes()))
        low_cols = list(getattr(low_model, "feature_names_in_", FEATURES))
        high_cols = list(getattr(high_model, "feature_names_in_", FEATURES))
        low = low_model.predict(X[low_cols])
        high = high_model.predict(X[high_cols])
    elif method == "offsets_by_hour":
        offsets = card["range_offsets_mw"]
        if len(offsets["low"]) != 24 or len(offsets["high"]) != 24:
            raise ValueError("range_offsets_mw needs 24 low and 24 high offsets, one per Connecticut hour (0 = midnight).")
        local_hours = pd.DatetimeIndex(targets).tz_convert("America/New_York").hour
        low = preds + np.array(offsets["low"], dtype=float)[local_hours]
        high = preds + np.array(offsets["high"], dtype=float)[local_hours]
    else:
        return None, None, "none"

    widen = float(card.get("range_widen", 1.0))
    low = preds - widen * (preds - low)
    high = preds + widen * (high - preds)
    low = np.minimum(low, preds)
    high = np.maximum(high, preds)
    return low, high, method


def fill_inputs(X, load):
    """Honest gap filling (rare): a missing weather value takes its neighbor; a missing load lag takes yesterday's or the newest load."""
    w_all = WEATHER + TEMP_LAG_FEATURES + DUCK_CURVE_FEATURES
    X[w_all] = X[w_all].interpolate(limit_direction="both").fillna(0.0)
    if "knn_weather_load_mw" in X.columns and X["knn_weather_load_mw"].isna().any():
        X["knn_weather_load_mw"] = np.round(compute_knn_weather_load(X.index, X), 1)
    X["load_yesterday_mw"] = X["load_yesterday_mw"].fillna(float(load.iloc[-1]))
    for col in LOAD_LAGS + ["knn_weather_load_mw"]:
        if col in X.columns:
            X[col] = X[col].fillna(X["load_yesterday_mw"]).fillna(float(load.iloc[-1]))
    return X


def hrrr_run_tables(card, now):
    """The recent HRRR runs from the class file, newest last: a list of weather tables indexed by UTC hour.
    Returns an empty list when the file is missing or stale."""
    try:
        runs = pd.read_csv(HRRR_RUNS_URL)
    except Exception as err:
        logging.warning("HRRR runs not available: %s", err)
        return []
    runs["init_utc"] = pd.to_datetime(runs["init_utc"], utc=True)
    runs["hour_utc"] = pd.to_datetime(runs["hour_utc"], utc=True)
    inits = sorted(runs["init_utc"].unique())
    if len(inits) == 0 or pd.Timestamp(now) - pd.Timestamp(inits[-1]) > pd.Timedelta(hours=HRRR_MAX_AGE_HOURS):
        return []
    n_runs = int(card.get("hrrr_runs", 8))
    avail_cols = [c for c in BASE_WEATHER + ["solar_wm2"] if c in runs.columns]
    tables = []
    for init in inits[-n_runs:]:
        sub = runs[runs["init_utc"] == init].set_index("hour_utc")[avail_cols]
        tables.append(ensure_weather_columns(sub.index, sub))
    return tables


def pooled_range(bucket, card, targets, load, nws_weather, now):
    """Lab 6C: the quantile forest's 20 percentiles (2.5%, 7.5%, ..., 97.5%) for every HRRR run that covers an hour, pooled
    into one pile per hour; low, expected and high are the 10th, 50th and 90th percentile of the pile ("hazen": each
    value is the middle of its 5% slice, so a single run gives back exactly its own 10th, 50th and 90th).
    The model's uncertainty is inside each run's percentiles; the weather's is in how much the runs disagree.
    Returns (low, expected, high, info)."""
    if not bucket.blob(QRF_MODEL_KEY).exists():
        raise ValueError("Your model card says hrrr_qrf_pooled, but energy/models/model_qrf.joblib is missing.")
    qrf = joblib.load(io.BytesIO(bucket.blob(QRF_MODEL_KEY).download_as_bytes()))
    qrf_cols = list(getattr(qrf, "feature_names_in_", FEATURES))
    tables = hrrr_run_tables(card, now)
    weather_source = "hrrr"
    if len(tables) == 0:
        tables = [nws_weather]
        weather_source = "nws_fallback"
    piles = []
    for i in range(len(targets)):
        piles.append([])
    for table in tables:
        covered = np.asarray(targets.isin(table.index))
        if not covered.any():
            continue
        X = fill_inputs(make_features(targets[covered], table, load), load)
        if X[BASE_WEATHER].isna().any().any():
            continue
        q = qrf.predict(X[qrf_cols], quantiles=POOL_QUANTILES)
        positions = np.where(covered)[0]
        for j in range(len(positions)):
            piles[positions[j]].extend(q[j])
    # an hour no HRRR run covers (should not happen) gets the NWS forecast
    empty = []
    for i in range(len(targets)):
        if len(piles[i]) == 0:
            empty.append(i)
    if len(empty) > 0:
        X = fill_inputs(make_features(targets[empty], nws_weather, load), load)
        q = qrf.predict(X[qrf_cols], quantiles=POOL_QUANTILES)
        for j in range(len(empty)):
            piles[empty[j]].extend(q[j])
    low = np.zeros(len(targets))
    expected = np.zeros(len(targets))
    high = np.zeros(len(targets))
    runs_per_hour = []
    for i in range(len(targets)):
        low[i], expected[i], high[i] = np.quantile(piles[i], [0.1, 0.5, 0.9], method="hazen")
        runs_per_hour.append(len(piles[i]) // len(POOL_QUANTILES))
    widen = float(card.get("range_widen", 1.0))
    low = expected - widen * (expected - low)
    high = expected + widen * (high - expected)
    info = {"weather_source": weather_source, "hrrr_runs_read": len(tables) if weather_source == "hrrr" else 0,
            "runs_per_hour_min": int(min(runs_per_hour)), "runs_per_hour_max": int(max(runs_per_hour))}
    return low, expected, high, info


def run_once(dry_run: bool = False) -> dict:
    global LATEST_REGIONAL, OBS_CLOUD_SERIES, OBS_SOLGEN_SERIES, OBS_TEMP_SERIES, OBS_DEWP_SERIES
    bucket = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET)
    if not bucket.blob(MODEL_KEY).exists():
        return {"status": "no_model", "message": f"Upload your model to gs://{GCS_BUCKET}/{MODEL_KEY} (Lab 5, Part 7)."}
    model = joblib.load(io.BytesIO(bucket.blob(MODEL_KEY).download_as_bytes()))
    card = {}
    if bucket.blob(CARD_KEY).exists():
        card = json.loads(bucket.blob(CARD_KEY).download_as_text())
    model_version = str(card.get("trained_at", "unknown"))

    forecasts = read_csv(bucket, WEATHER_KEY)
    if forecasts is None or len(forecasts) == 0:
        return {"status": "no_weather_data", "message": "scraper-weather-forecast has not written its master yet."}
    reg_forecasts = read_csv(bucket, REGIONAL_WEATHER_KEY)
    if reg_forecasts is not None and len(reg_forecasts) > 0:
        LATEST_REGIONAL = latest_weather(reg_forecasts)
    else:
        LATEST_REGIONAL = None

    load_table = read_csv(bucket, LOAD_KEY)
    if load_table is None or len(load_table) == 0:
        return {"status": "no_load_data", "message": "No ISO-NE load yet: add the ISO_NE_USERNAME and ISO_NE_PASSWORD "
                "repository secrets, re-run 'Deploy scraper-isone', and wait for its next run (:15)."}
    load_table["hour_utc"] = pd.to_datetime(load_table["hour_utc"], utc=True)
    load = load_table.set_index("hour_utc")["load_mw_5min_avg"].sort_index()

    # Load historical weather + load reference archive (2017-2026) + live observed weather for KNN weather analogs & 24h cloud/solar lags
    ref_table = read_csv(bucket, KNN_REF_KEY)
    if ref_table is not None and len(ref_table) > 0:
        ref_table["hour_utc"] = pd.to_datetime(ref_table["hour_utc"], utc=True)
        ref_df = ref_table.set_index("hour_utc").sort_index()
        ref_df = ensure_weather_columns(ref_df.index, ref_df)
        ref_df["load_mw"] = ref_table.set_index("hour_utc")["load_mw"].sort_index()
        obs_table = read_csv(bucket, OBS_WEATHER_KEY)
        if obs_table is not None and len(obs_table) > 0:
            obs_table["hour_utc"] = pd.to_datetime(obs_table["hour_utc"], utc=True)
            obs_df = obs_table.set_index("hour_utc").sort_index()
            obs_df = ensure_weather_columns(obs_df.index, obs_df)
            obs_df["load_mw"] = load.reindex(obs_df.index)
            obs_df = obs_df.dropna(subset=BASE_WEATHER)
            ref_df = pd.concat([ref_df, obs_df])
            ref_df = ref_df[~ref_df.index.duplicated(keep="last")].sort_index()
        # Keep all hours for OBS_CLOUD_SERIES / OBS_SOLGEN_SERIES, and hours with non-null load_mw for KNN buckets
        init_knn_buckets(ref_df.dropna(subset=["load_mw"]))
        # Also extend OBS_CLOUD_SERIES and OBS_SOLGEN_SERIES with any very recent observed hours whose load isn't finalized yet
        _, all_est_gen = compute_solar_geometry_and_gen(
            ref_df.index, ref_df["solar_wm2"].values, ref_df["cloud_cover_pct"].values, ref_df["temp_f"].values
        )
        OBS_CLOUD_SERIES = ref_df["cloud_cover_pct"].sort_index()
        OBS_SOLGEN_SERIES = pd.Series(all_est_gen, index=ref_df.index).sort_index()
        OBS_TEMP_SERIES = ref_df["temp_f"].sort_index()
        OBS_DEWP_SERIES = ref_df["dewpoint_f"].sort_index()

    now = datetime.now(timezone.utc)
    this_hour = pd.Timestamp(now).floor("h")
    targets = pd.date_range(this_hour + pd.Timedelta(hours=1), periods=HOURS_AHEAD, freq="h")
    weather = ensure_weather_columns(targets, latest_weather(forecasts))
    X = make_features(targets, weather, load)

    # Honest gap filling (rare): a missing weather value takes its neighbor; a missing load lag takes yesterday's or the newest load
    filled_weather = int(X[BASE_WEATHER].isna().sum().sum())
    w_all = WEATHER + TEMP_LAG_FEATURES + DUCK_CURVE_FEATURES
    X[w_all] = X[w_all].interpolate(limit_direction="both").fillna(0.0)
    if X["knn_weather_load_mw"].isna().any():
        X["knn_weather_load_mw"] = np.round(compute_knn_weather_load(X.index, X), 1)
    filled_load = int(X[LOAD_LAGS + ["knn_weather_load_mw"]].isna().sum().sum())
    X["load_yesterday_mw"] = X["load_yesterday_mw"].fillna(float(load.iloc[-1]))
    for col in LOAD_LAGS + ["knn_weather_load_mw"]:
        X[col] = X[col].fillna(X["load_yesterday_mw"]).fillna(float(load.iloc[-1]))
    if X[BASE_WEATHER].isna().any().any():
        return {"status": "no_weather_data", "message": "The weather forecast does not cover the next 24 hours yet."}

    model_cols = list(getattr(model, "feature_names_in_", FEATURES))
    preds = model.predict(X[model_cols])
    range_error = ""
    pool_info = {}
    try:                         # a broken range never costs you your expected-value forecast
        if str(card.get("range_method", "none")) == "hrrr_qrf_pooled":
            low, pooled_expected, high, pool_info = pooled_range(bucket, card, targets, load, weather, now)
            preds = pooled_expected            # the expected value is the middle of the pooled pile
            range_method = "hrrr_qrf_pooled"
            if pool_info["weather_source"] != "hrrr":
                range_method = "hrrr_qrf_pooled_nws"
        else:
            low, high, range_method = forecast_range(bucket, card, X, preds, targets)
    except Exception as err:
        low, high, range_method = None, None, "error"
        range_error = str(err)[:300]
    if low is None:
        low = np.full(len(preds), np.nan)
        high = np.full(len(preds), np.nan)
    out = pd.DataFrame({"issued_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "target_hour_utc": targets.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "lead_hours": np.arange(1, HOURS_AHEAD + 1),
                        "predicted_low_mw": np.round(low, 1),
                        "predicted_load_mw": np.round(preds, 1),
                        "predicted_high_mw": np.round(high, 1),
                        "model_version": model_version,
                        "range_method": range_method})
    summary = {"status": "ok", "model_version": model_version, "first_target_utc": out["target_hour_utc"].iloc[0],
               "last_target_utc": out["target_hour_utc"].iloc[-1], "newest_load_hour_utc": load.index.max().isoformat(),
               "min_mw": float(out["predicted_load_mw"].min()), "max_mw": float(out["predicted_load_mw"].max()),
               "range_method": range_method,
               "avg_range_width_mw": None if low is None or np.isnan(low).all() else round(float(np.mean(high - low)), 1),
               "range_error": range_error,
               "filled_weather_values": filled_weather, "filled_load_values": filled_load}
    summary.update(pool_info)
    if dry_run:
        summary["dry_run"] = True
        return summary

    csv_text = out.to_csv(index=False)
    bucket.blob(LATEST_KEY).upload_from_string(csv_text, content_type="text/csv")
    bucket.blob(f"{HISTORY_PREFIX}/{now.strftime('%Y%m%dT%H%M%SZ')}.csv").upload_from_string(csv_text, content_type="text/csv")
    old = read_csv(bucket, MASTER_KEY)
    master = out if old is None else pd.concat([old, out], ignore_index=True)
    bucket.blob(MASTER_KEY).upload_from_string(master.to_csv(index=False), content_type="text/csv")
    summary["latest_key"] = LATEST_KEY
    summary["master_rows"] = int(len(master))
    return summary


def forecast_24h_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        logging.info(json.dumps(result)[:500])
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
