# scraper-weather-obs: the weather that ACTUALLY happened at Bradley Airport (BDL) & CT airports, every hour.
#
# Every hour (:05) this function:
#   1. downloads the last few hours of official airport reports (METARs) for BDL (with CT backup stations HFD, BDR, HVN)
#      from the Iowa State Mesonet
#   2. cleans them exactly like the Lab 5 training notebook (one report per hour, rounded to the hour, UTC,
#      wind in mph, sky cover in percent, rain "T" -> 0.001) and computes observed shortwave solar (solar_wm2)
#      and estimated Connecticut grid solar PV generation (est_solar_gen_mw)
#   3. saves the raw download:  gs://<bucket>/energy/raw/weather_obs/<UTC time of this run>.csv
#   4. adds/updates rows in:    gs://<bucket>/energy/datasets/bdl_weather_obs_master.csv   (one row per UTC hour)
# HTTP entrypoint: scrape_weather_obs_http        Cleaning matches Module4/Lab5_Building_the_Training_Data.ipynb

import io
import json
import logging
import os
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from google.cloud import storage

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
ENERGY_PREFIX = os.getenv("ENERGY_PREFIX", "energy")
STATION = os.getenv("WEATHER_STATION", "BDL")
CT_STATIONS = ["BDL", "HFD", "BDR", "HVN", "GON", "DXR", "OXC", "MMK", "IJD"]
HOURS_BACK = int(os.getenv("HOURS_BACK", "12"))       # re-read 12 hours so late reports and yesterday's daytime cloud cover land
USER_AGENT = os.getenv("USER_AGENT", "OPIM5512-student-scraper (UConn)")

MASTER_KEY = f"{ENERGY_PREFIX}/datasets/bdl_weather_obs_master.csv"
KNOTS_TO_MPH = 1.15078
SKY_PERCENT = {"CLR": 0, "SKC": 0, "FEW": 19, "SCT": 44, "BKN": 75, "OVC": 100, "VV": 100}
COLUMNS = ["temp_f", "dewpoint_f", "rel_humidity_pct", "feels_like_f", "wind_speed_mph", "wind_gust_mph",
           "wind_dir_deg", "pressure_inhg", "visibility_mi", "cloud_cover_pct", "precip_in",
           "is_rain", "is_snow", "is_fog", "solar_wm2", "est_solar_gen_mw"]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


def compute_solar_from_cloud(hours_utc: pd.DatetimeIndex, cloud_pct: np.ndarray, temp_f: np.ndarray) -> tuple:
    """Computes observed/estimated (solar_wm2, est_solar_gen_mw) from solar geometry, cloud cover, and temperature."""
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
    solar_wm2 = np.round(clear_sky * atten, 1)

    frac_year = hours_utc.year.values + (doy - 1.0) / 365.25
    years_since_2017 = np.clip(frac_year - 2017.0, 0.0, 12.0)
    ct_pv_cap_mw = 420.0 + 75.0 * years_since_2017 + 3.2 * (years_since_2017 ** 2)
    direct_boost = 1.0 + 0.15 * (1.0 - cf) * np.clip(sin_el / 0.5, 0.0, 1.2)
    t_arr = np.nan_to_num(np.asarray(temp_f, dtype=float), nan=60.0)
    temp_derate = np.clip(1.0 - 0.0022 * np.maximum(0.0, t_arr - 70.0), 0.85, 1.05)
    est_gen_mw = np.where(sin_el > 0.01, ct_pv_cap_mw * (solar_wm2 / 1000.0) * direct_boost * temp_derate, 0.0)
    return solar_wm2, np.round(est_gen_mw, 1)


def download_reports(start_utc: datetime, end_utc: datetime) -> str:
    """Raw CSV of routine (3) and special (4) reports between two UTC times for BDL and backup CT stations."""
    fields = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "p01i", "alti", "vsby",
              "skyc1", "skyc2", "skyc3", "skyc4", "wxcodes", "feel"]
    params = [("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
              ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", "3"), ("report_type", "4"),
              ("year1", start_utc.year), ("month1", start_utc.month), ("day1", start_utc.day),
              ("hour1", start_utc.hour), ("minute1", 0),
              ("year2", end_utc.year), ("month2", end_utc.month), ("day2", end_utc.day),
              ("hour2", end_utc.hour), ("minute2", 59)]
    for st in CT_STATIONS:
        params.append(("station", st))
    for f in fields:
        params.append(("data", f))
    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params=params,
                     headers={"User-Agent": USER_AGENT}, timeout=120)
    r.raise_for_status()
    return r.text


def clean_reports(text: str) -> pd.DataFrame:
    """Same steps as the Lab 5 training notebook, using CT stations as fallback if BDL misses a cloud/weather reading."""
    raw = pd.read_csv(io.StringIO(text), na_values=["M"], keep_default_na=False)
    if len(raw) == 0:
        return pd.DataFrame(columns=COLUMNS)
    raw["valid"] = pd.to_datetime(raw["valid"], utc=True)
    raw["hour_utc"] = raw["valid"].dt.round("h")

    # Compute cloud cover across all CT stations first for backup
    for layer in ["skyc1", "skyc2", "skyc3", "skyc4"]:
        raw[f"_{layer}"] = raw[layer].astype(str).str.strip().map(SKY_PERCENT)
    raw["_cloud_pct"] = raw[["_skyc1", "_skyc2", "_skyc3", "_skyc4"]].max(axis=1)
    ct_cloud = raw.groupby("hour_utc")["_cloud_pct"].mean().round(1)

    bdl = raw[raw["station"] == STATION].copy()
    if len(bdl) == 0:
        bdl = raw.copy()
    bdl["minutes_off"] = (bdl["valid"] - bdl["hour_utc"]).abs().dt.total_seconds() / 60
    bdl["is_routine"] = bdl["valid"].dt.minute == 51
    bdl = bdl.sort_values(["hour_utc", "is_routine", "minutes_off"], ascending=[True, False, True])
    h = bdl.drop_duplicates("hour_utc", keep="first").set_index("hour_utc")

    out = pd.DataFrame(index=h.index)
    out["temp_f"] = pd.to_numeric(h["tmpf"], errors="coerce")
    out["dewpoint_f"] = pd.to_numeric(h["dwpf"], errors="coerce")
    out["rel_humidity_pct"] = pd.to_numeric(h["relh"], errors="coerce")
    out["feels_like_f"] = pd.to_numeric(h["feel"], errors="coerce")
    sknt = pd.to_numeric(h["sknt"], errors="coerce")
    gust = pd.to_numeric(h["gust"], errors="coerce")
    out["wind_speed_mph"] = sknt * KNOTS_TO_MPH
    out["wind_gust_mph"] = gust.fillna(sknt) * KNOTS_TO_MPH
    out["wind_dir_deg"] = pd.to_numeric(h["drct"], errors="coerce")
    out["pressure_inhg"] = pd.to_numeric(h["alti"], errors="coerce")
    out["visibility_mi"] = pd.to_numeric(h["vsby"], errors="coerce")
    layers = pd.DataFrame(index=h.index)
    for layer in ["skyc1", "skyc2", "skyc3", "skyc4"]:
        layers[layer] = h[layer].astype(str).str.strip().map(SKY_PERCENT)
    out["cloud_cover_pct"] = layers.max(axis=1).fillna(ct_cloud.reindex(h.index)).ffill().bfill().fillna(50.0)
    out["precip_in"] = pd.to_numeric(h["p01i"].astype(str).replace("T", "0.001"), errors="coerce").fillna(0.0)
    codes = h["wxcodes"].fillna("").astype(str)
    out["is_rain"] = codes.str.contains("RA|DZ").astype(float)
    out["is_snow"] = codes.str.contains("SN|PL|GR|GS").astype(float)
    out["is_fog"] = codes.str.contains("FG|BR").astype(float)

    sol_wm2, est_gen = compute_solar_from_cloud(out.index, out["cloud_cover_pct"].values, out["temp_f"].values)
    out["solar_wm2"] = sol_wm2
    out["est_solar_gen_mw"] = est_gen
    out.index.name = "hour_utc"
    return out[COLUMNS]


def run_once(dry_run: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=HOURS_BACK)
    text = download_reports(start, now)
    new = clean_reports(text)
    new["scraped_at_utc"] = now.isoformat()

    if dry_run:
        return {"status": "ok", "dry_run": True, "hours": len(new), "rows": json.loads(new.reset_index().to_json(orient="records", date_format="iso"))[-3:]}

    bucket = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    bucket.blob(f"{ENERGY_PREFIX}/raw/weather_obs/{stamp}.csv").upload_from_string(text, content_type="text/csv")

    blob = bucket.blob(MASTER_KEY)
    if blob.exists():
        old = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col="hour_utc")
        old.index = pd.to_datetime(old.index, utc=True)
        both = pd.concat([old, new])
        master = both[~both.index.duplicated(keep="last")].sort_index()
        if "solar_wm2" not in master.columns or master["solar_wm2"].isna().any():
            s_w, s_g = compute_solar_from_cloud(master.index, master["cloud_cover_pct"].values, master["temp_f"].values)
            master["solar_wm2"] = master["solar_wm2"].fillna(pd.Series(s_w, index=master.index)) if "solar_wm2" in master.columns else s_w
            master["est_solar_gen_mw"] = master["est_solar_gen_mw"].fillna(pd.Series(s_g, index=master.index)) if "est_solar_gen_mw" in master.columns else s_g
    else:
        master = new.sort_index()
    bucket.blob(MASTER_KEY).upload_from_string(master.to_csv(index_label="hour_utc"), content_type="text/csv")

    last = new.index.max().isoformat() if len(new) else None
    return {"status": "ok", "hours_this_run": int(len(new)), "latest_hour_utc": last,
            "master_rows": int(len(master)), "master_key": MASTER_KEY}


def scrape_weather_obs_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        logging.info(json.dumps(result)[:500])
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
