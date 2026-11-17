# scraper-weather-forecast: Multi-Airport & Multi-Model Weather Forecast Scraper for Connecticut & Surrounding Region
#
# Every hour (:05) this function:
#   1. Downloads the official NWS Digital hourly forecast (forecast.weather.gov) in parallel for all 25 airports:
#      - All 9 Connecticut airports: BDL, HFD, BDR, HVN, GON, DXR, OXC, MMK, IJD
#      - 13 Surrounding NY / Long Island / MA / RI airports: HPN, LGA, JFK, ISP, FOK, POU, ALB, BAF, CEF, ORH, PVD, WST, BOS
#      - 3 Remaining ISO-NE 8-City Index airports: CON, BTV, PWM
#   2. Downloads multi-model NWP forecasts (NOAA HRRR / BestMatch, NOAA GFS Seamless, ECMWF IFS, plus Solar W/m²)
#      for all 25 airports from Open-Meteo.
#   3. Preserves gs://<bucket>/energy/datasets/bdl_weather_forecast_master.csv (100% backwards-compatible with Lab 5/6)
#   4. Appends all 25 airports x 4 forecast models to:
#      gs://<bucket>/energy/datasets/multi_airport_weather_forecast_master.csv
#   5. Appends hourly regional & multi-model consensus features (CT population-weighted weather, Coastal vs. Inland
#      sea-breeze gradient, West-to-East frontal gradient, ISO-NE 8-City Index, and 4-model consensus/spread) to:
#      gs://<bucket>/energy/datasets/regional_weather_forecast_master.csv
# HTTP entrypoint: scrape_weather_forecast_http

import concurrent.futures
import io
import json
import logging
import os
import random
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from google.cloud import storage

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
ENERGY_PREFIX = os.getenv("ENERGY_PREFIX", "energy")
LAT = os.getenv("FORECAST_LAT", "41.9389")          # Bradley International Airport
LON = os.getenv("FORECAST_LON", "-72.6832")
MAX_LEAD = int(os.getenv("MAX_LEAD_HOURS", "48"))
USER_AGENT = os.getenv("USER_AGENT", "OPIM5512-student-scraper (UConn)")

MASTER_KEY = f"{ENERGY_PREFIX}/datasets/bdl_weather_forecast_master.csv"
MULTI_AIRPORT_KEY = f"{ENERGY_PREFIX}/datasets/multi_airport_weather_forecast_master.csv"
REGIONAL_MASTER_KEY = f"{ENERGY_PREFIX}/datasets/regional_weather_forecast_master.csv"

# 25 Airports: (lat, lon, region_group, ct_load_weight, isone_8city_weight)
AIRPORTS = {
    # --- Inside Connecticut (9 Airports) ---
    "BDL": (41.9389, -72.6832, "CT_Inland", 0.18, 0.14),   # Bradley Intl (Windsor Locks / Hartford North)
    "HFD": (41.7367, -72.6494, "CT_Inland", 0.10, 0.00),   # Hartford-Brainard (Central CT River Valley)
    "BDR": (41.1635, -73.1262, "CT_Coast",  0.22, 0.13),   # Bridgeport Sikorsky (Fairfield County Coast)
    "HVN": (41.2637, -72.8868, "CT_Coast",  0.16, 0.00),   # Tweed-New Haven (South-Central CT Coast)
    "GON": (41.3301, -72.0451, "CT_Coast",  0.06, 0.00),   # Groton-New London (SE CT Coast)
    "DXR": (41.3715, -73.4822, "CT_West",   0.08, 0.00),   # Danbury Municipal (Western CT / NY Border)
    "OXC": (41.4786, -73.1353, "CT_West",   0.06, 0.00),   # Waterbury-Oxford (Naugatuck Valley / Hills)
    "MMK": (41.5087, -72.8295, "CT_Inland", 0.06, 0.00),   # Meriden Markham (Central I-91 Corridor)
    "IJD": (41.7440, -72.1802, "CT_East",   0.03, 0.00),   # Windham / Willimantic (Eastern CT)
    # --- Surrounding NY & Long Island (7 Airports) ---
    "HPN": (41.0670, -73.7076, "NY_West",   0.05, 0.00),   # White Plains / Westchester (borders Stamford/Greenwich)
    "LGA": (40.7769, -73.8740, "NY_West",   0.00, 0.00),   # LaGuardia NYC
    "JFK": (40.6413, -73.7781, "NY_West",   0.00, 0.00),   # JFK Intl NYC
    "ISP": (40.7952, -73.1002, "NY_LI",     0.00, 0.00),   # Long Island MacArthur (Islip, NY)
    "FOK": (40.8437, -72.6318, "NY_LI",     0.00, 0.00),   # Gabreski (Westhampton / Eastern LI, NY)
    "POU": (41.6266, -73.8842, "NY_West",   0.00, 0.00),   # Poughkeepsie / Hudson Valley, NY
    "ALB": (42.7483, -73.8017, "NY_North",  0.00, 0.00),   # Albany Intl, NY
    # --- Surrounding MA & RI (6 Airports) ---
    "BAF": (42.1578, -72.7156, "MA_West",   0.00, 0.00),   # Westfield-Barnes (Springfield / Pioneer Valley, MA)
    "CEF": (42.1940, -72.5348, "MA_West",   0.00, 0.00),   # Westover (Chicopee / Springfield, MA)
    "ORH": (42.2673, -71.8757, "MA_East",   0.00, 0.11),   # Worcester Regional, MA
    "PVD": (41.7240, -71.4282, "RI_East",   0.00, 0.10),   # T.F. Green (Providence, RI)
    "WST": (41.3496, -71.8033, "RI_East",   0.00, 0.00),   # Westerly State (RI / SE CT Border)
    "BOS": (42.3656, -71.0096, "MA_East",   0.00, 0.32),   # Boston Logan Intl, MA
    # --- Remaining ISO-NE 8-City Weather Index Stations (3 Airports) ---
    "CON": (43.2027, -71.5023, "NE_North",  0.00, 0.07),   # Concord, NH
    "BTV": (44.4719, -73.1533, "NE_North",  0.00, 0.05),   # Burlington, VT
    "PWM": (43.6462, -70.3087, "NE_North",  0.00, 0.08),   # Portland, ME
}

# (DWML tag, its "type" attribute) -> our column name; units: F, %, mph, degrees, inches
FIELDS = {("temperature", "hourly"): "temp_f", ("temperature", "dew point"): "dewpoint_f",
          ("humidity", "relative"): "rel_humidity_pct", ("wind-speed", "sustained"): "wind_speed_mph",
          ("wind-speed", "gust"): "wind_gust_mph", ("direction", "wind"): "wind_dir_deg",
          ("cloud-amount", "total"): "cloud_cover_pct", ("probability-of-precipitation", "floating"): "precip_chance_pct",
          ("hourly-qpf", "floating"): "precip_in"}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


def download_nws_station(station: str, lat: float, lon: float) -> tuple:
    url = f"https://forecast.weather.gov/MapClick.php?lat={lat}&lon={lon}&FcstType=digitalDWML"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
    r.raise_for_status()
    return station, r.text


def parse_forecast(xml_text: str) -> tuple:
    """Returns (issued_at_utc, DataFrame indexed by target_hour_utc with one column per weather field)."""
    root = ET.fromstring(xml_text)
    issued = pd.Timestamp(root.find(".//creation-date").text).tz_convert("UTC")
    starts = [pd.Timestamp(t.text).tz_convert("UTC") for t in root.iter("start-valid-time")]
    table = pd.DataFrame(index=pd.DatetimeIndex(starts, name="target_hour_utc"))
    for el in root.iter():
        key = (el.tag, el.get("type"))
        if key in FIELDS:
            values = []
            for v in el.findall("value"):
                values.append(float(v.text) if v.text not in (None, "") else np.nan)
            if len(values) == len(table):
                table[FIELDS[key]] = values
    return issued, table


def fetch_all_nws_forecasts(scraped_hour: pd.Timestamp) -> tuple:
    """Fetches NWS Digital forecasts for all 25 airports in parallel. Returns (bdl_xml, nws_df)."""
    codes = list(AIRPORTS.keys())

    def _worker(code):
        lat, lon, _, _, _ = AIRPORTS[code]
        try:
            _, xml_text = download_nws_station(code, lat, lon)
            issued, tbl = parse_forecast(xml_text)
            keep = tbl[(tbl.index >= scraped_hour) & (tbl.index <= scraped_hour + pd.Timedelta(hours=MAX_LEAD))].copy()
            keep["station"] = code
            keep["model"] = "nws"
            keep["issued_at_utc"] = issued
            keep["scraped_hour_utc"] = scraped_hour
            keep["lead_hours"] = ((keep.index - scraped_hour) / pd.Timedelta(hours=1)).astype(int)
            return code, xml_text, keep.reset_index()
        except Exception as err:
            logging.warning("NWS fetch failed for %s: %s", code, err)
            return code, None, None

    bdl_xml = None
    frames = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for code, xml_text, df_st in ex.map(_worker, codes):
            if code == "BDL" and xml_text is not None:
                bdl_xml = xml_text
            if df_st is not None and len(df_st) > 0:
                frames.append(df_st)

    nws_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return bdl_xml, nws_df


def fetch_open_meteo_multi_model(scraped_hour: pd.Timestamp) -> pd.DataFrame:
    """Fetches NOAA HRRR/BestMatch, NOAA GFS Seamless, and ECMWF IFS (plus solar_wm2) for all 25 airports in one call."""
    codes = list(AIRPORTS.keys())
    lats = ",".join(str(AIRPORTS[c][0]) for c in codes)
    lons = ",".join(str(AIRPORTS[c][1]) for c in codes)
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}"
        "&hourly=temperature_2m,dew_point_2m,relative_humidity_2m,wind_speed_10m,wind_gusts_10m,wind_direction_10m,cloud_cover,precipitation,shortwave_radiation"
        "&models=best_match,gfs_seamless,ecmwf_ifs025"
        "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=UTC&forecast_days=3"
    )
    payload = None
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(10, 45))
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                delay = 3.0 * attempt + random.uniform(0.5, 3.0)
                logging.warning("Open-Meteo HTTP %s on attempt %d/%d; retrying in %.1fs", r.status_code, attempt, max_attempts, delay)
                time.sleep(delay)
                continue
            r.raise_for_status()
            payload = r.json()
            break
        except requests.exceptions.RequestException as err:
            if attempt < max_attempts:
                delay = 3.0 * attempt + random.uniform(0.5, 3.0)
                logging.warning("Open-Meteo fetch attempt %d/%d failed (%s); retrying in %.1fs", attempt, max_attempts, err, delay)
                time.sleep(delay)
            else:
                logging.warning("Open-Meteo multi-model fetch failed after %d attempts: %s", max_attempts, err)
                return pd.DataFrame()
        except Exception as err:
            logging.warning("Open-Meteo multi-model parse failed: %s", err)
            return pd.DataFrame()

    if payload is None:
        return pd.DataFrame()

    frames = []
    model_map = {
        "best_match": "hrrr_blend",
        "gfs_seamless": "gfs",
        "ecmwf_ifs025": "ecmwf",
    }
    for idx, code in enumerate(codes):
        item = payload[idx] if isinstance(payload, list) else payload
        hourly = item.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []), utc=True)
        if len(times) == 0:
            continue
        mask = (times >= scraped_hour) & (times <= scraped_hour + pd.Timedelta(hours=MAX_LEAD))
        t_keep = times[mask]
        if len(t_keep) == 0:
            continue

        for om_suffix, short_model in model_map.items():
            def _col(base_name):
                arr = hourly.get(f"{base_name}_{om_suffix}")
                if arr is None:
                    return np.full(len(t_keep), np.nan)
                return np.asarray(arr, dtype=float)[mask]

            df_m = pd.DataFrame({
                "target_hour_utc": t_keep,
                "scraped_hour_utc": scraped_hour,
                "issued_at_utc": scraped_hour,
                "lead_hours": ((t_keep - scraped_hour) / pd.Timedelta(hours=1)).astype(int),
                "station": code,
                "model": short_model,
                "temp_f": np.round(_col("temperature_2m"), 1),
                "dewpoint_f": np.round(_col("dew_point_2m"), 1),
                "rel_humidity_pct": np.round(_col("relative_humidity_2m"), 1),
                "wind_speed_mph": np.round(_col("wind_speed_10m"), 1),
                "wind_gust_mph": np.round(_col("wind_gusts_10m"), 1),
                "wind_dir_deg": np.round(_col("wind_direction_10m"), 0),
                "cloud_cover_pct": np.round(_col("cloud_cover"), 0),
                "precip_in": np.round(_col("precipitation"), 3),
                "solar_wm2": np.round(_col("shortwave_radiation"), 0),
            })
            frames.append(df_m)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_regional_summary(multi_df: pd.DataFrame, scraped_hour: pd.Timestamp) -> pd.DataFrame:
    """Aggregates the 25 airports x 4 forecast models into one row per (scraped_hour_utc, target_hour_utc)
    with CT population-weighted weather, coastal vs. inland gradients, West-East frontal gradients,
    ISO-NE 8-city index weather, and 4-model consensus + spread."""
    if len(multi_df) == 0:
        return pd.DataFrame()

    ct_weights = {k: v[3] for k, v in AIRPORTS.items() if v[3] > 0}
    isone_weights = {k: v[4] for k, v in AIRPORTS.items() if v[4] > 0}
    coast_stations = ["BDR", "HVN", "GON", "WST", "ISP", "FOK"]
    inland_ct_stations = ["BDL", "HFD", "MMK"]
    west_ny_stations = ["ALB", "POU", "HPN", "DXR", "LGA", "JFK"]
    east_ma_ri_stations = ["PVD", "ORH", "BOS", "WST"]

    # Average across available models per (target_hour_utc, station) first for spatial aggregations;
    # NWS-only fallback lacks solar_wm2 (or partial fields), so ensure numeric schema columns exist safely.
    agg_cols = ["temp_f", "dewpoint_f", "rel_humidity_pct", "wind_speed_mph", "cloud_cover_pct", "solar_wm2"]
    work = multi_df.copy()
    for col in agg_cols:
        if col not in work.columns:
            work[col] = np.nan
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    st_avg = work.groupby(["target_hour_utc", "station"], as_index=False)[agg_cols].mean()

    hours = sorted(work["target_hour_utc"].unique())
    out = pd.DataFrame({"target_hour_utc": hours})
    out["scraped_hour_utc"] = scraped_hour
    out["lead_hours"] = ((out["target_hour_utc"] - scraped_hour) / pd.Timedelta(hours=1)).astype(int)

    def _weighted_series(df_sub, weight_map, col):
        sub = df_sub[df_sub["station"].isin(weight_map)].copy()
        sub["w"] = sub["station"].map(weight_map)
        sub = sub.dropna(subset=[col])
        if len(sub) == 0:
            return pd.Series(dtype=float)
        sub["wx"] = sub[col] * sub["w"]
        g = sub.groupby("target_hour_utc")
        return (g["wx"].sum() / g["w"].sum()).round(2)

    for col in agg_cols:
        out[f"ct_pop_{col}"] = out["target_hour_utc"].map(_weighted_series(st_avg, ct_weights, col))
        out[f"isone_8city_{col}"] = out["target_hour_utc"].map(_weighted_series(st_avg, isone_weights, col))

    def _group_mean(stations, col):
        sub = st_avg[st_avg["station"].isin(stations)]
        return sub.groupby("target_hour_utc")[col].mean().round(2)

    out["ct_coast_temp_f"] = out["target_hour_utc"].map(_group_mean(coast_stations, "temp_f"))
    out["ct_inland_temp_f"] = out["target_hour_utc"].map(_group_mean(inland_ct_stations, "temp_f"))
    out["ct_coast_inland_temp_diff_f"] = (out["ct_coast_temp_f"] - out["ct_inland_temp_f"]).round(2)

    out["west_ny_temp_f"] = out["target_hour_utc"].map(_group_mean(west_ny_stations, "temp_f"))
    out["east_ma_ri_temp_f"] = out["target_hour_utc"].map(_group_mean(east_ma_ri_stations, "temp_f"))
    out["west_east_temp_diff_f"] = (out["west_ny_temp_f"] - out["east_ma_ri_temp_f"]).round(2)

    # Multi-model consensus & spread across NWS, HRRR, GFS, ECMWF at BDL + BDR (CT's two ISO-NE stations)
    ct_main = work[work["station"].isin(["BDL", "BDR"])].groupby(["target_hour_utc", "model"], as_index=False)["temp_f"].mean()
    for mod in ["nws", "hrrr_blend", "gfs", "ecmwf"]:
        s_mod = ct_main[ct_main["model"] == mod].set_index("target_hour_utc")["temp_f"].round(2)
        out[f"ct_{mod}_temp_f"] = out["target_hour_utc"].map(s_mod)

    mod_cols = [c for c in ["ct_nws_temp_f", "ct_hrrr_blend_temp_f", "ct_gfs_temp_f", "ct_ecmwf_temp_f"] if c in out.columns]
    out["ct_model_temp_spread_f"] = (out[mod_cols].max(axis=1) - out[mod_cols].min(axis=1)).round(2)
    bdl_sol = st_avg[st_avg["station"] == "BDL"].set_index("target_hour_utc")["solar_wm2"].round(1)
    out["bdl_solar_wm2"] = out["target_hour_utc"].map(bdl_sol)
    return out


def upsert_csv(bucket, key: str, new_df: pd.DataFrame, dedup_cols: list, date_cols: list) -> int:
    if len(new_df) == 0:
        return 0
    blob = bucket.blob(key)
    if blob.exists():
        old = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
        for c in date_cols:
            if c in old.columns:
                old[c] = pd.to_datetime(old[c], utc=True)
        both = pd.concat([old, new_df], ignore_index=True)
        master = both.drop_duplicates(dedup_cols, keep="last")
    else:
        master = new_df.copy()
    master = master.sort_values(dedup_cols)
    bucket.blob(key).upload_from_string(master.to_csv(index=False), content_type="text/csv")
    return len(master)


def run_once(dry_run: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    scraped_hour = pd.Timestamp(now).floor("h")

    # 1. Fetch NWS Digital XML for all 25 airports + Open-Meteo (HRRR, GFS, ECMWF, Solar) for all 25 airports
    bdl_xml, nws_all = fetch_all_nws_forecasts(scraped_hour)
    om_all = fetch_open_meteo_multi_model(scraped_hour)

    if bdl_xml is None:
        _, bdl_xml = download_nws_station("BDL", float(LAT), float(LON))
    issued, table = parse_forecast(bdl_xml)

    # Preserve standard 24-hour BDL master table for backwards compatibility
    keep_bdl = table[(table.index >= scraped_hour) & (table.index <= scraped_hour + pd.Timedelta(hours=24))].copy()
    keep_bdl["lead_hours"] = ((keep_bdl.index - scraped_hour) / pd.Timedelta(hours=1)).astype(int)
    keep_bdl["scraped_hour_utc"] = scraped_hour
    keep_bdl["issued_at_utc"] = issued
    keep_bdl = keep_bdl.reset_index()
    first = ["scraped_hour_utc", "issued_at_utc", "target_hour_utc", "lead_hours"]
    keep_bdl = keep_bdl[first + [c for c in keep_bdl.columns if c not in first]]

    # Combine all 25 airports x 4 models (nws, hrrr_blend, gfs, ecmwf)
    multi_df = pd.concat([nws_all, om_all], ignore_index=True) if len(om_all) > 0 else nws_all
    regional_df = build_regional_summary(multi_df, scraped_hour)

    # Enrich keep_bdl with solar_wm2 and regional multi-airport gradients so bdl_weather_forecast_master.csv has them ready
    if len(regional_df) > 0:
        reg_idx = regional_df.set_index("target_hour_utc")
        bdl_sol_s = reg_idx["bdl_solar_wm2"] if "bdl_solar_wm2" in reg_idx.columns else pd.Series(dtype=float)
        keep_bdl["solar_wm2"] = pd.to_numeric(keep_bdl["target_hour_utc"].map(bdl_sol_s), errors="coerce").fillna(0.0)
        for rcol in ["ct_coast_inland_temp_diff_f", "west_east_temp_diff_f", "ct_pop_temp_f", "isone_8city_temp_f",
                     "ct_hrrr_blend_temp_f", "ct_gfs_temp_f", "ct_ecmwf_temp_f", "ct_model_temp_spread_f"]:
            if rcol in reg_idx.columns:
                keep_bdl[rcol] = keep_bdl["target_hour_utc"].map(reg_idx[rcol])

    summary = {
        "status": "ok",
        "issued_at_utc": issued.isoformat(),
        "bdl_rows_this_run": int(len(keep_bdl)),
        "airports_scraped": int(multi_df["station"].nunique()) if len(multi_df) > 0 else 1,
        "models_scraped": sorted(multi_df["model"].unique().tolist()) if len(multi_df) > 0 else ["nws"],
        "multi_airport_rows_this_run": int(len(multi_df)),
        "regional_rows_this_run": int(len(regional_df)),
    }
    if dry_run:
        summary["dry_run"] = True
        summary["regional_sample"] = json.loads(regional_df.head(2).to_json(orient="records", date_format="iso")) if len(regional_df) > 0 else []
        return summary

    bucket = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    bucket.blob(f"{ENERGY_PREFIX}/raw/weather_forecast/{stamp}.xml").upload_from_string(bdl_xml, content_type="application/xml")

    summary["bdl_master_rows"] = upsert_csv(
        bucket, MASTER_KEY, keep_bdl,
        ["scraped_hour_utc", "target_hour_utc"],
        ["scraped_hour_utc", "issued_at_utc", "target_hour_utc"]
    )
    summary["regional_master_rows"] = upsert_csv(
        bucket, REGIONAL_MASTER_KEY, regional_df,
        ["scraped_hour_utc", "target_hour_utc"],
        ["scraped_hour_utc", "target_hour_utc"]
    )
    # Keep last 7 days of raw multi-airport rows in multi_airport_weather_forecast_master.csv so file stays compact
    if len(multi_df) > 0:
        blob_m = bucket.blob(MULTI_AIRPORT_KEY)
        if blob_m.exists():
            old_m = pd.read_csv(io.BytesIO(blob_m.download_as_bytes()))
            for c in ["scraped_hour_utc", "issued_at_utc", "target_hour_utc"]:
                if c in old_m.columns:
                    old_m[c] = pd.to_datetime(old_m[c], utc=True)
            cutoff = scraped_hour - pd.Timedelta(days=7)
            old_m = old_m[old_m["scraped_hour_utc"] >= cutoff]
            both_m = pd.concat([old_m, multi_df], ignore_index=True)
            master_m = both_m.drop_duplicates(["scraped_hour_utc", "target_hour_utc", "station", "model"], keep="last")
        else:
            master_m = multi_df
        master_m = master_m.sort_values(["scraped_hour_utc", "target_hour_utc", "station", "model"])
        blob_m.upload_from_string(master_m.to_csv(index=False), content_type="text/csv")
        summary["multi_airport_master_rows"] = int(len(master_m))
    else:
        summary["multi_airport_master_rows"] = 0
    return summary


def scrape_weather_forecast_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        logging.info(json.dumps(result)[:500])
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
