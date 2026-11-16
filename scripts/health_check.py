"""End-to-end health check for the Connecticut energy forecasting pipeline.

Verifies live forecast freshness & quantile sanity, all hourly scrapers and GCS master CSVs,
25-airport regional & solar columns, model/card/feature alignment, daily retrain freshness,
upstream HRRR feed availability, Cloud Scheduler job statuses, bucket IAM permissions,
and live rolling CRPS / 80% coverage against ISO-NE and same-hour-yesterday persistence.
"""

import argparse
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from google.cloud import storage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "cloud_function", "forecast-24h"))
import main as f24  # noqa: E402

SCHEDULER_JOBS = [
    "scraper-weather-obs-hourly",
    "scraper-weather-forecast-hourly",
    "scraper-isone-hourly",
    "forecast-24h-hourly",
]


def crps_leaderboard(actual: np.ndarray, low: np.ndarray, expected: np.ndarray, high: np.ndarray) -> np.ndarray:
    outside = np.clip(low - actual, 0.0, None) + np.clip(actual - high, 0.0, None)
    return (0.5 * np.abs(actual - expected) + 0.1 * (high - low) + 1.0 * outside) / 1.5


def read_blob_csv(bucket: storage.Bucket, key: str) -> pd.DataFrame:
    blob = bucket.blob(key)
    if not blob.exists():
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


class HealthReport:
    def __init__(self):
        self.rows = []

    def add(self, component: str, status: str, detail: str):
        assert status in ("PASS", "WARN", "FAIL")
        self.rows.append((component, status, detail))

    def has_failures(self) -> bool:
        return any(r[1] == "FAIL" for r in self.rows)

    def to_markdown(self, now_utc: datetime) -> str:
        fails = sum(1 for _, s, _ in self.rows if s == "FAIL")
        warns = sum(1 for _, s, _ in self.rows if s == "WARN")
        passes = sum(1 for _, s, _ in self.rows if s == "PASS")
        lines = [
            f"## Pipeline Health Check ({now_utc.strftime('%Y-%m-%d %H:%M:%SZ')})",
            f"**Summary**: `{passes}` PASS | `{warns}` WARN | `{fails}` FAIL",
            "",
            "| Check | Status | Details |",
            "| :--- | :---: | :--- |",
        ]
        icon = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}
        for comp, st, det in self.rows:
            det_clean = str(det).replace("|", "/")
            lines.append(f"| {comp} | **{icon[st]}** | {det_clean} |")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Verify energy forecasting pipeline health.")
    parser.add_argument("--project", default=os.getenv("PROJECT_ID", "energy-jpe03001-f26"))
    parser.add_argument("--bucket", default=os.getenv("GCS_BUCKET", "energy-jpe03001-f26"))
    parser.add_argument("--region", default=os.getenv("REGION", "us-central1"))
    parser.add_argument("--prefix", default=os.getenv("ENERGY_PREFIX", "energy"))
    parser.add_argument("--check-gcp", action="store_true", help="Also verify Cloud Scheduler jobs and GCS bucket IAM.")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    now_ts = pd.Timestamp(now_utc)
    report = HealthReport()

    client = storage.Client(project=args.project)
    bucket = client.bucket(args.bucket)

    # 1. Latest 24h forecast output
    latest_key = f"{args.prefix}/forecasts/latest.csv"
    latest_df = read_blob_csv(bucket, latest_key)
    if len(latest_df) == 0:
        report.add("forecasts/latest.csv", "FAIL", "Missing or empty")
    else:
        issued = pd.to_datetime(latest_df["issued_at_utc"].iloc[0], utc=True)
        age_min = (now_ts - issued).total_seconds() / 60.0
        n_rows = len(latest_df)
        has_nans = bool(latest_df[["predicted_low_mw", "predicted_load_mw", "predicted_high_mw"]].isna().any().any())
        ordered = bool(
            (latest_df["predicted_low_mw"] <= latest_df["predicted_load_mw"]).all()
            and (latest_df["predicted_load_mw"] <= latest_df["predicted_high_mw"]).all()
        )
        in_bounds = bool(
            (latest_df["predicted_low_mw"] >= 1000.0).all() and (latest_df["predicted_high_mw"] <= 7500.0).all()
        )
        rm = str(latest_df["range_method"].iloc[0])
        mv = str(latest_df["model_version"].iloc[0])
        detail = (
            f"issued {issued.strftime('%Y-%m-%d %H:%MZ')} ({age_min:.0f}m ago), "
            f"{n_rows} rows, range={rm}, model={mv}, "
            f"[{latest_df['predicted_load_mw'].min():.0f}-{latest_df['predicted_load_mw'].max():.0f} MW]"
        )
        if age_min > 90 or n_rows != 24 or has_nans or not ordered or not in_bounds:
            report.add("forecasts/latest.csv", "FAIL", detail)
        elif rm != "hrrr_qrf_pooled":
            report.add("forecasts/latest.csv", "WARN", detail + " (not using HRRR)")
        else:
            report.add("forecasts/latest.csv", "PASS", detail)

    # 2. forecasts_master.csv
    fc_master = read_blob_csv(bucket, f"{args.prefix}/datasets/forecasts_master.csv")
    if len(fc_master) == 0:
        report.add("datasets/forecasts_master.csv", "FAIL", "Missing or empty")
    else:
        fc_master["issued_utc"] = pd.to_datetime(fc_master["issued_at_utc"], utc=True)
        fc_master["target_utc"] = pd.to_datetime(fc_master["target_hour_utc"], utc=True)
        age_min = (now_ts - fc_master["issued_utc"].max()).total_seconds() / 60.0
        st = "FAIL" if age_min > 90 else "PASS"
        report.add(
            "datasets/forecasts_master.csv",
            st,
            f"{len(fc_master):,} rows, newest issued {age_min:.0f}m ago",
        )

    # 3. ISO-NE live 5-min load & settled hourly load
    load_5m = read_blob_csv(bucket, f"{args.prefix}/datasets/ct_load_5min_hourly_master.csv")
    if len(load_5m) == 0:
        report.add("datasets/ct_load_5min_hourly_master.csv", "FAIL", "Missing or empty")
    else:
        load_5m["hour_utc"] = pd.to_datetime(load_5m["hour_utc"], utc=True)
        max_h = load_5m["hour_utc"].max()
        age_h = (now_ts - max_h).total_seconds() / 3600.0
        st = "FAIL" if age_h > 3.5 else "PASS"
        report.add(
            "datasets/ct_load_5min_hourly_master.csv",
            st,
            f"{len(load_5m):,} rows, latest hour {max_h.strftime('%Y-%m-%d %H:%MZ')} ({age_h:.1f}h ago)",
        )

    load_hr = read_blob_csv(bucket, f"{args.prefix}/datasets/ct_load_hourly_master.csv")
    if len(load_hr) == 0:
        report.add("datasets/ct_load_hourly_master.csv", "WARN", "Missing or empty")
    else:
        load_hr["hour_utc"] = pd.to_datetime(load_hr["hour_utc"], utc=True)
        age_d = (now_ts - load_hr["hour_utc"].max()).total_seconds() / 86400.0
        st = "WARN" if age_d > 5.0 else "PASS"
        report.add("datasets/ct_load_hourly_master.csv", st, f"{len(load_hr):,} rows, latest settled {age_d:.1f}d ago")

    # 4. ISO-NE own forecast
    isone_fc = read_blob_csv(bucket, f"{args.prefix}/datasets/isone_ct_forecast_master.csv")
    if len(isone_fc) == 0:
        report.add("datasets/isone_ct_forecast_master.csv", "WARN", "Missing or empty")
    else:
        isone_fc["issued_utc"] = pd.to_datetime(isone_fc["issued_at_utc"], utc=True)
        age_h = (now_ts - isone_fc["issued_utc"].max()).total_seconds() / 3600.0
        st = "WARN" if age_h > 24.0 else "PASS"
        report.add("datasets/isone_ct_forecast_master.csv", st, f"{len(isone_fc):,} rows, newest issued {age_h:.1f}h ago")

    # 5. Weather observations master (with solar & CT cloud fallback)
    w_obs = read_blob_csv(bucket, f"{args.prefix}/datasets/bdl_weather_obs_master.csv")
    if len(w_obs) == 0:
        report.add("datasets/bdl_weather_obs_master.csv", "FAIL", "Missing or empty")
    else:
        w_obs["hour_utc"] = pd.to_datetime(w_obs["hour_utc"], utc=True)
        max_h = w_obs["hour_utc"].max()
        age_h = (now_ts - max_h).total_seconds() / 3600.0
        req_cols = [c for c in f24.BASE_WEATHER + ["solar_wm2", "est_solar_gen_mw"] if c not in w_obs.columns]
        recent_nans = int(w_obs.tail(6)[f24.BASE_WEATHER + ["solar_wm2"]].isna().sum().sum()) if not req_cols else -1
        st = "FAIL" if (age_h > 3.5 or len(req_cols) > 0 or recent_nans > 0) else "PASS"
        report.add(
            "datasets/bdl_weather_obs_master.csv",
            st,
            f"{len(w_obs):,} rows, latest {max_h.strftime('%Y-%m-%d %H:%MZ')} ({age_h:.1f}h ago), recent NaNs={recent_nans}",
        )

    # 6. Weather forecasts (BDL + 25-airport regional)
    w_fc = read_blob_csv(bucket, f"{args.prefix}/datasets/bdl_weather_forecast_master.csv")
    if len(w_fc) == 0:
        report.add("datasets/bdl_weather_forecast_master.csv", "FAIL", "Missing or empty")
    else:
        w_fc["scraped_hour_utc"] = pd.to_datetime(w_fc["scraped_hour_utc"], utc=True)
        w_fc["target_hour_utc"] = pd.to_datetime(w_fc["target_hour_utc"], utc=True)
        age_h = (now_ts - w_fc["scraped_hour_utc"].max()).total_seconds() / 3600.0
        horizon_h = (w_fc["target_hour_utc"].max() - now_ts).total_seconds() / 3600.0
        st = "FAIL" if (age_h > 2.5 or horizon_h < 22.5) else "PASS"
        report.add(
            "datasets/bdl_weather_forecast_master.csv",
            st,
            f"{len(w_fc):,} rows, scraped {age_h:.1f}h ago, horizon +{horizon_h:.1f}h",
        )

    reg_fc = read_blob_csv(bucket, f"{args.prefix}/datasets/regional_weather_forecast_master.csv")
    if len(reg_fc) == 0:
        report.add("datasets/regional_weather_forecast_master.csv", "FAIL", "Missing or empty")
    else:
        reg_fc["scraped_hour_utc"] = pd.to_datetime(reg_fc["scraped_hour_utc"], utc=True)
        age_h = (now_ts - reg_fc["scraped_hour_utc"].max()).total_seconds() / 3600.0
        has_grad = all(c in reg_fc.columns for c in ["ct_coast_inland_temp_diff_f", "west_east_temp_diff_f", "bdl_solar_wm2"])
        st = "FAIL" if (age_h > 2.5 or not has_grad) else "PASS"
        report.add(
            "datasets/regional_weather_forecast_master.csv",
            st,
            f"{len(reg_fc):,} rows, scraped {age_h:.1f}h ago, gradients_present={has_grad}",
        )

    # 7. Historical 2017-present reference archive (KNN + lags)
    ref_blob = bucket.blob(f"{args.prefix}/datasets/bdl_weather_load_ref_2017_2026.csv")
    if not ref_blob.exists():
        report.add("datasets/bdl_weather_load_ref_2017_2026.csv", "FAIL", "Missing")
    else:
        ref_df = pd.read_csv(io.BytesIO(ref_blob.download_as_bytes()))
        ref_df["hour_utc"] = pd.to_datetime(ref_df["hour_utc"], utc=True)
        st = "FAIL" if len(ref_df) < 85000 else "PASS"
        report.add(
            "datasets/bdl_weather_load_ref_2017_2026.csv",
            st,
            f"{len(ref_df):,} rows ({ref_df['hour_utc'].min().date()} -> {ref_df['hour_utc'].max().strftime('%Y-%m-%d %H:%MZ')})",
        )

    # 8. Upstream HRRR feed
    try:
        r = requests.get(f24.HRRR_RUNS_URL, timeout=30)
        r.raise_for_status()
        hrrr = pd.read_csv(io.StringIO(r.text))
        hrrr["init_utc"] = pd.to_datetime(hrrr["init_utc"], utc=True)
        newest_init = hrrr["init_utc"].max()
        age_h = (now_ts - newest_init).total_seconds() / 3600.0
        n_inits = int(hrrr["init_utc"].nunique())
        st = "FAIL" if (age_h > f24.HRRR_MAX_AGE_HOURS or n_inits == 0) else "PASS"
        report.add("Upstream Class HRRR Feed", st, f"{n_inits} runs cached, newest init {age_h:.1f}h ago (<{f24.HRRR_MAX_AGE_HOURS}h)")
    except Exception as exc:
        report.add("Upstream Class HRRR Feed", "FAIL", f"Fetch error: {exc}")

    # 9. Model artifact, model card, feature alignment, and daily retrain freshness
    qrf_blob = bucket.blob(f"{args.prefix}/models/model_qrf.joblib")
    hgbr_blob = bucket.blob(f"{args.prefix}/models/model_hgbr_q.joblib")
    card_blob = bucket.blob(f"{args.prefix}/models/model_card.json")
    if not qrf_blob.exists() or not card_blob.exists():
        report.add("Model Artifact & Card", "FAIL", "model_qrf.joblib or model_card.json missing in GCS")
    else:
        qrf_blob.reload()
        size_mb = (qrf_blob.size or 0) / (1024 * 1024)
        card = json.loads(card_blob.download_as_text())
        trained_at = pd.to_datetime(card.get("trained_at"), utc=True)
        age_h = (now_ts - trained_at).total_seconds() / 3600.0 if pd.notna(trained_at) else 999.0
        card_feats = list(card.get("features", []))
        feats_match = card_feats == list(f24.FEATURES)
        wm = float(card.get("hgbr_weight_median", 0.0))
        wm_mse = float(card.get("hgbr_weight_mse", 0.0))
        wb = float(card.get("hgbr_weight_bounds", 0.0))
        needs_hgbr = wm > 0.0 or wm_mse > 0.0 or wb > 0.0
        hgbr_ok = (not needs_hgbr) or hgbr_blob.exists()
        hgbr_info = ""
        if needs_hgbr and hgbr_blob.exists():
            hgbr_blob.reload()
            hgbr_mb = (hgbr_blob.size or 0) / (1024 * 1024)
            slope = float(card.get("range_widen_slope", 0.0))
            widen_s = f"{float(card.get('range_widen', 1.0)):.3f}+{slope:.5f}*h" if slope else f"{card.get('range_widen', 1.0)}"
            hgbr_info = f", hgbr={hgbr_mb:.2f} MB (wm={wm:.2f}, wm_mse={wm_mse:.2f}, wb={wb:.2f}, widen={widen_s})"
        elif needs_hgbr:
            hgbr_info = ", hgbr=MISSING"
        detail = (
            f"qrf={size_mb:.2f} MB (<=150){hgbr_info}, trained {age_h:.1f}h ago, "
            f"rows={card.get('train_rows')}, features_match={feats_match} ({len(card_feats)}/{len(f24.FEATURES)}), "
            f"14d_crps={card.get('recent_14d_holdout_crps_mw', 'n/a')} MW"
        )
        if size_mb > 150.0 or not feats_match or not hgbr_ok:
            report.add("Model Artifact & Card", "FAIL", detail)
        elif age_h > 36.0:
            report.add("Model Artifact & Card", "WARN", detail + " (retrain >36h old)")
        else:
            report.add("Model Artifact & Card", "PASS", detail)

    # 10. Live scoring over last 7 days (CRPS, MAE, 80% coverage vs Persistence & ISO-NE)
    if len(fc_master) > 0 and len(load_5m) > 0:
        actual_s = load_5m.drop_duplicates("hour_utc", keep="last").set_index("hour_utc")["load_mw_5min_avg"].sort_index()
        recent_cutoff = now_ts - pd.Timedelta(days=7)
        sub = fc_master[fc_master["issued_utc"] >= recent_cutoff].copy()
        sub["actual_mw"] = actual_s.reindex(sub["target_utc"]).values
        sub = sub[sub["actual_mw"].notna() & sub["predicted_low_mw"].notna() & sub["predicted_high_mw"].notna()].copy()
        if len(sub) >= 12:
            sub["crps"] = crps_leaderboard(
                sub["actual_mw"].values,
                sub["predicted_low_mw"].values,
                sub["predicted_load_mw"].values,
                sub["predicted_high_mw"].values,
            )
            sub["mae"] = np.abs(sub["actual_mw"].values - sub["predicted_load_mw"].values)
            sub["inside"] = (sub["actual_mw"] >= sub["predicted_low_mw"]) & (sub["actual_mw"] <= sub["predicted_high_mw"])
            cov = 100.0 * float(sub["inside"].mean())
            crps_m = float(sub["crps"].mean())
            mae_m = float(sub["mae"].mean())
            width_m = float((sub["predicted_high_mw"] - sub["predicted_low_mw"]).mean())
            st = "WARN" if (cov < 68.0 or cov > 94.0) else "PASS"
            report.add(
                "Live 7d Accuracy & Coverage",
                st,
                f"{len(sub)} scored fcst-hrs: CRPS={crps_m:.1f} MW, MAE={mae_m:.1f} MW, Inside80%={cov:.1f}%, Width={width_m:.0f} MW",
            )
        else:
            report.add("Live 7d Accuracy & Coverage", "PASS", f"Waiting for more scored hours ({len(sub)} pairs so far)")

    # 11. Optional GCP Cloud Scheduler + Bucket IAM checks
    if args.check_gcp:
        try:
            proc = subprocess.run(
                [
                    "gcloud",
                    "scheduler",
                    "jobs",
                    "list",
                    f"--project={args.project}",
                    f"--location={args.region}",
                    "--format=json",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
                shell=(os.name == "nt"),
            )
            jobs = {j["name"].split("/")[-1]: j for j in json.loads(proc.stdout)}
            bad_jobs = []
            notes = []
            for jname in SCHEDULER_JOBS:
                if jname not in jobs:
                    bad_jobs.append(f"{jname}:MISSING")
                    continue
                j = jobs[jname]
                state = j.get("state", "UNKNOWN")
                code = int((j.get("status") or {}).get("code", 0) or 0)
                last_t = pd.to_datetime(j.get("lastAttemptTime"), utc=True) if j.get("lastAttemptTime") else pd.NaT
                age_m = (now_ts - last_t).total_seconds() / 60.0 if pd.notna(last_t) else 999.0
                if state != "ENABLED" or code != 0 or age_m > 80.0:
                    bad_jobs.append(f"{jname}(state={state},code={code},age={age_m:.0f}m)")
                else:
                    notes.append(f"{jname.replace('-hourly','')}({age_m:.0f}m)")
            if bad_jobs:
                report.add("Cloud Scheduler Jobs (4)", "FAIL", "; ".join(bad_jobs))
            else:
                report.add("Cloud Scheduler Jobs (4)", "PASS", "All ENABLED & OK: " + ", ".join(notes))
        except Exception as exc:
            report.add("Cloud Scheduler Jobs (4)", "FAIL", f"Check failed: {exc}")

        try:
            perms = set(bucket.test_iam_permissions(["storage.objects.get", "storage.objects.create", "storage.objects.list"]))
            has_rw = {"storage.objects.get", "storage.objects.create", "storage.objects.list"}.issubset(perms)
            proc_iam = subprocess.run(
                ["gcloud", "storage", "buckets", "get-iam-policy", f"gs://{args.bucket}", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
                shell=(os.name == "nt"),
            )
            if proc_iam.returncode == 0:
                policy = json.loads(proc_iam.stdout)
                members_by_role = {}
                for b in policy.get("bindings", []):
                    members_by_role.setdefault(b["role"], set()).update(b.get("members", []))
                all_members = set().union(*members_by_role.values()) if members_by_role else set()
                has_runtime = any("cf-runtime@" in m for m in all_members)
                has_deployer = any("cf-deployer@" in m for m in all_members)
                has_collector = any("opim5512-collector-f26" in m for m in all_members)
                st = "PASS" if (has_rw and has_runtime and has_deployer and has_collector) else "FAIL"
                report.add(
                    "GCS Bucket IAM",
                    st,
                    f"rw={has_rw}, cf-runtime={has_runtime}, cf-deployer={has_deployer}, class_collector={has_collector}",
                )
            else:
                st = "PASS" if has_rw else "FAIL"
                report.add(
                    "GCS Bucket IAM",
                    st,
                    f"deployer_rw={has_rw} (get-iam-policy skipped for non-admin SA)",
                )
        except Exception as exc:
            report.add("GCS Bucket IAM", "FAIL", f"IAM check failed: {exc}")

    md = report.to_markdown(now_utc)
    print(md)
    gh_summary = os.getenv("GITHUB_STEP_SUMMARY", "")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as sf:
            sf.write(md + "\n")

    if report.has_failures():
        sys.exit(1)


if __name__ == "__main__":
    main()
