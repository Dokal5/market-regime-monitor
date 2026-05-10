from __future__ import annotations

import os
import subprocess
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import (
    UPDATE_HEALTH_COLUMNS,
    UPDATE_HEALTH_MAX_DATA_AGE_DAYS,
    UPDATE_HEALTH_MIN_SUCCESS_RATE,
)
from src.data_quality import build_data_quality_summary


NEW_YORK_TZ = ZoneInfo("America/New_York")


def isoformat_seconds(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat()


def git_sha_from_local_repo() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def github_run_url(environment: dict[str, str]) -> str:
    repository = environment.get("GITHUB_REPOSITORY", "")
    run_id = environment.get("GITHUB_RUN_ID", "")
    if not repository or not run_id:
        return ""

    server_url = environment.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    return f"{server_url}/{repository}/actions/runs/{run_id}"


def market_data_age_days(latest_market_date: Any, generated_at_new_york: datetime) -> int | None:
    if pd.isna(latest_market_date) or not latest_market_date:
        return None

    try:
        market_date = date.fromisoformat(str(latest_market_date))
    except ValueError:
        return None

    return (generated_at_new_york.date() - market_date).days


def classify_update_health(summary: dict[str, Any], age_days: int | None) -> tuple[str, str]:
    latest_market_date = summary.get("latest_market_date")
    if not latest_market_date:
        return "unknown", "無法取得最新市場日期。"

    warning_reasons = []
    if age_days is not None and age_days > UPDATE_HEALTH_MAX_DATA_AGE_DAYS:
        warning_reasons.append(f"市場資料已落後 {age_days} 天。")
    if summary.get("missing_count", 0) > 0:
        warning_reasons.append(f"{summary['missing_count']} 檔缺資料。")
    if summary.get("stale_count", 0) > 0:
        warning_reasons.append(f"{summary['stale_count']} 檔資料落後。")
    if summary.get("success_rate", 0) < UPDATE_HEALTH_MIN_SUCCESS_RATE:
        warning_reasons.append(f"資料成功率低於 {UPDATE_HEALTH_MIN_SUCCESS_RATE:.0%}。")

    if warning_reasons:
        return "warning", " ".join(warning_reasons)

    return "healthy", "排程輸出與資料新鮮度目前正常。"


def build_update_health_output(
    ticker_output: pd.DataFrame,
    generated_at_utc: datetime | None = None,
    environment: dict[str, str] | None = None,
) -> pd.DataFrame:
    env = dict(os.environ if environment is None else environment)
    generated_utc = generated_at_utc or datetime.now(timezone.utc)
    if generated_utc.tzinfo is None:
        generated_utc = generated_utc.replace(tzinfo=timezone.utc)
    generated_utc = generated_utc.astimezone(timezone.utc)
    generated_new_york = generated_utc.astimezone(NEW_YORK_TZ)

    summary = build_data_quality_summary(ticker_output)
    age_days = market_data_age_days(summary.get("latest_market_date"), generated_new_york)
    status, note = classify_update_health(summary, age_days)

    run_context = "github_actions" if env.get("GITHUB_ACTIONS", "").lower() == "true" else "local"
    row = {
        "generated_at_utc": isoformat_seconds(generated_utc),
        "generated_at_new_york": isoformat_seconds(generated_new_york),
        "run_context": run_context,
        "github_workflow": env.get("GITHUB_WORKFLOW", ""),
        "github_run_id": env.get("GITHUB_RUN_ID", ""),
        "github_run_url": github_run_url(env),
        "git_sha": env.get("GITHUB_SHA", "") or git_sha_from_local_repo(),
        "latest_market_date": summary.get("latest_market_date"),
        "market_data_age_days": age_days,
        "success_rate": summary.get("success_rate", 0),
        "missing_count": summary.get("missing_count", 0),
        "stale_count": summary.get("stale_count", 0),
        "limited_history_count": summary.get("limited_history_count", 0),
        "update_health_status": status,
        "update_health_note": note,
    }
    return pd.DataFrame([row], columns=UPDATE_HEALTH_COLUMNS)
