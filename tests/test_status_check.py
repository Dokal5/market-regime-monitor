from __future__ import annotations

import urllib.error

import pytest

import scripts.status_check as status_check
from scripts.status_check import (
    GitStatus,
    HashComparison,
    StatusCheckError,
    compare_hashes,
    fetch_text,
    local_file_hashes,
    parse_update_health_csv,
    recommend_action,
)


def test_parse_update_health_csv_reads_first_data_row() -> None:
    csv_text = (
        "latest_market_date,update_health_status,success_rate\n"
        "2026-05-15,healthy,1.000000\n"
    )

    row = parse_update_health_csv(csv_text)

    assert row["latest_market_date"] == "2026-05-15"
    assert row["update_health_status"] == "healthy"


def test_parse_update_health_csv_rejects_empty_csv() -> None:
    with pytest.raises(RuntimeError):
        parse_update_health_csv("latest_market_date,update_health_status\n")


def test_compare_hashes_detects_matching_files() -> None:
    comparison = compare_hashes({"index.html": "abc"}, {"index.html": "abc"})

    assert comparison.synced is True
    assert comparison.mismatched_files == []


def test_compare_hashes_detects_changed_and_missing_files() -> None:
    comparison = compare_hashes(
        {"index.html": "abc", "ticker_momentum.csv": "def"},
        {"index.html": "zzz", "update_health.csv": "123"},
    )

    assert comparison.synced is False
    assert comparison.mismatched_files == [
        "index.html",
        "ticker_momentum.csv",
        "update_health.csv",
    ]


def test_local_file_hashes_reads_requested_files(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "index.html").write_text("dashboard", encoding="utf-8")

    monkeypatch.setattr(status_check, "OUTPUT_DIR", output_dir)

    hashes = local_file_hashes(["index.html"])

    assert hashes == {
        "index.html": "66cd9688a2ae068244ea01e70f0e230f5623b7fa4cdecb65070a09ec06452262"
    }


def test_fetch_text_wraps_network_failure(monkeypatch) -> None:
    def fail_urlopen(url: str, timeout: int):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(status_check.urllib.request, "urlopen", fail_urlopen)

    with pytest.raises(StatusCheckError, match="failed to fetch"):
        fetch_text("https://example.com/update_health.csv")


def test_recommend_action_when_local_is_behind() -> None:
    action = recommend_action(
        GitStatus(branch="main", head="a", origin_main="b", ahead=0, behind=2),
        HashComparison(synced=False, mismatched_files=["index.html"]),
        HashComparison(synced=False, mismatched_files=["index.html"]),
        HashComparison(synced=True, mismatched_files=[]),
    )

    assert action == "git pull --ff-only origin main"


def test_recommend_action_when_pages_differs_from_raw() -> None:
    action = recommend_action(
        GitStatus(branch="main", head="a", origin_main="a", ahead=0, behind=0),
        HashComparison(synced=True, mismatched_files=[]),
        HashComparison(synced=False, mismatched_files=["index.html"]),
        HashComparison(synced=False, mismatched_files=["index.html"]),
    )

    assert action == "wait_for_pages_or_rerun_deploy"


def test_recommend_action_when_everything_is_synced() -> None:
    action = recommend_action(
        GitStatus(branch="main", head="a", origin_main="a", ahead=0, behind=0),
        HashComparison(synced=True, mismatched_files=[]),
        HashComparison(synced=True, mismatched_files=[]),
        HashComparison(synced=True, mismatched_files=[]),
    )

    assert action == "none"
