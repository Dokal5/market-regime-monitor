from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
PAGES_BASE_URL = "https://dokal5.github.io/market-regime-monitor/"
RAW_BASE_URL = "https://raw.githubusercontent.com/Dokal5/market-regime-monitor/main/outputs/"
REMOTE_NAME = "origin"
REMOTE_BRANCH = "main"

CHECK_FILES = [
    "index.html",
    "ticker_momentum.csv",
    "industry_momentum.csv",
    "watchlist_alerts.csv",
    "update_health.csv",
    "journal/latest.md",
]


@dataclass(frozen=True)
class GitStatus:
    branch: str
    head: str
    origin_main: str
    ahead: int
    behind: int

    @property
    def synced(self) -> bool:
        return self.ahead == 0 and self.behind == 0 and self.head == self.origin_main


@dataclass(frozen=True)
class HashComparison:
    synced: bool
    mismatched_files: list[str]


class StatusCheckError(RuntimeError):
    pass


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise StatusCheckError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def fetch_origin() -> None:
    run_git(["fetch", "--quiet", REMOTE_NAME, f"{REMOTE_BRANCH}:refs/remotes/{REMOTE_NAME}/{REMOTE_BRANCH}"])


def get_git_status(fetch: bool = True) -> GitStatus:
    if fetch:
        fetch_origin()

    branch = run_git(["branch", "--show-current"]) or "DETACHED"
    head = run_git(["rev-parse", "HEAD"])
    origin_main = run_git(["rev-parse", f"{REMOTE_NAME}/{REMOTE_BRANCH}"])
    counts = run_git(["rev-list", "--left-right", "--count", f"HEAD...{REMOTE_NAME}/{REMOTE_BRANCH}"])
    ahead_text, behind_text = counts.split()

    return GitStatus(
        branch=branch,
        head=head,
        origin_main=origin_main,
        ahead=int(ahead_text),
        behind=int(behind_text),
    )


def parse_update_health_csv(csv_text: str) -> dict[str, str]:
    rows = list(csv.DictReader(csv_text.splitlines()))
    if not rows:
        raise StatusCheckError("update_health.csv has no data rows")
    return rows[0]


def read_local_update_health() -> dict[str, str]:
    path = OUTPUT_DIR / "update_health.csv"
    if not path.exists():
        raise StatusCheckError(f"missing local file: {path.relative_to(REPO_ROOT)}")
    return parse_update_health_csv(path.read_text(encoding="utf-8"))


def fetch_text(url: str, timeout: int = 20) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as exc:
        raise StatusCheckError(f"failed to fetch {url}: {exc}") from exc


def fetch_bytes(url: str, timeout: int = 20) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise StatusCheckError(f"failed to fetch {url}: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_file_hashes(files: list[str] = CHECK_FILES) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in files:
        path = OUTPUT_DIR / relative_path
        if not path.exists():
            raise StatusCheckError(f"missing local file: {path.relative_to(REPO_ROOT)}")
        hashes[relative_path] = sha256_bytes(path.read_bytes())
    return hashes


def remote_file_hashes(base_url: str, files: list[str] = CHECK_FILES) -> dict[str, str]:
    return {relative_path: sha256_bytes(fetch_bytes(urljoin(base_url, relative_path))) for relative_path in files}


def compare_hashes(left: dict[str, str], right: dict[str, str]) -> HashComparison:
    files = sorted(set(left) | set(right))
    mismatches = [file for file in files if left.get(file) != right.get(file)]
    return HashComparison(synced=not mismatches, mismatched_files=mismatches)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def short_sha(value: str) -> str:
    return value[:7] if value else ""


def format_files(files: list[str]) -> str:
    return ",".join(files) if files else "none"


def recommend_action(
    git_status: GitStatus,
    raw_comparison: HashComparison,
    pages_comparison: HashComparison,
    pages_raw_comparison: HashComparison,
) -> str:
    if git_status.behind > 0:
        return "git pull --ff-only origin main"
    if git_status.ahead > 0:
        return "push_or_rebase_local_commits"
    if not raw_comparison.synced:
        return "inspect_local_outputs_vs_origin_main"
    if not pages_raw_comparison.synced:
        return "wait_for_pages_or_rerun_deploy"
    if not pages_comparison.synced:
        return "inspect_pages_output"
    return "none"


def build_report() -> tuple[int, list[str]]:
    git_status = get_git_status(fetch=True)
    local_health = read_local_update_health()
    online_health = parse_update_health_csv(fetch_text(urljoin(PAGES_BASE_URL, "update_health.csv")))

    local_hashes = local_file_hashes()
    raw_hashes = remote_file_hashes(RAW_BASE_URL)
    pages_hashes = remote_file_hashes(PAGES_BASE_URL)

    raw_comparison = compare_hashes(local_hashes, raw_hashes)
    pages_comparison = compare_hashes(local_hashes, pages_hashes)
    pages_raw_comparison = compare_hashes(pages_hashes, raw_hashes)

    exit_code = 0
    if not (git_status.synced and raw_comparison.synced and pages_comparison.synced and pages_raw_comparison.synced):
        exit_code = 1

    lines = [
        f"LOCAL_BRANCH={git_status.branch}",
        f"LOCAL_HEAD={short_sha(git_status.head)}",
        f"ORIGIN_MAIN={short_sha(git_status.origin_main)}",
        f"LOCAL_AHEAD={git_status.ahead}",
        f"LOCAL_BEHIND={git_status.behind}",
        f"LOCAL_SYNCED={bool_text(git_status.synced)}",
        f"LOCAL_LATEST_MARKET_DATE={local_health.get('latest_market_date', '')}",
        f"ONLINE_LATEST_MARKET_DATE={online_health.get('latest_market_date', '')}",
        f"UPDATE_HEALTH_STATUS={online_health.get('update_health_status', '')}",
        f"RAW_SYNCED={bool_text(raw_comparison.synced)}",
        f"RAW_MISMATCHES={format_files(raw_comparison.mismatched_files)}",
        f"PAGES_SYNCED={bool_text(pages_comparison.synced)}",
        f"PAGES_MISMATCHES={format_files(pages_comparison.mismatched_files)}",
        f"PAGES_RAW_SYNCED={bool_text(pages_raw_comparison.synced)}",
        f"PAGES_RAW_MISMATCHES={format_files(pages_raw_comparison.mismatched_files)}",
        f"ACTION={recommend_action(git_status, raw_comparison, pages_comparison, pages_raw_comparison)}",
    ]
    return exit_code, lines


def main() -> int:
    try:
        exit_code, lines = build_report()
    except StatusCheckError as exc:
        print(f"ERROR={exc}", file=sys.stderr)
        return 2

    for line in lines:
        print(line)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
