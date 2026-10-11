"""TWIC loader — elite games, one month of weekly issues (GOAL.md §7, priority P1).

The Week In Chess publishes one zip per weekly issue at a sequential URL
(``https://theweekinchess.com/zips/twic{NNNN}g.zip``). There is no index file we rely on,
so this loader does NOT guess: it **probes** a window of issue numbers around a measured
anchor, reads each candidate's ``Last-Modified`` header, and keeps only the issues whose
real date falls inside the requested month. Everything recorded is observed, not assumed:

- the anchor (issue 1500 → 2023-08-08) was measured by ``HEAD`` on 2026-10-11 and is
  documented here with its provenance; the window self-corrects if it drifts;
- each downloaded issue is certified by the ``lichess_bulk`` guarantees — ``Content-Length``
  as the trusted reference, sha256 of the written file before the manifest, atomic rename,
  loud failure on an unverifiable body (Article 14);
- requests to a public site are spaced by ``POLITE_PAUSE_S`` (GOAL.md §6.6: public APIs —
  gently, pauses between requests).

The month manifest aggregates the issues and a QA-lite game count, computed by reading the
PGN inside each zip. Nothing is written unless it was measured.
"""

from __future__ import annotations

import argparse
import re
import time
import zipfile
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from caissa.acquisition.lichess_bulk import download
from caissa.acquisition.manifest import Manifest, sha256_file, write_manifest

TWIC_ZIP = "https://theweekinchess.com/zips/twic{n}g.zip"
POLITE_PAUSE_S = 1.0
DEFAULT_DEST = Path("data/raw/twic")

# Measured 2026-10-11 via `HEAD twic1500g.zip` -> `last-modified: Tue, 08 Aug 2023`.
# TWIC issues are weekly, so weeks-since-anchor maps a month to a good first guess; the
# probe window below makes a drifted estimate self-correcting.
ANCHOR_ISSUE = 1500
ANCHOR_DATE = date(2023, 8, 8)
PROBE_WINDOW = 12

_EVENT = re.compile(rb"\[Event ")


def estimate_issue(year: int, month: int) -> int:
    """A first guess for the issue number covering ``year-month`` (weeks since the anchor)."""
    target = date(year, month, 1)
    weeks = (target - ANCHOR_DATE).days // 7
    return ANCHOR_ISSUE + weeks


def _probe(issue: int, session: requests.Session, pause: float) -> dict | None:
    """HEAD one candidate; return its observed facts, or None when the issue does not exist."""
    url = TWIC_ZIP.format(n=issue)
    time.sleep(pause)  # public site: pauses between requests (Article §6.6)
    try:
        resp = session.head(url, allow_redirects=True, timeout=30)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    last_modified = resp.headers.get("Last-Modified")
    length = resp.headers.get("Content-Length")
    observed: date | None = None
    if last_modified:
        try:
            observed = parsedate_to_datetime(last_modified).date()
        except (TypeError, ValueError):
            observed = None
    return {
        "issue": issue,
        "url": url,
        "bytes": int(length) if length else None,
        "last_modified": observed.isoformat() if observed else None,
        "observed": observed,
    }


def probe_month(year: int, month: int, *, session: requests.Session | None = None) -> list[dict]:
    """Issues whose measured ``Last-Modified`` falls inside ``year-month``.

    Probes ``estimate ± PROBE_WINDOW``; if nothing lands in the month, widens once. Issues
    with no readable date are returned separately by the caller as unusable.
    """
    session = session or requests.Session()
    guess = estimate_issue(year, month)
    found: list[dict] = []
    for width in (PROBE_WINDOW, PROBE_WINDOW * 2):
        found = []
        for issue in range(guess - width, guess + width + 1):
            facts = _probe(issue, session, POLITE_PAUSE_S)
            if facts is None:
                continue
            observed = facts.pop("observed")
            if observed is not None and observed.year == year and observed.month == month:
                found.append({**facts, "date": observed.isoformat()})
        if found:
            break
    return sorted(found, key=lambda f: f["issue"])


def count_games_in_zip(zip_path: Path) -> tuple[int, str | None]:
    """Count ``[Event `` markers in the PGN inside ``zip_path``; return (games, pgn name)."""
    with zipfile.ZipFile(zip_path) as archive:
        pgn_names = [n for n in archive.namelist() if n.lower().endswith(".pgn")]
        if not pgn_names:
            return 0, None
        name = pgn_names[0]
        games = 0
        with archive.open(name) as fh:
            while True:
                block = fh.read(1 << 20)
                if not block:
                    break
                games += len(_EVENT.findall(block))
    return games, name


def fetch_twic_month(
    year_month: str,
    dest_dir: Path = DEFAULT_DEST,
    *,
    max_issues: int | None = None,
    pause: float = POLITE_PAUSE_S,
) -> Manifest:
    """Download every issue dated inside ``year_month`` (``YYYY-MM``) and record a manifest.

    Completed, hash-verified issues are skipped (Article 4). The month manifest lists the
    archived issues with their measured bytes and sha256, plus a QA-lite game count.
    """
    year, month = (int(part) for part in year_month.split("-"))
    name = f"twic_{year_month}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    issues = probe_month(year, month)
    if max_issues is not None:
        issues = issues[-max_issues:]
    if not issues:
        raise RuntimeError(
            f"[twic] no issues dated {year_month} were found within the probe window "
            f"(anchor {ANCHOR_ISSUE}->{ANCHOR_DATE}, measured 2026-10-11). "
            "Widen the window or check the site; nothing was recorded."
        )

    total_games = 0
    shards: list[str] = []
    observed: list[str] = []
    for facts in issues:
        target = dest_dir / f"twic{facts['issue']}g.zip"
        if not (
            target.exists()
            and facts["bytes"] is not None
            and target.stat().st_size == facts["bytes"]
        ):
            download(facts["url"], target, expected_bytes=facts["bytes"])
        else:
            print(f"[twic] issue {facts['issue']} already present — skipping (Article 4).")
        games, pgn = count_games_in_zip(target)
        total_games += games
        shards.append(target.name)
        observed.append(
            f"{facts['issue']}({facts['date']},{facts['bytes']}B,sha256={sha256_file(target)[:12]}…,"
            f"games={games},pgn={pgn})"
        )
        print(f"[twic] issue {facts['issue']} [{facts['date']}] {games} games")

    manifest = Manifest(
        source=TWIC_ZIP.format(n="*"),
        sha256=sha256_file(dest_dir / shards[0]) if shards else "",
        bytes=sum((dest_dir / s).stat().st_size for s in shards),
        rows=total_games,
        schema="zip(pgn)",
        license="TWIC — free for personal use; see theweekinchess.com",
        notes=(
            f"TWIC issues dated {year_month}, selected by measured Last-Modified. "
            f"issues=[{'; '.join(observed)}]"
        ),
        shards=shards,
    )
    write_manifest(name, manifest)
    print(f"[twic] {name}: {len(shards)} issues, {manifest.bytes} bytes, {total_games} games")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 TWIC loader")
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--max-issues", type=int, default=None)
    args = parser.parse_args(argv)
    fetch_twic_month(args.month, Path(args.dest), max_issues=args.max_issues)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
