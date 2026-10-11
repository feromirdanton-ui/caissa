"""Ocean loaders against a local HTTP fixture — no network in CI.

Covers the three Phase 1 sources and, crucially, the guarantees that a downloader's failure
modes are silent (§3 Article 4, Article 14):

- ``puzzles`` — download + one-pass scan + sample + manifest certifying the written file;
- ``twic`` — month discovery by *measured* ``Last-Modified``, not a guessed number;
- every outbound request carries an identifying ``User-Agent`` — the site answers the
  default ``python-requests`` one with HTTP 406 (the first clean-runner failure, session-0005);
- ``download_prefix`` — a resumable partial download whose resume re-fetches only the tail,
  and which restarts cleanly when the server ignores ``Range``.
"""

from __future__ import annotations

import hashlib
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests
import zstandard as zstd

from caissa.acquisition import lichess_bulk, puzzles, twic
from caissa.acquisition.lichess_bulk import download_prefix

EIGHT_MB = 1 << 23


class _Handler(BaseHTTPRequestHandler):
    """Routes: path -> (body, Last-Modified). Optionally ignores ``Range``."""

    routes: dict[str, tuple[bytes, str]] = {}
    ignore_range: bool = False
    seen: list[tuple[str | None, int]] = []

    def _route(self) -> tuple[bytes, str] | None:
        return type(self).routes.get(urlsplit(self.path).path)

    def do_HEAD(self) -> None:  # noqa: N802 (http.server API)
        route = self._route()
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        body, last_modified = route
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Last-Modified", last_modified)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        cls = type(self)
        route = self._route()
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        body, _ = route
        rng = self.headers.get("Range")

        if cls.ignore_range:
            cls.seen.append((rng, 200))
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        start = 0
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
        if start >= len(body):
            cls.seen.append((rng, 416))
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(body)}")
            self.end_headers()
            return
        tail = body[start:]
        status = 206 if rng else 200
        cls.seen.append((rng, status))
        self.send_response(status)
        self.send_header("Content-Length", str(len(tail)))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        self.end_headers()
        self.wfile.write(tail)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture
def server():
    """Factory: routes -> (handler_class, base_url). Servers are shut down after the test."""
    servers: list[ThreadingHTTPServer] = []

    def factory(routes: dict[str, tuple[bytes, str]], *, ignore_range: bool = False):
        cls = type(
            "Handler",
            (_Handler,),
            {"routes": routes, "ignore_range": ignore_range, "seen": []},
        )
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), cls)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return cls, f"http://127.0.0.1:{httpd.server_address[1]}"

    yield factory
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(autouse=True)
def _isolate_manifests(tmp_path_factory, monkeypatch):
    """Keep every manifest this file writes inside a temp dir (Article 14)."""
    from caissa.acquisition import manifest as manifest_module

    isolated = tmp_path_factory.mktemp("ocean-manifests")
    monkeypatch.setattr(manifest_module, "MANIFEST_DIR", isolated)
    return isolated


# --- puzzles ------------------------------------------------------------------


def _puzzle_csv(rows: int) -> bytes:
    lines = [puzzles.HEADER]
    for i in range(rows):
        rating = 800 + (i % 20) * 100
        lines.append(f"p{i:05d},8/8/8/8/8/8/8/K6k,e2e4,{rating},80,90,100,https://x/{i},fork,open")
    return zstd.ZstdCompressor().compress(("\n".join(lines) + "\n").encode())


def test_puzzles_download_samples_and_certifies(tmp_path: Path, server) -> None:
    payload = _puzzle_csv(50)
    _, base = server({"/puzzles.csv.zst": (payload, "Fri, 02 Oct 2026 08:51:45 GMT")})

    import caissa.acquisition.puzzles as pz

    orig = pz.PUZZLES_URL
    pz.PUZZLES_URL = base + "/puzzles.csv.zst"
    try:
        dest = tmp_path / "raw"
        manifest = pz.fetch_puzzles(dest, limit=10)
    finally:
        pz.PUZZLES_URL = orig

    archive = dest / "lichess_db_puzzle.csv.zst"
    assert archive.read_bytes() == payload
    assert manifest.sha256 == hashlib.sha256(payload).hexdigest()
    assert manifest.rows == 50, "the scan pass must count every row, not just the sample"
    sample = (dest / "puzzles_sample.csv").read_text().splitlines()
    assert sample[0] == puzzles.HEADER
    assert len(sample) == 11, "header + 10 sampled rows"


def test_puzzles_reverify_skips_a_complete_archive(tmp_path: Path, server) -> None:
    payload = _puzzle_csv(20)
    cls, base = server({"/puzzles.csv.zst": (payload, "Fri, 02 Oct 2026 08:51:45 GMT")})

    import caissa.acquisition.puzzles as pz

    orig = pz.PUZZLES_URL
    pz.PUZZLES_URL = base + "/puzzles.csv.zst"
    try:
        dest = tmp_path / "raw"
        pz.fetch_puzzles(dest, limit=5)
        hits_after_first = len(cls.seen)
        pz.fetch_puzzles(dest, limit=5)  # second run: verified complete -> no download
    finally:
        pz.PUZZLES_URL = orig

    assert (
        len(cls.seen) == hits_after_first
    ), "a verified archive must not be re-downloaded (Article 4)"


def test_puzzles_record_nothing_when_the_server_will_not_measure(tmp_path: Path, server) -> None:
    """A body that cannot be certified must raise, and no manifest may be written."""
    payload = _puzzle_csv(5)
    _, base = server({"/puzzles.csv.zst": (payload, "Fri, 02 Oct 2026 08:51:45 GMT")})

    import caissa.acquisition.puzzles as pz

    orig = pz.PUZZLES_URL
    pz.PUZZLES_URL = base + "/puzzles.csv.zst"
    written: list[str] = []
    real_write = pz.write_manifest
    pz.write_manifest = lambda name, m: written.append(name) or real_write(name, m)
    try:
        # no HEAD length is impossible here (our server always sends one), so contradict it:
        with pytest.raises(lichess_bulk.UnverifiableDownload):
            pz.fetch_puzzles(tmp_path / "raw", limit=5, expected_bytes=len(payload) + 1)
    finally:
        pz.PUZZLES_URL = orig
        pz.write_manifest = real_write

    assert written == [], "a contradicting reference must never produce a manifest (Article 14)"


# --- twic ---------------------------------------------------------------------


def _twic_zip(games: int) -> bytes:
    import io

    buffer = io.BytesIO()
    pgn = "".join(
        f'[Event "Test {i}"]\n[White "A"]\n[Black "B"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n\n'
        for i in range(games)
    )
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("twic1500.pgn", pgn)
    return buffer.getvalue()


def test_twic_month_is_discovered_by_measured_date(tmp_path: Path, server, monkeypatch) -> None:
    payload = _twic_zip(3)
    _, base = server({"/zips/twic1500g.zip": (payload, "Mon, 08 Jan 2024 00:00:00 GMT")})
    monkeypatch.setattr(twic, "TWIC_ZIP", base + "/zips/twic{n}g.zip")
    # make the estimate land on 1500 for 2024-01, so the probe finds the served issue
    monkeypatch.setattr(twic, "ANCHOR_ISSUE", 1500)
    monkeypatch.setattr(twic, "ANCHOR_DATE", __import__("datetime").date(2024, 1, 1))

    manifest = twic.fetch_twic_month("2024-01", tmp_path / "twic", pause=0.0)

    assert manifest.shards == ["twic1500g.zip"], "only issues dated inside the month are kept"
    assert manifest.rows == 3, "the QA-lite game count must come from the PGN, not a guess"
    assert (tmp_path / "twic" / "twic1500g.zip").exists()
    assert manifest.sha256 == hashlib.sha256(payload).hexdigest()


def test_twic_records_nothing_when_no_issue_matches(tmp_path: Path, server, monkeypatch) -> None:
    _, base = server({"/zips/twic1500g.zip": (b"x", "Mon, 08 Jan 2024 00:00:00 GMT")})
    monkeypatch.setattr(twic, "TWIC_ZIP", base + "/zips/twic{n}g.zip")
    monkeypatch.setattr(twic, "PROBE_WINDOW", 1)

    import datetime

    monkeypatch.setattr(twic, "ANCHOR_ISSUE", 1500)
    monkeypatch.setattr(twic, "ANCHOR_DATE", datetime.date(2030, 1, 1))  # estimate far away

    with pytest.raises(RuntimeError):
        twic.fetch_twic_month("2024-01", tmp_path / "twic", pause=0.0)


# --- download_prefix ----------------------------------------------------------


def _payload(size: int = 300_000) -> bytes:
    return bytes((i * 7) % 256 for i in range(size))


def test_prefix_download_is_exact_and_certified(tmp_path: Path, server) -> None:
    payload = _payload()
    cls, base = server({"/month.pgn.zst": (payload, "Fri, 04 Nov 2022 16:10:48 GMT")})
    target = tmp_path / "day.pgn.zst"

    result = download_prefix(base + "/month.pgn.zst", target, 100_000)

    assert target.stat().st_size == 100_000
    assert result.sha256 == hashlib.sha256(payload[:100_000]).hexdigest()
    assert cls.seen[0] == (
        "bytes=0-99999",
        206,
    ), "a bounded prefix is an explicit Range request, so an honest server answers 206"
    assert result.remote_total == len(payload), "the server's real size is reported, never assumed"


def test_prefix_resume_refetches_only_the_tail(tmp_path: Path, server) -> None:
    payload = _payload()
    cls, base = server({"/month.pgn.zst": (payload, "Fri, 04 Nov 2022 16:10:48 GMT")})
    target = tmp_path / "day.pgn.zst"
    part = tmp_path / "day.pgn.zst.part"
    part.write_bytes(payload[:40_000])  # a killed run

    result = download_prefix(base + "/month.pgn.zst", target, 100_000)

    assert cls.seen[0] == ("bytes=40000-99999", 206), f"expected a tail Range, saw {cls.seen[0]}"
    assert result.resumed_from == 40_000
    assert result.fetched_bytes == 60_000, "only the missing tail is fetched"
    assert target.read_bytes() == payload[:100_000]
    assert not part.exists(), "the .part must be gone after an atomic rename"


def test_prefix_restarts_when_the_server_ignores_range(tmp_path: Path, server) -> None:
    payload = _payload()
    cls, base = server(
        {"/month.pgn.zst": (payload, "Fri, 04 Nov 2022 16:10:48 GMT")}, ignore_range=True
    )
    target = tmp_path / "day.pgn.zst"
    part = tmp_path / "day.pgn.zst.part"
    part.write_bytes(payload[:40_000])

    result = download_prefix(base + "/month.pgn.zst", target, 100_000)

    assert any(status == 200 for _, status in cls.seen), "the Range-ignoring server was reached"
    assert target.stat().st_size == 100_000, "an oversized corrupt file must not be published"
    assert target.read_bytes() == payload[:100_000]
    assert result.sha256 == hashlib.sha256(payload[:100_000]).hexdigest()


# --- outbound identity (session-0005: the clean-runner 406) --------------------


def test_every_outbound_request_identifies_itself(tmp_path: Path, server, monkeypatch) -> None:
    """A request without a User-Agent is what the TWIC site answers with HTTP 406.

    The local fixture, like a static file server, answers every client, so this stayed
    invisible for the whole build and only surfaced on the first clean-runner run. The
    regression is cheap: assert that every request the loaders send carries the identity.
    """
    from caissa.acquisition.http import USER_AGENT

    class _RecordingHandler(_Handler):
        agents: list[str | None] = []

        def do_HEAD(self) -> None:  # noqa: N802 (http.server API)
            type(self).agents.append(self.headers.get("User-Agent"))
            super().do_HEAD()

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            type(self).agents.append(self.headers.get("User-Agent"))
            super().do_GET()

    body = _payload(120_000)
    cls = type(
        "Handler",
        (_RecordingHandler,),
        {
            "routes": {
                "/shard.bin": (body, "Fri, 04 Nov 2022 16:10:48 GMT"),
                "/prefix.bin": (body, "Fri, 04 Nov 2022 16:10:48 GMT"),
                "/twic.zip": (body, "Mon, 08 Jan 2024 00:00:00 GMT"),
            },
            "ignore_range": False,
            "seen": [],
            "agents": [],
        },
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        lichess_bulk.download(base + "/shard.bin", tmp_path / "shard.bin")  # HEAD + GET
        download_prefix(base + "/prefix.bin", tmp_path / "prefix.bin", 100_000)  # GET
        monkeypatch.setattr(twic, "TWIC_ZIP", base + "/twic.zip")
        twic._probe(1500, requests.Session(), 0.0)  # HEAD, a bare session
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert cls.agents, "the fixture was never reached"
    assert all(agent == USER_AGENT for agent in cls.agents), (
        f"every outbound request must carry the User-Agent; saw {cls.agents}"
    )


@pytest.mark.network
def test_twic_anchor_is_still_honest() -> None:
    """The TWIC anchor (issue 1500 -> 2023-08-08) must still be true on the live site.

    Deselected by default (`-m "not network"`). Run it when the anchor is in doubt: if the
    issue's real Last-Modified has moved, update ANCHOR_DATE rather than quietly widening the
    window, so the constant stays a measurement and not a memory (ADR-0006).
    """
    import datetime

    from caissa.acquisition.twic import ANCHOR_ISSUE, _probe

    facts = _probe(ANCHOR_ISSUE, requests.Session(), 0.0)
    assert facts is not None, f"issue {ANCHOR_ISSUE} is not served anymore"
    assert facts["last_modified"], "the site stopped sending Last-Modified; discovery broke"
    assert datetime.date.fromisoformat(facts["last_modified"]) == datetime.date(2023, 8, 8), (
        f"the anchor drifted: issue {ANCHOR_ISSUE} is now {facts['last_modified']}; "
        "update ANCHOR_DATE in twic.py (and this test)"
    )
