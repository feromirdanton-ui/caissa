"""Resumable-download tests against a local HTTP fixture (BACKLOG issue #4 DoD).

These tests exist because the failure modes of a resumable downloader are silent: a server
that ignores `Range` yields a file larger than the resource that still looks finished, and a
corrupted local shard used to be trusted because the manifest was never re-verified.
Everything here runs on 127.0.0.1 and needs no network.
"""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from caissa.acquisition.lichess_bulk import IncompleteDownload, download

PAYLOAD = bytes(range(256)) * 1200  # 307_200 bytes


class _Handler(BaseHTTPRequestHandler):
    """Configurable file server: honest Range, Range-ignoring, short body, or lying length."""

    payload: bytes = PAYLOAD
    ignore_range: bool = False
    truncate: bool = False
    lie_about_length: bool = False
    head_ok: bool = True
    seen: list[tuple[str | None, int]] = []

    def do_HEAD(self) -> None:  # noqa: N802 (http.server API)
        cls = type(self)
        if not cls.head_ok:
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(cls.payload)))
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        cls = type(self)
        rng = self.headers.get("Range")
        payload = cls.payload

        if cls.ignore_range:
            # A server (or transparent proxy) that strips Range and always sends everything.
            cls.seen.append((rng, 200))
            body = payload
            sent = body[: len(body) // 2] if cls.truncate else body
            self.send_response(200)
            self.send_header(
                "Content-Length", str(len(body) if cls.lie_about_length else len(sent))
            )
            self.end_headers()
            self.wfile.write(sent)
            return

        start = 0
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
        if start >= len(payload):
            cls.seen.append((rng, 416))
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(payload)}")
            self.end_headers()
            return

        body = payload[start:]
        sent = body[: len(body) // 2] if cls.truncate else body
        status = 206 if rng else 200
        cls.seen.append((rng, status))
        self.send_response(status)
        self.send_header("Content-Length", str(len(body) if cls.lie_about_length else len(sent)))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
        self.end_headers()
        self.wfile.write(sent)

    def log_message(self, *args: object) -> None:  # keep the test output clean
        return


@pytest.fixture
def server():
    """Local file server factory; returns (handler_class, url)."""
    servers: list[ThreadingHTTPServer] = []

    def factory(
        *,
        ignore_range: bool = False,
        truncate: bool = False,
        lie_about_length: bool = False,
        head_ok: bool = True,
    ):
        cls = type(
            "Handler",
            (_Handler,),
            {
                "payload": PAYLOAD,
                "ignore_range": ignore_range,
                "truncate": truncate,
                "lie_about_length": lie_about_length,
                "head_ok": head_ok,
                "seen": [],
            },
        )
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), cls)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return cls, f"http://127.0.0.1:{httpd.server_address[1]}/shard.bin"

    yield factory
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def _sha() -> str:
    return hashlib.sha256(PAYLOAD).hexdigest()


def test_plain_download(tmp_path: Path, server) -> None:
    _, url = server()
    target = tmp_path / "shard.bin"
    download(url, target, chunk=1 << 16)
    assert target.read_bytes() == PAYLOAD
    assert hashlib.sha256(target.read_bytes()).hexdigest() == _sha()
    assert not (tmp_path / "shard.bin.part").exists()


def test_resume_requests_only_the_tail(tmp_path: Path, server) -> None:
    """kill -9 mid-download -> restart must re-fetch only the missing tail."""
    cls, url = server()
    target = tmp_path / "shard.bin"
    part = tmp_path / "shard.bin.part"
    head = 100_000
    part.write_bytes(PAYLOAD[:head])  # left over from the killed run

    download(url, target, chunk=1 << 16)

    assert cls.seen[0][0] == f"bytes={head}-", f"expected a Range request, saw {cls.seen[0]}"
    assert cls.seen[0][1] == 206, "the server must answer 206 for a resumed download"
    assert target.read_bytes() == PAYLOAD, "a resumed file must be byte-exact"
    assert hashlib.sha256(target.read_bytes()).hexdigest() == _sha()
    assert not part.exists(), "the .part file must be gone after a successful download"


def test_range_ignoring_server_does_not_corrupt(tmp_path: Path, server) -> None:
    """P0-2: a server that ignores Range must NOT be appended to the partial file."""
    cls, url = server(ignore_range=True)
    target = tmp_path / "shard.bin"
    part = tmp_path / "shard.bin.part"
    part.write_bytes(PAYLOAD[:100_000])

    download(url, target, chunk=1 << 16)

    assert target.stat().st_size == len(PAYLOAD), "an oversized corrupt file was published"
    assert target.read_bytes() == PAYLOAD
    assert hashlib.sha256(target.read_bytes()).hexdigest() == _sha()
    assert cls.seen[0][0] == "bytes=100000-", "the first attempt should still try to resume"
    assert any(status == 200 for _, status in cls.seen), "the Range-ignoring server was reached"


def test_stale_part_is_discarded_after_416(tmp_path: Path, server) -> None:
    """A stale .part must not deadlock the shard forever (416 -> discard and restart)."""
    cls, url = server(head_ok=False)  # no advertised length, so the 416 path is reached
    target = tmp_path / "shard.bin"
    part = tmp_path / "shard.bin.part"
    part.write_bytes(b"x" * (len(PAYLOAD) + 500))

    download(url, target, chunk=1 << 16)

    assert any(status == 416 for _, status in cls.seen), "the stale part should provoke a 416"
    assert target.read_bytes() == PAYLOAD
    assert not part.exists()


def test_truncated_body_is_not_published(tmp_path: Path, server) -> None:
    """A short body must raise instead of being renamed into place as a complete shard.

    The server answers honestly for the short body it sends, so requests sees a *complete*
    response; only the size check against the advertised length can catch it.
    """
    _, url = server(truncate=True)
    target = tmp_path / "shard.bin"
    part = tmp_path / "shard.bin.part"

    with pytest.raises(IncompleteDownload):
        download(url, target, chunk=1 << 16)

    assert not target.exists(), "an incomplete file must never be published"
    assert part.exists(), "the .part file is kept so the next run can resume"
    assert part.stat().st_size < len(PAYLOAD)


def test_connection_cut_is_not_published(tmp_path: Path, server) -> None:
    """A dropped connection mid-body must leave no published file and a resumable .part."""
    _, url = server(truncate=True, lie_about_length=True)
    target = tmp_path / "shard.bin"
    part = tmp_path / "shard.bin.part"

    with pytest.raises(requests.RequestException):
        download(url, target, chunk=1 << 16)

    assert not target.exists(), "a cut connection must never publish a file"
    assert part.exists(), "the .part file is kept so the next run can resume"
    assert part.stat().st_size < len(PAYLOAD)


def test_fetch_month_reverifies_hash_and_redownloads(tmp_path: Path, server, monkeypatch) -> None:
    """A corrupted local shard must be re-downloaded, not trusted (Article 5)."""
    from caissa.acquisition import lichess_bulk

    _, url = server()
    monkeypatch.setattr(lichess_bulk, "LICHESS_STANDARD", url + "?month={month}")
    # the manifest dir is isolated per-test by the autouse fixture in conftest.py

    dest = tmp_path / "raw"
    first = lichess_bulk.fetch_month("2013-01", dest)
    assert first.sha256 == _sha()
    assert first.fetched_at, "fetched_at must be populated (Article 8)"

    local = dest / "lichess_db_standard_rated_2013-01.pgn.zst"
    local.write_bytes(b"corrupted")

    second = lichess_bulk.fetch_month("2013-01", dest)
    assert local.read_bytes() == PAYLOAD, "a hash mismatch must trigger a re-download"
    assert second.sha256 == _sha()

    # a third run is a genuine no-op
    before = local.stat().st_mtime_ns
    lichess_bulk.fetch_month("2013-01", dest)
    assert local.stat().st_mtime_ns == before
