"""The outbound identity as a standalone regression (session-0005).

The Week In Chess answers the default ``python-requests`` User-Agent with HTTP 406, so every
request the loaders sent — the HEAD probe and the GET download alike — was refused, while the
local fixture (like any static file server) answered every client and the build stayed green.
This file asserts the half of that lesson a fixture can check: every request the loaders send
carries an identifying ``User-Agent``. Whether a given site *accepts* the identity can only be
proven on the wire (the runner, or the ``network`` test below) — see ADR-0008.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from caissa.acquisition import lichess_bulk, twic
from caissa.acquisition.http import USER_AGENT
from caissa.acquisition.lichess_bulk import download_prefix

BODY = bytes((i * 7) % 256 for i in range(120_000))


class _Handler(BaseHTTPRequestHandler):
    """A static-ish server that records the User-Agent of every request it receives."""

    agents: list[str | None] = []

    def do_HEAD(self) -> None:  # noqa: N802 (http.server API)
        type(self).agents.append(self.headers.get("User-Agent"))
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.send_header("Last-Modified", "Fri, 04 Nov 2022 16:10:48 GMT")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        type(self).agents.append(self.headers.get("User-Agent"))
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        tail = BODY[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(tail)))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
        self.end_headers()
        self.wfile.write(tail)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture
def recording_server():
    cls = type("Handler", (_Handler,), {"agents": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield cls, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_every_outbound_request_identifies_itself(tmp_path: Path, recording_server, monkeypatch):
    cls, base = recording_server
    lichess_bulk.download(base + "/shard.bin", tmp_path / "shard.bin")  # HEAD + GET
    download_prefix(base + "/prefix.bin", tmp_path / "prefix.bin", 100_000)  # GET
    monkeypatch.setattr(twic, "TWIC_ZIP", base + "/twic.zip")
    twic._probe(1500, requests.Session(), 0.0)  # HEAD from a bare session

    assert cls.agents, "the fixture was never reached"
    assert all(
        agent == USER_AGENT for agent in cls.agents
    ), f"every outbound request must carry the User-Agent; saw {cls.agents}"


@pytest.mark.network
def test_twic_anchor_is_still_honest() -> None:
    """The TWIC anchor (issue 1500 -> 2023-08-08) must still be true on the live site.

    Deselected by default (``-m "not network"``). Run it when the anchor is in doubt: if the
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
