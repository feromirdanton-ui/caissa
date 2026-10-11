"""Data manifests — the only data that enters git (Article 8).

A manifest records: source, sha256, bytes, rows, schema, fetched_at, license, notes.
It is written at download time, never from memory (GOAL.md §7).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_DIR = Path("data/manifests")


@dataclass
class Manifest:
    source: str
    sha256: str = ""
    bytes: int = 0
    rows: int | None = None
    schema: str = ""
    fetched_at: str = ""
    license: str = ""
    notes: str = ""
    shards: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


@dataclass
class ManifestStore:
    root: Path = MANIFEST_DIR

    def path_for(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()

    def load(self, name: str) -> Manifest:
        data = json.loads(self.path_for(name).read_text())
        return Manifest(**data)

    def save(self, name: str, manifest: Manifest) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(name)
        path.write_text(manifest.to_json())
        return path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(name: str, manifest: Manifest) -> Path:
    if not manifest.fetched_at:
        manifest.fetched_at = datetime.now(timezone.utc).isoformat()
    return ManifestStore().save(name, manifest)


def load_manifest(name: str) -> Manifest:
    return ManifestStore().load(name)


def _status() -> int:
    store = ManifestStore()
    if not store.root.exists():
        print("[manifest] no manifests yet — run `make data-sample` (Phase 1).")
        return 0
    files = sorted(store.root.glob("*.json"))
    if not files:
        print("[manifest] no manifests yet — run `make data-sample` (Phase 1).")
        return 0
    for path in files:
        m = Manifest(**json.loads(path.read_text()))
        print(f"[manifest] {path.stem}: {m.bytes} bytes sha256={m.sha256[:12]}… rows={m.rows}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 data manifest status")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    if args.status:
        return _status()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
