#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "metadata/output_checksums_sha256.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


paths = sorted((ROOT / "outputs/figures").glob("*.png")) + sorted((ROOT / "outputs/tables").glob("*.csv"))
if len(paths) != 10:
    raise SystemExit(f"Refusing to update release checksums: expected 10 manuscript outputs, observed {len(paths)}")

with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["relative_path", "sha256"])
    writer.writeheader()
    for path in paths:
        writer.writerow({"relative_path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)})

print(f"Updated immutable release checksums for {len(paths)} outputs: {OUTPUT.relative_to(ROOT)}")
