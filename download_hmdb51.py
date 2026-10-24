#!/usr/bin/env python3
"""Download and extract official HMDB51 videos and train/test splits."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import gdown

VIDEO_ID = "17anw5Oxp7lmp9cMwXPOyOpL5olDLmPpj"
SPLIT_ID = "1NQxJWJSYWefyNS-LFYCenFih4gRDUPCX"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--unrar", default=None, help="unrar executable (auto-detected by default)")
    args = ap.parse_args()
    root = args.root.resolve()
    dl, videos, splits = root / "downloads", root / "videos", root / "splits"
    for p in (dl, videos, splits):
        p.mkdir(parents=True, exist_ok=True)
    unrar = (args.unrar or shutil.which("unrar") or shutil.which("unrar-nonfree")
             or str(root / "tools/usr/bin/unrar-nonfree"))
    if not Path(unrar).exists() and shutil.which(unrar) is None:
        raise SystemExit("unrar not found; install it or pass --unrar PATH")

    timing = {"source": "official Serre Lab Google Drive", "download_seconds": 0.0,
              "extract_seconds": 0.0, "downloaded_files": []}
    targets = [(SPLIT_ID, dl / "test_train_splits.rar"), (VIDEO_ID, dl / "hmdb51_org.rar")]
    for file_id, path in targets:
        if path.exists() and path.stat().st_size > 10_000:
            continue
        t = time.perf_counter()
        result = gdown.download(id=file_id, output=str(path), quiet=False, resume=True)
        timing["download_seconds"] += time.perf_counter() - t
        if not result:
            raise SystemExit(f"Download failed: {path.name}")
        timing["downloaded_files"].append(path.name)

    t = time.perf_counter()
    if not any(splits.glob("*_test_split1.txt")):
        run([unrar, "x", "-o+", str(dl / "test_train_splits.rar"), str(splits) + "/"])
    nested = root / "nested_rars"
    nested.mkdir(exist_ok=True)
    if not any(nested.glob("*.rar")):
        run([unrar, "x", "-o+", str(dl / "hmdb51_org.rar"), str(nested) + "/"])
    wanted = {"walk", "run", "jump", "sit", "stand", "wave"}
    for archive in sorted(nested.rglob("*.rar")):
        cls = archive.stem
        if cls in wanted and not (videos / cls).exists():
            run([unrar, "x", "-o+", str(archive), str(videos) + "/"])
    timing["extract_seconds"] = time.perf_counter() - t
    timing["finished_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    (root / "download_extract_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()

