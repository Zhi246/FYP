#!/usr/bin/env python3
"""Download the six original KTH archives; resume downloads, never use mirrors."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import zipfile

BASE = "https://www.csc.kth.se/cvap/actions/"
CLASSES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if destination.exists():
        return {"url": url, "path": str(destination.resolve()), "bytes": destination.stat().st_size,
                "download_seconds": None, "reused_existing": True, "sha256": sha256(destination)}
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "FYP-KTH-research/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                resumed = response.status == 206 and offset > 0
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Server returned an unexpected resume offset")
                received = offset if resumed else 0
                expected = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
                transferred = 0
                last_progress = time.perf_counter()
                with partial.open("ab" if resumed else "wb") as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
                        received += len(block)
                        transferred += len(block)
                        if time.perf_counter() - last_progress > 10:
                            print(f"{destination.name}: {received / 1e6:.1f} MB", flush=True)
                            last_progress = time.perf_counter()
                if expected is not None and transferred != expected:
                    raise RuntimeError(f"Incomplete response: {transferred}/{expected} bytes")
            os.replace(partial, destination)
            print(f"Downloaded {destination.name}: {destination.stat().st_size / 1e6:.1f} MB", flush=True)
            return {"url": url, "path": str(destination.resolve()), "bytes": destination.stat().st_size,
                    "download_seconds": time.perf_counter() - started, "reused_existing": False,
                    "resumed_bytes": offset if resumed else 0, "sha256": sha256(destination)}
        except Exception as error:
            print(f"Download attempt {attempt + 1}: {destination.name}: {error}", flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


def prepare_class(root, cls):
    result = fetch(BASE + cls + ".zip", root / "archives" / (cls + ".zip"))
    started = time.perf_counter()
    directory = root / "videos" / cls
    directory.mkdir(parents=True, exist_ok=True)
    extracted, reused = 0, 0
    with zipfile.ZipFile(result["path"]) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Archive CRC failure: {bad}")
        names = set()
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".avi"):
                continue
            name = Path(member.filename).name
            if name in names or not name.startswith("person") or f"_{cls}_" not in name:
                raise RuntimeError(f"Unexpected or duplicate archive member: {member.filename}")
            names.add(name)
            path = directory / name
            if path.exists() and path.stat().st_size == member.file_size:
                reused += 1
                continue
            temporary = path.with_suffix(".avi.part")
            with archive.open(member) as source, temporary.open("wb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
            os.replace(temporary, path)
            extracted += 1
    result.update({"class": cls, "archive_crc_passed": True, "videos": extracted + reused,
                   "videos_extracted_this_run": extracted, "videos_reused": reused,
                   "validation_and_extraction_seconds": time.perf_counter() - started})
    print(f"Ready {cls}: {extracted + reused} videos", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/kth"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"source": BASE, "status": "RUNNING", "classes": [],
              "start_local": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    report_path = args.root / "download_extract_timing.json"
    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    try:
        report["annotations"] = fetch(BASE + "00sequences.txt", args.root / "00sequences.txt")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(prepare_class, args.root, cls) for cls in CLASSES]
            for future in concurrent.futures.as_completed(futures):
                report["classes"].append(future.result())
                save()
        report["status"] = "COMPLETE"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = repr(error)
        raise
    finally:
        report["wall_seconds_this_run"] = time.perf_counter() - started
        save()


if __name__ == "__main__":
    main()
