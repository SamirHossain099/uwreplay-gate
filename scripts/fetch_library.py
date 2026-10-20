"""Fetch files of the Underwater Acoustic Channel Library from Zenodo and verify their MD5 digests.

Usage: python -u src/fetch_library.py <name> [<name> ...]
Names are file keys of record 10.5281/zenodo.21287414 (for example yellow_1.mat); a bare site word such as
"yellow" expands to every channel and noise file of that site. Sizes, digests and download URLs come from
$UWA_LIBRARY_ROOT/zenodo_record_21287414.json, saved from the record API on 2026-10-18.

For each file: skip when a verified marker exists; otherwise download with curl (resumable, retried), compare
size and MD5 with the record, write $UWA_LIBRARY_ROOT/verified/<name>.md5 on success, and append one
line to $UWA_LIBRARY_ROOT/download.log either way. Exit status is the number of failures.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

DATA = os.environ.get("UWA_LIBRARY_ROOT", "N:/Datasets/UWChannelLibrary")
RECORD = os.path.join(DATA, "zenodo_record_21287414.json")
MARKERS = os.path.join(DATA, "verified")
LOG = os.path.join(DATA, "download.log")
CURL = "C:/WINDOWS/system32/curl.exe"


def stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(line: str) -> None:
    print(stamp(), line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp()} {line}\n")


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record_files() -> dict:
    with open(RECORD, encoding="utf-8") as fh:
        rec = json.load(fh)
    out = {}
    for f in rec["files"]:
        out[f["key"]] = dict(size=int(f["size"]), md5=f["checksum"].split(":", 1)[1], url=f["links"]["self"])
    return out


def expand(names: list[str], files: dict) -> list[str]:
    out = []
    for n in names:
        if n in files:
            out.append(n)
        else:
            site = [k for k in files if k == f"{n}.mat" or k.startswith(f"{n}_")]
            if not site:
                raise SystemExit(f"unknown file or site: {n}")
            out.extend(sorted(site, key=lambda k: (not k.endswith("noise.mat") and "noise" not in k, k)))
    return list(dict.fromkeys(out))


def verify(name: str, info: dict) -> bool:
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size != info["size"]:
        log(f"{name} SIZE MISMATCH on disk {size} record {info['size']}")
        return False
    digest = md5_of(path)
    if digest != info["md5"]:
        log(f"{name} MD5 MISMATCH on disk {digest} record {info['md5']}")
        return False
    with open(os.path.join(MARKERS, f"{name}.md5"), "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  {name}\n")
    log(f"{name} OK size {size} md5 {digest}")
    return True


def fetch(name: str, info: dict) -> bool:
    path = os.path.join(DATA, name)
    part = path + ".part"
    for attempt in range(1, 6):
        t0 = time.time()
        cmd = [CURL, "-L", "--fail", "--silent", "--show-error", "--retry", "10", "--retry-delay", "20",
               "--retry-all-errors", "-C", "-", "-o", part, info["url"]]
        log(f"{name} download attempt {attempt} ({info['size'] / 1e6:.1f} MB) from {info['url']}")
        rc = subprocess.call(cmd)
        if rc != 0:
            log(f"{name} curl exit {rc} after {time.time() - t0:.0f} s")
            time.sleep(60)
            continue
        if os.path.exists(part) and os.path.getsize(part) >= info["size"]:
            if os.path.exists(path):
                os.remove(path)
            os.replace(part, path)
            log(f"{name} downloaded in {time.time() - t0:.0f} s")
            if verify(name, info):
                return True
            os.remove(path)
        else:
            log(f"{name} partial file {os.path.getsize(part) if os.path.exists(part) else 0} bytes, retrying")
        time.sleep(60)
    log(f"{name} FAILED after 5 attempts")
    return False


def main(names: list[str]) -> int:
    os.makedirs(MARKERS, exist_ok=True)
    files = record_files()
    todo = expand(names, files)
    log(f"fetch_library start: {len(todo)} files: {' '.join(todo)}")
    failures = 0
    for name in todo:
        if os.path.exists(os.path.join(MARKERS, f"{name}.md5")):
            log(f"{name} already verified, skipped")
            continue
        if os.path.exists(os.path.join(DATA, name)) and verify(name, files[name]):
            continue
        if not fetch(name, files[name]):
            failures += 1
    log(f"fetch_library done: {len(todo) - failures} ok, {failures} failed")
    return failures


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
