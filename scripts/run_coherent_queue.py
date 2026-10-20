"""Second coherent wave: the ceiling scheme and the one below it on every file of the multi-file sites, so the
coherent ceilings are pooled over the same records as the FH-BFSK ones. Launches psr_coherent.py jobs through
launch_detached.py, at most MAX_JOBS at a time, and exits when every job has a result or has failed.

Usage: python -u src/run_coherent_queue.py
Logs: results/logs/coh2_<file>.log; status line printed every poll.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_detached import launch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
MAX_JOBS = 14
# Wave 2 (2026-10-19, after the receiver revision): the file-1 winners of the first sweep on every file.
# Wave 3: the file-1 winners of the revised receiver, which differ for Blue and Yellow (Blue 4000 BPSK and 2000 QPSK
# instead of 4000 QPSK and 2000 BPSK; Yellow 4000 QPSK and 2000 QPSK), plus Purple's 1000 BPSK. The runner skips any
# job whose result file exists, so the list is cumulative.
WAVE = [("blue", [f"blue_{k}" for k in range(2, 21)], "4000q,2000b"),
        ("red", [f"red_{k}" for k in range(2, 5)], "4000b,2000b"),
        ("yellow", [f"yellow_{k}" for k in range(2, 7)], "4000b,2000b"),
        ("purple", [f"purple_{k}" for k in range(2, 16)], "2000b"),
        ("blue", [f"blue_{k}" for k in range(2, 21)], "4000b,2000q"),
        ("yellow", [f"yellow_{k}" for k in range(2, 7)], "4000q,2000q"),
        ("purple", [f"purple_{k}" for k in range(2, 16)], "1000b")]


def expected(site: str, fkey: str, schemes: str) -> list[str]:
    out = []
    for tok in schemes.split(","):
        mod = {"b": "bpsk", "q": "qpsk"}[tok[-1]]
        out.append(os.path.join(ROOT, "results", "coherent", f"psr_coh_{site}_{fkey}_{tok[:-1]}_{mod}.json"))
    return out


def alive(pid: int) -> bool:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue) -ne $null"],
                       capture_output=True, text=True)
    return "True" in r.stdout


def main() -> None:
    jobs = [(site, fkey, schemes) for site, files, schemes in WAVE for fkey in files]
    pending = [j for j in jobs if not all(os.path.exists(p) for p in expected(*j))]
    running: dict[tuple, int] = {}
    print(f"{len(jobs)} jobs, {len(pending)} pending", flush=True)
    while pending or running:
        for j, pid in list(running.items()):
            if not alive(pid):
                done = all(os.path.exists(p) for p in expected(*j))
                print(time.strftime("%H:%M:%S"), "finished" if done else "FAILED", j[1], flush=True)
                del running[j]
        while pending and len(running) < MAX_JOBS:
            j = pending.pop(0)
            log = os.path.join(ROOT, "results", "logs", f"coh2_{j[1]}.log")
            pid = launch(log, [PY, "-u", os.path.join(ROOT, "src", "psr_coherent.py"), j[0], j[1], "--schemes", j[2]])
            running[j] = pid
            print(time.strftime("%H:%M:%S"), "started", j[1], "pid", pid, flush=True)
        time.sleep(30)
    n = len(glob.glob(os.path.join(ROOT, "results", "coherent", "psr_coh_*.json")))
    print("queue finished;", n, "coherent result files", flush=True)


if __name__ == "__main__":
    main()
