"""Background queue for the long replay sweeps (PLAN items 2 and 3). Meant to run detached:

  python src/launch_detached.py results/logs/long_queue.log -- python -u src/run_queue.py

Every 30 s it looks at the plan below, starts jobs whose channel and noise files carry a verified marker in
$UWA_LIBRARY_ROOT/verified/ (written by fetch_library.py), keeps at most MAX_RUNNING jobs and an
estimated MAX_MEM_GB of channel data in flight, records exits, retries a failed job once, and rewrites the
status block of results/BACKGROUND-STATUS.md between the AUTO markers. A job whose result JSON already exists
is marked done and not rerun, so restarting this script after a crash or reboot resumes where it stopped.
Jobs whose files never arrive are marked blocked once both download streams have finished.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
SRC = os.path.join(ROOT, "src")
DATA = os.environ.get("UWA_LIBRARY_ROOT", "N:/Datasets/UWChannelLibrary")
MARKERS = os.path.join(DATA, "verified")
LOGS = os.path.join(ROOT, "results", "logs")
OUT = os.path.join(ROOT, "results", "long")
STATUS_JSON = os.path.join(LOGS, "long_queue_status.json")
STATUS_MD = os.path.join(ROOT, "results", "BACKGROUND-STATUS.md")
FETCH_LOGS = [os.path.join(LOGS, "fetch_stream_A.log"), os.path.join(LOGS, "fetch_stream_B.log")]
MAX_RUNNING = 16
MAX_MEM_GB = 20.0  # sum of per-job estimates in flight; measured: Green chunks peak near 2 GB, then 0.3 GB
MIN_FREE_GB = 4.0  # do not start a job when the machine has less physical memory free than this
POLL_S = 30


def noise_file(site: str, file_key: str) -> str:
    from psr_replay_long import noise_file as nf
    return nf(site, file_key)


def build_plan() -> list[dict]:
    """Ordered job list. Earlier entries start first when their files are verified."""
    jobs = []

    def add(site, file_key, rate, elements=None):
        el = f"_e{elements[0]:02d}-{elements[1]:02d}" if elements else ""
        name = f"{site}_{file_key}{el}_r{rate}"
        jobs.append(dict(name=name, site=site, file=file_key, rate=rate, elements=elements,
                         out=os.path.join(OUT, f"psr_{name}.json"), log=os.path.join(LOGS, f"long_psr_{name}.log"),
                         needs=[f"{file_key}.mat", noise_file(site, file_key)]))

    # Item 2: 160 and 320 chips/s through the files on disk first.
    for r in (160, 320):
        add("blue", "blue_1", r)
        add("red", "red_1", r)
    for r in (160, 320):
        for a in range(0, 64, 8):
            add("green", "green", r, (a, a + 7))
    for r in (160, 320):
        for k in range(2, 21):
            add("blue", f"blue_{k}", r)
        for k in range(2, 5):
            add("red", f"red_{k}", r)
    # Item 2 and 3: Yellow at 160 and 320, then the rest of the ladder on Yellow.
    for r in (160, 320):
        for k in range(1, 7):
            add("yellow", f"yellow_{k}", r)
    for r in (640, 1920):
        for k in range(1, 7):
            add("yellow", f"yellow_{k}", r)
    # Item 3: Purple, then Pink if time allows (last in the queue).
    for r in (160, 320, 640, 1920):
        for k in range(1, 16):
            add("purple", f"purple_{k}", r)
    for r in (160, 320, 640, 1920):
        for k in range(1, 4):
            add("pink", f"pink_{k}", r)
    return jobs


def file_gb(name: str) -> float:
    p = os.path.join(DATA, name)
    return os.path.getsize(p) / 1e9 if os.path.exists(p) else 0.5


def est_mem_gb(job: dict) -> float:
    return 0.3 + 1.0 * file_gb(f"{job['file']}.mat")


def free_gb() -> float:
    import ctypes

    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    s = MemStatus()
    s.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 1e9


def verified(name: str) -> bool:
    return os.path.exists(os.path.join(MARKERS, f"{name}.md5"))


def fetch_streams_done() -> bool:
    for fn in FETCH_LOGS:
        if not os.path.exists(fn):
            return False
        with open(fn, encoding="utf-8", errors="replace") as fh:
            if "fetch_library done" not in fh.read():
                return False
    return True


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(now(), msg, flush=True)


def write_status(jobs: list[dict], running: dict) -> None:
    counts = {}
    for j in jobs:
        counts[j["state"]] = counts.get(j["state"], 0) + 1
    state = dict(updated=now(), counts=counts, max_running=MAX_RUNNING, max_mem_gb=MAX_MEM_GB,
                 jobs=[{k: v for k, v in j.items() if k != "proc"} for j in jobs])
    tmp = STATUS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, STATUS_JSON)

    lines = [f"Queue status at {state['updated']} (rewritten by src/run_queue.py every {POLL_S} s while it runs).", "",
             "| state | jobs |", "|---|---|"]
    for k in ("done", "running", "pending", "waiting_download", "failed", "blocked"):
        lines.append(f"| {k} | {counts.get(k, 0)} |")
    lines.append("")
    if counts.get("running"):
        lines.append("Running: " + ", ".join(j["name"] for j in jobs if j["state"] == "running"))
        lines.append("")
    if counts.get("failed") or counts.get("blocked"):
        lines.append("| job | state | log | note |")
        lines.append("|---|---|---|---|")
        for j in jobs:
            if j["state"] in ("failed", "blocked"):
                lines.append(f"| {j['name']} | {j['state']} | {j['log']} | {j.get('note', '')} |")
        lines.append("")
    lines.append("| job | state | started | finished | result |")
    lines.append("|---|---|---|---|---|")
    for j in jobs:
        lines.append(f"| {j['name']} | {j['state']} | {j.get('started', '')} | {j.get('finished', '')} | "
                     f"{os.path.relpath(j['out'], ROOT) if j['state'] == 'done' else ''} |")
    block = "\n".join(lines) + "\n"
    if os.path.exists(STATUS_MD):
        with open(STATUS_MD, encoding="utf-8") as fh:
            md = fh.read()
        a, b = "<!-- AUTO-STATUS-BEGIN -->", "<!-- AUTO-STATUS-END -->"
        if a in md and b in md:
            head, rest = md.split(a, 1)
            _, tail = rest.split(b, 1)
            md = head + a + "\n" + block + b + tail
            tmp = STATUS_MD + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(md)
            os.replace(tmp, STATUS_MD)


class AdoptedProcess:
    """Poll a job started by an earlier run of this script, by PID, with the same interface as Popen.poll()."""

    STILL_ACTIVE = 259

    def __init__(self, pid: int):
        import ctypes
        self.pid = pid
        self.k32 = ctypes.windll.kernel32
        self.h = self.k32.OpenProcess(0x1000 | 0x100000, False, pid)  # QUERY_LIMITED_INFORMATION | SYNCHRONIZE

    def poll(self):
        import ctypes
        if not self.h:
            return 1
        code = ctypes.c_ulong()
        if not self.k32.GetExitCodeProcess(self.h, ctypes.byref(code)) or code.value == self.STILL_ACTIVE:
            return None
        self.k32.CloseHandle(self.h)
        self.h = None
        return int(code.value)


def adopt_running(jobs: list[dict], running: dict) -> None:
    """Pick up jobs an earlier runner left running (their PIDs are in the status file) instead of relaunching."""
    if not os.path.exists(STATUS_JSON):
        return
    with open(STATUS_JSON, encoding="utf-8") as fh:
        old = {j["name"]: j for j in json.load(fh).get("jobs", [])}
    for j in jobs:
        o = old.get(j["name"])
        if not o or o.get("state") != "running" or j["state"] == "done":
            continue
        p = AdoptedProcess(int(o["pid"]))
        if p.poll() is None:
            running[j["name"]] = p
            j.update(state="running", pid=o["pid"], started=o.get("started", ""), attempts=o.get("attempts", 1))
            log(f"adopted {j['name']} pid {o['pid']} (still running)")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    jobs = build_plan()
    for j in jobs:
        j["state"] = "done" if os.path.exists(j["out"]) else "pending"
        j["attempts"] = 0
    with open(os.path.join(LOGS, "long_jobs.txt"), "w", encoding="utf-8") as fh:
        for j in jobs:
            fh.write(f"{j['site']} {j['file']} {j['rate']} {j['elements'] or ''}\n".rstrip() + "\n")
    log(f"queue start: {len(jobs)} jobs, {sum(j['state'] == 'done' for j in jobs)} already done")
    running: dict[str, subprocess.Popen] = {}
    adopt_running(jobs, running)
    write_status(jobs, running)
    while True:
        changed = False
        for j in jobs:
            if j["state"] == "running":
                p = running[j["name"]]
                rc = p.poll()
                if rc is None:
                    continue
                del running[j["name"]]
                j["finished"] = now()
                if rc == 0 and os.path.exists(j["out"]):
                    j["state"] = "done"
                    log(f"done {j['name']}")
                elif j["attempts"] < 2:
                    j["state"] = "pending"
                    j["note"] = f"exit {rc} on attempt {j['attempts']}, retrying"
                    log(f"retry {j['name']} after exit {rc}")
                else:
                    j["state"] = "failed"
                    j["note"] = f"exit {rc} after {j['attempts']} attempts"
                    log(f"FAILED {j['name']} exit {rc}, log {j['log']}")
                changed = True
        mem = sum(est_mem_gb(j) for j in jobs if j["state"] == "running")
        streams_done = fetch_streams_done()
        for j in jobs:
            if j["state"] not in ("pending", "waiting_download"):
                continue
            if not all(verified(n) for n in j["needs"]):
                if streams_done:
                    j["state"] = "blocked"
                    j["note"] = "download of " + ", ".join(n for n in j["needs"] if not verified(n)) + " failed"
                    log(f"BLOCKED {j['name']}: {j['note']}")
                    changed = True
                elif j["state"] != "waiting_download":
                    j["state"] = "waiting_download"
                    changed = True
                continue
            if len(running) >= MAX_RUNNING:
                continue
            need = est_mem_gb(j)
            if running and (mem + need > MAX_MEM_GB or free_gb() < MIN_FREE_GB + need):
                continue
            cmd = [PY, "-u", os.path.join(SRC, "psr_replay_long.py"), j["site"], j["file"], str(j["rate"])]
            if j["elements"]:
                cmd += ["--elements", f"{j['elements'][0]}-{j['elements'][1]}"]
            if os.path.exists(j["log"]):
                os.replace(j["log"], j["log"] + f".attempt{j['attempts']}")
            fh = open(j["log"], "a", encoding="utf-8", buffering=1)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            # No DETACHED_PROCESS: see launch_detached.py for why (one Windows Terminal window per job otherwise).
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                 creationflags=flags, close_fds=True)
            running[j["name"]] = p
            j["state"] = "running"
            j["attempts"] += 1
            j["pid"] = p.pid
            j["started"] = now()
            j.pop("finished", None)
            mem += need
            log(f"start {j['name']} pid {p.pid} (running {len(running)}, est mem {mem:.1f} GB)")
            changed = True
        if changed:
            write_status(jobs, running)
        if not running and all(j["state"] in ("done", "failed", "blocked") for j in jobs):
            write_status(jobs, running)
            log("queue finished: " + json.dumps({s: sum(j['state'] == s for j in jobs) for s in ('done', 'failed', 'blocked')}))
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
