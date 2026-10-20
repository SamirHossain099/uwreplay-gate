"""Start a command as a detached Windows process with stdout and stderr streamed to a log file.

Usage: python src/launch_detached.py <log_path> -- <command> [args ...]
Prints the PID. The child survives the launching shell. Environment: BLAS and OpenMP thread counts are pinned to
one so each replay job stays single-threaded.
"""
from __future__ import annotations

import os
import subprocess
import sys


def launch(log_path: str, cmd: list[str], env_extra: dict | None = None) -> int:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    if env_extra:
        env.update(env_extra)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    # CREATE_NO_WINDOW gives the child a hidden console that its own children inherit. DETACHED_PROCESS must not
    # be added: a console-less parent makes the venv's python.exe re-launcher allocate a fresh console for the
    # real interpreter, and on Windows 11 that opens a Windows Terminal window per job.
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                         creationflags=flags, close_fds=True)
    return p.pid


if __name__ == "__main__":
    sep = sys.argv.index("--")
    pid = launch(sys.argv[1], sys.argv[sep + 1:])
    print(pid)
