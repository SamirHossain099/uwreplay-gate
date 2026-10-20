"""Gate, freshness arm: do AoI scheduler rankings survive the error structure of replayed measured channels?

Published underwater AoI schedulers (Fang et al. TVT 2022; Chen et al. TGCN 2024; Cao et al. Frontiers 2025;
and the terrestrial results they build on, Kadota et al. 2018) evaluate on links whose packet losses are
independent across slots with a fixed success probability. Replayed sea-trial packets are not independent: a
measured channel fails in runs. This script (1) measures the run structure of the replayed packet outcomes
from results/psr_<channel>_r<rate>.json at every SNR where the success rate is between 0.05 and 0.95,
(2) fits a two-state Gilbert chain (P(fail | fail), P(fail | success)) with the same marginal success rate,
(3) simulates a single sink serving N sources, one transmission per slot, under round-robin, max-age-first
(MAF, the published optimal order for i.i.d. erasures with equal rates), and a burst-aware MAF that defers a
source whose last attempt failed, on (a) i.i.d. erasures at the measured success rate and (b) the fitted
Gilbert chain, and (4) replays the measured outcome sequences directly as a check. Writes results/gate_aoi.json.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
N_SOURCES = 4
HORIZON = 100_000
SEED = 3


def sequences(pj: dict, snr: str) -> list:
    """Time-ordered success sequences, one per (element, seed)."""
    tr = np.array(pj["measured"][snr]["trials"])
    seqs = []
    for el in np.unique(tr[:, 0]):
        for sd in np.unique(tr[:, 2]):
            m = (tr[:, 0] == el) & (tr[:, 2] == sd)
            sub = tr[m]
            sub = sub[np.argsort(sub[:, 1])]
            seqs.append(sub[:, 3].astype(int))
    return seqs


def burst_stats(seqs: list) -> dict:
    """Marginal failure rate, P(fail | previous fail), P(fail | previous success), mean failure run length."""
    ff = fs_ = nf = ns = 0
    fails = 0
    total = 0
    runs = []
    for s in seqs:
        total += s.size
        fails += int((s == 0).sum())
        for a, b in zip(s[:-1], s[1:]):
            if a == 0:
                nf += 1
                ff += int(b == 0)
            else:
                ns += 1
                fs_ += int(b == 0)
        run = 0
        for v in s:
            if v == 0:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
    pf = fails / total
    p_ff = ff / nf if nf else float("nan")
    p_sf = fs_ / ns if ns else float("nan")
    return dict(n_slots=int(total), p_fail=pf, p_fail_given_fail=p_ff, p_fail_given_success=p_sf,
                burst_ratio=(p_ff / pf if pf and not np.isnan(p_ff) else float("nan")),
                mean_fail_run=float(np.mean(runs)) if runs else 0.0, iid_mean_fail_run=(1 / (1 - pf) if pf < 1 else float("inf")))


def gilbert_sample(rng: np.random.Generator, n: int, p_ff: float, p_sf: float, start_fail: bool) -> np.ndarray:
    """Success (1) / failure (0) sequence from a two-state chain on the previous outcome."""
    out = np.empty(n, dtype=np.int8)
    fail = start_fail
    u = rng.random(n)
    for i in range(n):
        p = p_ff if fail else p_sf
        fail = u[i] < p
        out[i] = 0 if fail else 1
    return out


def simulate(outcomes: np.ndarray, policy: str) -> float:
    """outcomes: (N_SOURCES, T) channel outcome per source per slot (state evolves whether used or not).

    One source transmits per slot; a success resets its age to 1 at the next slot. Returns mean AoI over sources
    and slots after a burn-in of 1000 slots.
    """
    n, T = outcomes.shape
    age = np.ones(n, dtype=np.int64)
    last_fail = np.zeros(n, dtype=bool)
    total = 0
    rr = 0
    for t in range(T):
        if policy == "round_robin":
            i = rr % n
            rr += 1
        elif policy == "maf":
            i = int(np.argmax(age))
        elif policy == "maf_burst_aware":
            cand = np.where(~last_fail)[0]
            i = int(cand[np.argmax(age[cand])]) if cand.size else int(np.argmax(age))
        else:
            raise ValueError(policy)
        ok = outcomes[i, t] == 1
        last_fail[:] = False  # memory of one slot: only the source tried last slot carries a flag
        last_fail[i] = not ok
        age += 1
        if ok:
            age[i] = 1
        if t >= 1000:
            total += int(age.sum())
    return total / (n * (T - 1000))


def main() -> None:
    files = sorted(glob.glob(os.path.join(RES, "psr_*_r*.json")))
    results = dict(n_sources=N_SOURCES, horizon=HORIZON, cases=[])
    policies = ("round_robin", "maf", "maf_burst_aware")
    for fn in files:
        m = re.search(r"psr_(\w+)_r(\d+)\.json$", fn)
        if not m:
            continue
        name, rate = m.group(1), int(m.group(2))
        pj = json.load(open(fn, encoding="utf-8"))
        for snr in pj["snr_grid"]:
            psr = pj["measured"][str(snr)]["psr"]
            if not (0.05 <= psr <= 0.95):
                continue
            seqs = sequences(pj, str(snr))
            bs = burst_stats(seqs)
            rng = np.random.default_rng([SEED, sum(map(ord, name)), rate, int(10 * snr + 1000)])  # per-case seed
            case = dict(channel=name, chip_rate=rate, snr_db=snr, psr=psr, burst=bs, aoi={})
            if np.isnan(bs["p_fail_given_fail"]):
                results["cases"].append(case)
                continue
            # (a) i.i.d. erasures at the measured marginal
            iid = (rng.random((N_SOURCES, HORIZON)) < psr).astype(np.int8)
            # (b) Gilbert chain with the fitted transitions
            gil = np.stack([gilbert_sample(rng, HORIZON, bs["p_fail_given_fail"], bs["p_fail_given_success"], bool(rng.random() < bs["p_fail"]))
                            for _ in range(N_SOURCES)])
            # (c) measured sequences concatenated per source (independent noise seeds and elements joined end to end)
            cat = np.concatenate(seqs)
            reps = int(np.ceil(HORIZON / cat.size))
            meas = np.stack([np.tile(np.roll(cat, int(rng.integers(0, cat.size))), reps)[:HORIZON] for _ in range(N_SOURCES)])
            for arm, oc in (("iid", iid), ("gilbert", gil), ("measured_tiled", meas)):
                case["aoi"][arm] = {p: simulate(oc, p) for p in policies}
            for arm in case["aoi"]:
                a = case["aoi"][arm]
                a["maf_gain_over_rr_pct"] = 100 * (a["round_robin"] - a["maf"]) / a["round_robin"]
                a["burst_aware_gain_over_maf_pct"] = 100 * (a["maf"] - a["maf_burst_aware"]) / a["maf"]
            results["cases"].append(case)
            print(f"{name:6s} r{rate:4d} SNR {snr:+6.1f} PSR {psr:.2f} P(f|f)/P(f) {bs['burst_ratio']:.2f} "
                  f"run {bs['mean_fail_run']:.2f} vs iid {bs['iid_mean_fail_run']:.2f} | AoI iid RR/MAF/BA "
                  f"{case['aoi']['iid']['round_robin']:.2f}/{case['aoi']['iid']['maf']:.2f}/{case['aoi']['iid']['maf_burst_aware']:.2f} "
                  f"gilbert {case['aoi']['gilbert']['round_robin']:.2f}/{case['aoi']['gilbert']['maf']:.2f}/{case['aoi']['gilbert']['maf_burst_aware']:.2f} "
                  f"tiled {case['aoi']['measured_tiled']['round_robin']:.2f}/{case['aoi']['measured_tiled']['maf']:.2f}/{case['aoi']['measured_tiled']['maf_burst_aware']:.2f}")
    with open(os.path.join(RES, "gate_aoi.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("wrote results/gate_aoi.json with", len(results["cases"]), "cases")


if __name__ == "__main__":
    main()
