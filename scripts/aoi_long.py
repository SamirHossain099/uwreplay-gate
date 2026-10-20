"""Freshness arm on the long records (PLAN item 2, analysis half): the AoI comparison of gate_aoi.py rerun with the
replayed packet outcomes from results/long/, pooled per (site, chip rate, SNR) across every file and element of the
site, so the burst statistics and the directly replayed sequences rest on hundreds to thousands of consecutive
slots rather than the gate's 64 to 360.

Per file and element the outcomes are consecutive segments of one record (and for Green consecutive ten-minute
slots); files of one site are concatenated in file order. The three arms are those of the gate: i.i.d. erasures at
the measured success rate, the fitted two-state chain, and the measured sequences replayed directly (each source
starts at its own offset of the concatenation). Writes results/aoi_long.json.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from gate_aoi import HORIZON, N_SOURCES, SEED, burst_stats, gilbert_sample, simulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG = os.path.join(ROOT, "results", "long")
POLICIES = ("round_robin", "maf", "maf_burst_aware")


def file_sort_key(fn: str):
    m = re.search(r"psr_(\w+?)_(\w+?)(?:_e(\d+)-\d+)?_r(\d+)\.json$", os.path.basename(fn))
    site, fkey, el, rate = m.group(1), m.group(2), m.group(3), int(m.group(4))
    num = int(fkey.split("_")[1]) if "_" in fkey else 0
    return site, rate, num, int(el or 0)


def sequences_long(pj: dict, snr: str) -> dict:
    """{seed: [sequence per element in element order]} with segments in time order."""
    tr = np.array(pj["measured"][snr]["trials"])
    out = defaultdict(list)
    for sd in sorted(np.unique(tr[:, 2])):
        for el in sorted(np.unique(tr[:, 0])):
            sub = tr[(tr[:, 0] == el) & (tr[:, 2] == sd)]
            out[int(sd)].append(sub[np.argsort(sub[:, 1])][:, 3].astype(int))
    return out


def main() -> None:
    files = sorted(glob.glob(os.path.join(LONG, "psr_*_r*.json")), key=file_sort_key)
    groups = defaultdict(list)
    for fn in files:
        site, rate, _, _ = file_sort_key(fn)
        groups[(site, rate)].append(fn)
    results = dict(n_sources=N_SOURCES, horizon=HORIZON, cases=[])
    for (site, rate), fns in sorted(groups.items()):
        loaded = [json.load(open(fn, encoding="utf-8")) for fn in fns]
        grid = loaded[0]["snr_grid"]
        for snr in grid:
            key = str(snr)
            n_tot = sum(pj["measured"][key]["n"] for pj in loaded)
            psr = sum(pj["measured"][key]["psr"] * pj["measured"][key]["n"] for pj in loaded) / n_tot
            if not (0.05 <= psr <= 0.95):
                continue
            per_seed = defaultdict(list)
            for pj in loaded:
                for sd, seqs in sequences_long(pj, key).items():
                    per_seed[sd].extend(seqs)
            seqs = [np.concatenate(v) for v in per_seed.values()]  # one long sequence per noise seed
            bs = burst_stats(seqs)
            rng = np.random.default_rng([SEED, sum(map(ord, site)), rate, int(10 * snr + 1000), 7])
            case = dict(site=site, chip_rate=rate, snr_db=snr, psr=psr, n_files=len(fns), burst=bs, aoi={})
            if np.isnan(bs["p_fail_given_fail"]):
                results["cases"].append(case)
                continue
            iid = (rng.random((N_SOURCES, HORIZON)) < psr).astype(np.int8)
            gil = np.stack([gilbert_sample(rng, HORIZON, bs["p_fail_given_fail"], bs["p_fail_given_success"], bool(rng.random() < bs["p_fail"]))
                            for _ in range(N_SOURCES)])
            cat = np.concatenate(seqs)
            reps = int(np.ceil(HORIZON / cat.size))
            meas = np.stack([np.tile(np.roll(cat, int(rng.integers(0, cat.size))), reps)[:HORIZON] for _ in range(N_SOURCES)])
            for arm, oc in (("iid", iid), ("gilbert", gil), ("measured_tiled", meas)):
                case["aoi"][arm] = {p: simulate(oc, p) for p in POLICIES}
            for arm in case["aoi"]:
                a = case["aoi"][arm]
                a["maf_gain_over_rr_pct"] = 100 * (a["round_robin"] - a["maf"]) / a["round_robin"]
                a["burst_aware_gain_over_maf_pct"] = 100 * (a["maf"] - a["maf_burst_aware"]) / a["maf"]
            case["measured_slots_per_source_copy"] = int(cat.size)
            results["cases"].append(case)
            print(f"{site:7s} r{rate:4d} SNR {snr:+6.1f} PSR {psr:.3f} files {len(fns):2d} slots {bs['n_slots']:5d} "
                  f"P(f|f)/P(f) {bs['burst_ratio']:.2f} run {bs['mean_fail_run']:.2f} vs iid {bs['iid_mean_fail_run']:.2f} | "
                  f"MAF over RR: iid {case['aoi']['iid']['maf_gain_over_rr_pct']:+6.1f}% chain {case['aoi']['gilbert']['maf_gain_over_rr_pct']:+6.1f}% "
                  f"direct {case['aoi']['measured_tiled']['maf_gain_over_rr_pct']:+6.1f}% | burst-aware over MAF direct "
                  f"{case['aoi']['measured_tiled']['burst_aware_gain_over_maf_pct']:+6.1f}%", flush=True)
    with open(os.path.join(ROOT, "results", "aoi_long.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("wrote results/aoi_long.json with", len(results["cases"]), "cases")


if __name__ == "__main__":
    main()
