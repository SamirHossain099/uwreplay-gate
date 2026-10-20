"""Print every number FINDINGS.md quotes, from results/*.json. Also writes results/summary.json."""
from __future__ import annotations

import glob
import json
import os
import re

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modem import Rate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def snr_at_psr(grid, vals, level=0.5):
    """Linear interpolation of the SNR at which PSR first reaches `level` (nan if never)."""
    g, v = np.array(grid, float), np.array(vals, float)
    for i in range(1, len(g)):
        if v[i - 1] < level <= v[i]:
            return float(g[i - 1] + (level - v[i - 1]) * (g[i] - g[i - 1]) / (v[i] - v[i - 1]))
    return float("nan") if v.max() < level else float(g[0])


def main() -> None:
    summary = dict(psr={}, linkbudget=None, offloading=None, aoi=None)
    print("== PSR sweeps: SNR (dB, nominal in-band) at PSR = 0.5, and PSR at 30 dB ==")
    print(f"{'channel':8s} {'rate':>5s} {'bit/s':>6s} {'n':>5s} {'awgn@0.5':>9s} {'meas@0.5':>9s} {'shift':>6s} {'meas@30':>8s} {'awgn@30':>8s} {'meas@0':>7s}")
    for fn in sorted(glob.glob(os.path.join(RES, "psr_*_r*.json"))):
        m = re.search(r"psr_(\w+)_r(\d+)\.json$", fn)
        if not m:
            continue
        pj = json.load(open(fn, encoding="utf-8"))
        grid = pj["snr_grid"]
        mv = [pj["measured"][str(s)]["psr"] for s in grid]
        av = [pj["awgn"][str(s)]["psr"] for s in grid]
        a50, m50 = snr_at_psr(grid, av), snr_at_psr(grid, mv)
        # Payload rate from the chip rate (preamble included, C4), not the value the sweep stored before the definition changed.
        row = dict(channel=m.group(1), chip_rate=int(m.group(2)), bits_per_s=Rate(int(m.group(2))).bits_per_s, n_trials=pj["measured"]["30.0"]["n"],
                   awgn_snr_at_psr50=a50, measured_snr_at_psr50=m50, shift_db=(m50 - a50), measured_psr_30db=mv[-1],
                   awgn_psr_30db=av[-1], measured_psr_0db=pj["measured"]["0.0"]["psr"], measured=dict(zip(map(str, grid), mv)),
                   awgn=dict(zip(map(str, grid), av)), p_rx_mean=pj["p_rx_mean"], n_segments=pj["n_segments"], elements=pj["elements"])
        summary["psr"][f"{m.group(1)}_r{m.group(2)}"] = row
        print(f"{row['channel']:8s} {row['chip_rate']:5d} {row['bits_per_s']:6.0f} {row['n_trials']:5d} {a50:9.1f} {m50:9.1f} {row['shift_db']:6.1f} "
              f"{row['measured_psr_30db']:8.3f} {row['awgn_psr_30db']:8.3f} {row['measured_psr_0db']:7.3f}")
    lb = json.load(open(os.path.join(RES, "linkbudget.json"), encoding="utf-8"))
    summary["linkbudget"] = lb
    print("\n== Link budgets (published models) ==")
    for d in lb["jsac"]["device_to_auv"][:6]:
        print(f"JSAC      d={d['d_m']:5d} m SNR {d['snr_db']:5.1f} dB Shannon {d['rate_bps']/1e3:6.1f} kbit/s")
    for d in lb["frontiers"]["sensor_to_auv"]:
        if d["d_m"] in (100, 400, 1000):
            print(f"Frontiers d={d['d_m']:5d} m SNR {d['snr_db']:5.1f} dB Shannon {d['rate_bps']/1e3:6.1f} kbit/s")
    fo = os.path.join(RES, "gate_offloading.json")
    if os.path.exists(fo):
        go = json.load(open(fo, encoding="utf-8"))
        summary["offloading"] = go
        print("\n== Offloading gate (JSAC economy) ==")
        print("break-even per-device rate R*: median", round(go["breakeven_rate_bps"]["median"]), "min", round(go["breakeven_rate_bps"]["min"]),
              "max", round(go["breakeven_rate_bps"]["max"]), "bit/s")
        print("paper's per-device Shannon rate at 10-100 m: full band", round(go["parametric_rate_bps_example"]["fullband"]),
              "shared by 12:", round(go["parametric_rate_bps_example"]["shared_12"]), "bit/s")
        print("paper SNR (full band) at 10/30/100 m:", {k: round(v, 1) for k, v in go["paper_snr_db_full_band"].items()})
        print("goodput ceilings (bit/s, highest ladder rate with PSR >= 0.9 at 30 dB):", go["goodput_ceiling_bps"])
        print("plateau PSR at 30 dB, 160 chip/s:", {k: round(v, 3) for k, v in go["plateau_psr_30db"].items()})
        for bw in go["arms"]:
            for arm, s in go["arms"][bw].items():
                print(f"  {bw:8s} {arm:12s} full {s['full']['mean']:9.1f} random {s['random']['mean']:9.1f} partial {s['partial']['mean']:9.1f} "
                      f"optimum {s['optimum']['mean']:9.1f} (sd {s['optimum']['sd']:.1f}) n_off {s['optimum_n_off']['mean']:4.1f} ranking {s['ranking'][:3]} pays {s['offloading_pays']}")
    fa = os.path.join(RES, "gate_aoi.json")
    if os.path.exists(fa):
        ga = json.load(open(fa, encoding="utf-8"))
        summary["aoi"] = ga
        print("\n== AoI gate ==")
        for c in ga["cases"]:
            if "iid" not in c["aoi"]:
                continue
            b = c["burst"]
            print(f"{c['channel']:6s} r{c['chip_rate']:4d} SNR {c['snr_db']:+6.1f} PSR {c['psr']:.2f} n={b['n_slots']} P(f|f)/P(f) {b['burst_ratio']:.2f} "
                  f"run {b['mean_fail_run']:.2f} (iid {b['iid_mean_fail_run']:.2f}) | MAF gain over RR: iid {c['aoi']['iid']['maf_gain_over_rr_pct']:5.1f}% "
                  f"gilbert {c['aoi']['gilbert']['maf_gain_over_rr_pct']:5.1f}% tiled {c['aoi']['measured_tiled']['maf_gain_over_rr_pct']:5.1f}% | "
                  f"burst-aware over MAF: iid {c['aoi']['iid']['burst_aware_gain_over_maf_pct']:5.1f}% gilbert {c['aoi']['gilbert']['burst_aware_gain_over_maf_pct']:5.1f}% "
                  f"tiled {c['aoi']['measured_tiled']['burst_aware_gain_over_maf_pct']:5.1f}%")
    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
