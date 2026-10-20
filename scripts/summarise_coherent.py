"""Coherent-modem ladder next to the FH-BFSK ladder: per site, the highest payload rate with PSR >= 0.9 at 30 dB,
the equaliser output-SNR ceiling, and the waterfall shift against AWGN. Reads results/coherent/psr_coh_*.json and
results/summary.json; writes results/coherent/summary_coherent.json and prints the table FINDINGS quotes."""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modem import Rate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
COH = os.path.join(RES, "coherent")
PSR_CEILING = 0.9


def snr_at_psr(grid, vals, level=0.5):
    g, v = np.array(grid, float), np.array(vals, float)
    for i in range(1, len(g)):
        if v[i - 1] < level <= v[i]:
            return float(g[i - 1] + (level - v[i - 1]) * (g[i] - g[i - 1]) / (v[i] - v[i - 1]))
    return float("nan") if v.max() < level else float(g[0])


def fh_ceilings_pooled() -> dict:
    """FH-BFSK packet success at 30 dB pooled over every long-replay file and element of a site (Black from the gate
    sweeps, which are its only ones), and the highest rate with pooled PSR >= PSR_CEILING."""
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    files = glob.glob(os.path.join(RES, "long", "psr_*_r*.json")) + glob.glob(os.path.join(RES, "psr_black_r*.json"))
    for fn in files:
        m = re.search(r"psr_(\w+?)(?:_(\w+?))?(?:_e\d+-\d+)?_r(\d+)\.json$", os.path.basename(fn))
        pj = json.load(open(fn, encoding="utf-8"))
        a = acc[m.group(1)][int(m.group(3))]
        a[0] += pj["measured"]["30.0"]["psr"] * pj["measured"]["30.0"]["n"]
        a[1] += pj["measured"]["30.0"]["n"]
    out = {}
    for site, rates in acc.items():
        psr = {r: v[0] / v[1] for r, v in rates.items()}
        n = {r: v[1] for r, v in rates.items()}
        ok = [r for r in rates if psr[r] >= PSR_CEILING]
        bps = {r: Rate(r).bits_per_s for r in rates}  # payload over frame duration, preamble included (C4)
        arq = max(rates, key=lambda r: bps[r] * psr[r])
        out[site] = dict(psr_30db={str(r): psr[r] for r in sorted(rates)}, n={str(r): n[r] for r in sorted(rates)},
                         bits_per_s={str(r): bps[r] for r in sorted(rates)},
                         ceiling_bps=(max(bps[r] for r in ok) if ok else 0.0),
                         arq_bps=bps[arq] * psr[arq], arq_rate=arq, arq_psr=psr[arq])
    return out


def main() -> None:
    rows = {}
    for fn in sorted(glob.glob(os.path.join(COH, "psr_coh_*.json"))):
        m = re.search(r"psr_coh_(\w+?)_(\w+?)(_e\d+-\d+)?_(\d+)_(bpsk|qpsk)\.json$", os.path.basename(fn))
        if not m or fn.endswith("_smoke.json"):
            continue
        pj = json.load(open(fn, encoding="utf-8"))
        grid = pj["snr_grid"]
        mv = [pj["measured"][str(s)]["psr"] for s in grid]
        av = [pj["awgn"][str(s)]["psr"] for s in grid]
        out30 = pj["measured"]["30.0"]["out_snr_db"]
        key = f"{m.group(1)}_{m.group(2)}{m.group(3) or ''}_{int(m.group(4))}_{m.group(5)}"
        rows[key] = dict(site=m.group(1), file=m.group(2), symbol_rate=int(m.group(4)), mod=m.group(5), bits_per_s=pj["scheme"]["bits_per_s"],
                         band_hz=pj["scheme"]["band_hz"], frame_s=pj["scheme"]["frame_s"], n_trials=pj["measured"]["30.0"]["n"],
                         fb_symbols=pj["fb_symbols"], fb_needed=pj["fb_needed"], fb_truncated=pj["fb_truncated"],
                         delay_window_ms=pj["delay_window_ms"], measured_psr_30db=mv[-1], awgn_psr_30db=av[-1],
                         out_snr_30db_mean=out30["mean"], out_snr_30db_median=out30["median"], out_snr_30db_p10=out30["p10"],
                         awgn_out_snr_30db=pj["awgn"]["30.0"]["out_snr_db"]["mean"],
                         measured_snr_at_psr50=snr_at_psr(grid, mv), awgn_snr_at_psr50=snr_at_psr(grid, av),
                         measured=dict(zip(map(str, grid), mv)), awgn=dict(zip(map(str, grid), av)),
                         out_snr_by_snr={str(s): pj["measured"][str(s)]["out_snr_db"]["mean"] for s in grid})
        rows[key]["shift_db"] = rows[key]["measured_snr_at_psr50"] - rows[key]["awgn_snr_at_psr50"]
    print(f"{'site':7s} {'file':10s} {'sym/s':>6s} {'mod':5s} {'bit/s':>6s} {'n':>4s} {'fb':>4s}{'':2s} {'meas@30':>8s} {'awgn@30':>8s} {'outSNR30':>9s} {'awgnOut':>8s} {'meas@.5':>8s} {'shift':>6s}")
    for k, r in sorted(rows.items(), key=lambda kv: (kv[1]["site"], kv[1]["symbol_rate"], kv[1]["mod"], kv[1]["file"])):
        flag = "*" if r["fb_truncated"] else " "
        print(f"{r['site']:7s} {r['file']:10s} {r['symbol_rate']:6d} {r['mod']:5s} {r['bits_per_s']:6.0f} {r['n_trials']:4d} {r['fb_symbols']:4d}{flag:2s} "
              f"{r['measured_psr_30db']:8.3f} {r['awgn_psr_30db']:8.3f} {r['out_snr_30db_mean']:9.1f} {r['awgn_out_snr_30db']:8.1f} "
              f"{r['measured_snr_at_psr50']:8.1f} {r['shift_db']:6.1f}")
    ladder = {}
    sites = sorted({r["site"] for r in rows.values()})
    fh_pooled = fh_ceilings_pooled()
    print("\n== Goodput ceilings (highest payload bit/s with measured PSR >= 0.9 at 30 dB) ==")
    for site in sites:
        # Coherent: pooled over every file of the site with trial weights, per scheme.
        per_scheme = defaultdict(lambda: [0.0, 0, None])
        for r in rows.values():
            if r["site"] == site:
                a = per_scheme[(r["symbol_rate"], r["mod"])]
                a[0] += r["measured_psr_30db"] * r["n_trials"]
                a[1] += r["n_trials"]
                a[2] = r["bits_per_s"]
        n_files_site = len({r["file"] for r in rows.values() if r["site"] == site})
        files_per_scheme = defaultdict(set)
        for r in rows.values():
            if r["site"] == site:
                files_per_scheme[(r["symbol_rate"], r["mod"])].add(r["file"])
        pooled = {k: dict(psr=v[0] / v[1], n=v[1], bits_per_s=v[2], n_files=len(files_per_scheme[k]),
                          all_files=(len(files_per_scheme[k]) == n_files_site)) for k, v in per_scheme.items()}
        # The ceiling is judged only on schemes evaluated on every file of the site (a single benign file set the
        # first sweep's Blue ceiling, CORRECTIONS C2). Schemes run on fewer files are listed but cannot set it.
        cands = [(k, v) for k, v in pooled.items() if v["psr"] >= PSR_CEILING and v["all_files"]]
        best = max(cands, key=lambda kv: kv[1]["bits_per_s"]) if cands else None
        partial = [(k, v) for k, v in pooled.items() if v["psr"] >= PSR_CEILING and not v["all_files"]]
        partial_best = max(partial, key=lambda kv: kv[1]["bits_per_s"]) if partial else None
        # ARQ goodput: rate times pooled success, best over the schemes evaluated on every file (what a link with
        # retransmissions delivers; the quantity the offloading economies consume).
        arq_c = [(k, v) for k, v in pooled.items() if v["all_files"]]
        arq_best = max(arq_c, key=lambda kv: kv[1]["bits_per_s"] * kv[1]["psr"]) if arq_c else None
        out_ceiling = max((r["out_snr_30db_mean"] for r in rows.values() if r["site"] == site), default=float("nan"))
        ladder[site] = dict(coherent_ceiling_bps=(best[1]["bits_per_s"] if best else 0.0),
                            coherent_best=(f"{best[0][0]} {best[0][1]}" if best else None),
                            coherent_pooled={f"{k[0]}_{k[1]}": v for k, v in pooled.items()},
                            n_files=n_files_site,
                            partial_file_best=(f"{partial_best[0][0]} {partial_best[0][1]} on {partial_best[1]['n_files']} of {n_files_site} files: "
                                               f"{partial_best[1]['bits_per_s']:.0f} bit/s, not used" if partial_best else None),
                            coherent_arq_bps=(arq_best[1]["bits_per_s"] * arq_best[1]["psr"] if arq_best else 0.0),
                            coherent_arq_scheme=(f"{arq_best[0][0]} {arq_best[0][1]}" if arq_best else None),
                            coherent_arq_psr=(arq_best[1]["psr"] if arq_best else None),
                            fh_ceiling_bps=fh_pooled.get(site, {}).get("ceiling_bps"), fh_arq_bps=fh_pooled.get(site, {}).get("arq_bps"),
                            fh_pooled=fh_pooled.get(site),
                            out_snr_ceiling_db=out_ceiling,
                            any_fb_truncated=any(r["fb_truncated"] for r in rows.values() if r["site"] == site))
        L = ladder[site]
        print(f"{site:7s} coherent ceiling {L['coherent_ceiling_bps']:5.0f} bit/s ({L['coherent_best']}, {L['n_files']} files), ARQ goodput {L['coherent_arq_bps']:5.0f} bit/s "
              f"({L['coherent_arq_scheme']} at PSR {L['coherent_arq_psr']:.3f}) | FH-BFSK ceiling {L['fh_ceiling_bps']}, ARQ {L['fh_arq_bps']:.0f} | out-SNR ceiling {out_ceiling:5.1f} dB  "
              f"fb truncated {L['any_fb_truncated']}" + (f"  [partial: {L['partial_file_best']}]" if L["partial_file_best"] else ""))
    with open(os.path.join(COH, "summary_coherent.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, ladder=ladder, psr_ceiling=PSR_CEILING), f, indent=1)


if __name__ == "__main__":
    main()
