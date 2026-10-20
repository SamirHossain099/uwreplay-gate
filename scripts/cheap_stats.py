"""PLAN item 5: can a node tell from a short probe which regime it is in?

For every (site, file, rate) long-replay result, three statistics are computed from only the first PROBE_S seconds
of the record at 30 dB nominal SNR, which is what a node could measure by sending frames for half a minute:
  (a) the probe packet-success rate;
  (b) the mean failure-run length (and the conditional-failure ratio) of the consecutive probe outcomes;
  (c) the variance, in dB, of the per-segment received power over the probe.
The decisions they must predict come from the full records:
  D1 (offloading): under the JSAC economy with the full-band reading, full offloading beats no offloading only
     while the link's packet-success rate stays above a break-even p*, found from results/gate_offloading.json
     (profit at PSR 1.0 and at Red's plateau, linear in PSR because ARQ goodput is); truth = full-record PSR >= p*.
  D2 (freshness): max-age-first beats round-robin on this file's link, truth = the chain fitted to the file's own
     full-record outcomes at 30 dB and simulated as in gate_aoi.py (HORIZON_FILE slots); defined only where the
     file's full-record PSR lies in 0.05 to 0.95, since a link that never fails has nothing to order.
Each statistic is turned into a rule "statistic >= threshold" (or <=) and judged leave-one-site-out: the threshold
is chosen on the other sites and applied to the held-out site, so no site is judged by a threshold fitted on it.
Writes results/cheap_stats.json; the table FINDINGS quotes is printed.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_aoi import N_SOURCES, burst_stats, gilbert_sample, simulate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PROBE_S = 30.0
SNR = "30.0"
HORIZON_FILE = 30_000


def file_aoi_truth(pj: dict, seed_key: tuple) -> dict | None:
    """Chain fit to the file's full-record outcomes at 30 dB and the MAF-versus-RR comparison, per gate_aoi.py."""
    psr = pj["measured"][SNR]["psr"]
    if not (0.05 <= psr <= 0.95):
        return None
    tr = np.array(pj["measured"][SNR]["trials"])
    seqs = []
    for el in np.unique(tr[:, 0]):
        for sd in np.unique(tr[:, 2]):
            sub = tr[(tr[:, 0] == el) & (tr[:, 2] == sd)]
            seqs.append(sub[np.argsort(sub[:, 1])][:, 3].astype(int))
    bs = burst_stats(seqs)
    if np.isnan(bs["p_fail_given_fail"]):
        return None
    rng = np.random.default_rng([7] + [sum(map(ord, str(k))) for k in seed_key])
    gil = np.stack([gilbert_sample(rng, HORIZON_FILE, bs["p_fail_given_fail"], bs["p_fail_given_success"], bool(rng.random() < bs["p_fail"]))
                    for _ in range(N_SOURCES)])
    rr, maf = simulate(gil, "round_robin"), simulate(gil, "maf")
    return dict(maf_beats_rr=bool(maf < rr), maf_gain_over_rr_pct=100 * (rr - maf) / rr, full_mean_fail_run=bs["mean_fail_run"],
                full_burst_ratio=bs["burst_ratio"], n_slots=bs["n_slots"])


def probe_stats(pj: dict) -> dict:
    """Statistics from the segments starting inside the first PROBE_S seconds of the record, element 0 (or the first
    element), both noise seeds; the AoI-relevant run statistics use each seed's consecutive outcomes."""
    fs_delay = float(pj["fs_delay"])
    starts = sorted(pj["segment_starts"])
    probe_starts = [s for s in starts if s / fs_delay < PROBE_S]
    tr = np.array(pj["measured"][SNR]["trials"])
    el0 = int(tr[:, 0].min())
    sel = tr[(tr[:, 0] == el0) & np.isin(tr[:, 1], probe_starts)]
    psr = float(sel[:, 3].mean())
    runs, ff, nf = [], 0, 0
    for sd in np.unique(sel[:, 2]):
        seq = sel[sel[:, 2] == sd]
        seq = seq[np.argsort(seq[:, 1])][:, 3].astype(int)
        run = 0
        for v in seq:
            if v == 0:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
        for a, b in zip(seq[:-1], seq[1:]):
            if a == 0:
                nf += 1
                ff += int(b == 0)
    p_fail = 1 - psr
    burst_ratio = (ff / nf) / p_fail if nf and p_fail > 0 else float("nan")
    pw = [10 * np.log10(pj["p_rx_segments"][f"{el0}:{s}"]) for s in probe_starts if f"{el0}:{s}" in pj["p_rx_segments"]]
    return dict(n_probe_segments=len(probe_starts), probe_s=max(probe_starts) / fs_delay + pj["slot_s"] if probe_starts else 0.0,
                psr=psr, mean_fail_run=(float(np.mean(runs)) if runs else 0.0), burst_ratio=burst_ratio,
                power_var_db=float(np.var(pw)) if len(pw) > 1 else 0.0, n_trials=int(sel.shape[0]))


def breakeven_psr() -> float:
    g = json.load(open(os.path.join(RES, "gate_offloading.json"), encoding="utf-8"))
    arms = g["arms"]["fullband"]
    p1, pr = arms["parametric"]["full"]["mean"], arms["red_psr"]["full"]["mean"]
    psr_red = g["plateau_psr_30db"]["red"]
    # profit(PSR) linear through (1, p1) and (psr_red, pr); zero crossing.
    slope = (p1 - pr) / (1 - psr_red)
    return float(1 - p1 / slope)


def loso(points: list, key: str, truth: str, direction: str) -> dict:
    """Leave-one-site-out accuracy of the rule stat >= t (direction 'ge') or stat <= t ('le')."""
    sites = sorted({p["site"] for p in points})
    correct, total, chosen = 0, 0, {}
    for held in sites:
        train = [p for p in points if p["site"] != held and not np.isnan(p[key])]
        test = [p for p in points if p["site"] == held and not np.isnan(p[key])]
        if not train or not test:
            continue
        vals = sorted({p[key] for p in train})
        cands = [vals[0] - 1e-9] + [(a + b) / 2 for a, b in zip(vals[:-1], vals[1:])] + [vals[-1] + 1e-9]
        best_t, best_acc = None, -1.0
        for t in cands:
            acc = np.mean([((p[key] >= t) if direction == "ge" else (p[key] <= t)) == p[truth] for p in train])
            if acc > best_acc:
                best_t, best_acc = t, acc
        chosen[held] = best_t
        for p in test:
            pred = (p[key] >= best_t) if direction == "ge" else (p[key] <= best_t)
            correct += int(pred == p[truth])
            total += 1
    return dict(accuracy=correct / total if total else float("nan"), n=total, thresholds=chosen)


def main() -> None:
    p_star = breakeven_psr()
    points = []
    for fn in sorted(glob.glob(os.path.join(RES, "long", "psr_*_r*.json"))):
        m = re.search(r"psr_(\w+?)_(\w+?)(?:_e(\d+)-\d+)?_r(\d+)\.json$", os.path.basename(fn))
        site, fkey, rate = m.group(1), m.group(2), int(m.group(4))
        pj = json.load(open(fn, encoding="utf-8"))
        st = probe_stats(pj)
        full_psr = pj["measured"][SNR]["psr"]
        pt = dict(site=site, file=fkey + (f"_e{m.group(3)}" if m.group(3) else ""), rate=rate, **st,
                  full_psr=full_psr, offload_pays_full=bool(full_psr >= p_star), offload_pays_probe=bool(st["psr"] >= p_star))
        truth = file_aoi_truth(pj, (site, fkey, rate))
        if truth is None:
            pt["maf_beats_rr"] = None  # no failures (or almost none) over the full record: nothing to order
        else:
            pt.update(dict(maf_beats_rr=truth["maf_beats_rr"], full_maf_gain_pct=truth["maf_gain_over_rr_pct"],
                           full_mean_fail_run=truth["full_mean_fail_run"], full_burst_ratio=truth["full_burst_ratio"]))
        pt["probe_minus_full_psr"] = st["psr"] - full_psr
        points.append(pt)
        print(f"  {site} {pt['file']} r{rate}: probe PSR {st['psr']:.3f} full {full_psr:.3f} run {st['mean_fail_run']:.2f}"
              + (f" | D2 truth MAF {'wins' if truth['maf_beats_rr'] else 'loses'} ({truth['maf_gain_over_rr_pct']:+.1f}%, full run {truth['full_mean_fail_run']:.2f})" if truth else " | D2 undefined"), flush=True)
    d1 = [p for p in points]
    d2 = [p for p in points if p["maf_beats_rr"] is not None]
    out = dict(probe_s=PROBE_S, breakeven_psr=p_star, n_points=len(points), n_points_aoi=len(d2), points=points, rules={})
    print(f"break-even PSR for full offloading (JSAC full-band reading): {p_star:.3f}; {len(points)} (site, file, rate) points, {len(d2)} with an AoI truth")
    # D1 by the probe PSR directly (same statistic, probe-length estimate), then LOSO for the three statistics.
    agree = np.mean([p["offload_pays_probe"] == p["offload_pays_full"] for p in d1])
    out["rules"]["D1_probe_psr_at_pstar"] = dict(accuracy=float(agree), n=len(d1))
    print(f"D1 offloading: probe PSR >= p* agrees with the full record in {100 * agree:.1f}% of {len(d1)} points")
    for key, direction in (("psr", "ge"), ("mean_fail_run", "le"), ("burst_ratio", "le"), ("power_var_db", "le")):
        r = loso(d1, key, "offload_pays_full", direction)
        out["rules"][f"D1_{key}"] = r
        print(f"  D1 LOSO rule on {key:14s}: accuracy {100 * r['accuracy']:5.1f}% (n={r['n']})")
    for key, direction in (("mean_fail_run", "le"), ("burst_ratio", "le"), ("psr", "ge"), ("power_var_db", "le")):
        r = loso(d2, key, "maf_beats_rr", direction)
        out["rules"][f"D2_{key}"] = r
        print(f"  D2 LOSO rule on {key:14s}: accuracy {100 * r['accuracy']:5.1f}% (n={r['n']}) thresholds {({k: round(v, 2) for k, v in r['thresholds'].items()})}")
    # Fixed, pre-stated rule from F10: mean failure run >= 2 slots means max-age-first loses.
    fixed = [((p["mean_fail_run"] < 2.0) == p["maf_beats_rr"]) for p in d2]
    out["rules"]["D2_fixed_run2"] = dict(accuracy=float(np.mean(fixed)), n=len(fixed))
    print(f"  D2 fixed rule (probe mean failure run < 2 slots => MAF wins): accuracy {100 * np.mean(fixed):.1f}% (n={len(fixed)})")
    base = max(np.mean([p["maf_beats_rr"] for p in d2]), 1 - np.mean([p["maf_beats_rr"] for p in d2]))
    out["rules"]["D2_majority_baseline"] = float(base)
    print(f"  D2 majority baseline: {100 * base:.1f}%")
    # Relative clustering: run length over its independent-error expectation (1 / PSR) and the conditional-failure
    # ratio, as pre-stated rules on the full record and on the probe where the probe saw at least three failures.
    def iid_run(psr):
        return 1 / psr if psr > 0 else float("inf")

    def acc(points, f):
        return float(np.mean([f(p) for p in points])) if points else float("nan")

    full_rules = {
        "full_ratio_ge_1p5_loses": lambda p: (p["full_burst_ratio"] < 1.5) == p["maf_beats_rr"],
        "full_ratio_ge_2_loses": lambda p: (p["full_burst_ratio"] < 2.0) == p["maf_beats_rr"],
        "full_run_ge_2_loses": lambda p: (p["full_mean_fail_run"] < 2.0) == p["maf_beats_rr"],
        "full_run_over_iid_ge_1p5_loses": lambda p: (p["full_mean_fail_run"] / iid_run(p["full_psr"]) < 1.5) == p["maf_beats_rr"],
        "full_run_over_iid_ge_2_loses": lambda p: (p["full_mean_fail_run"] / iid_run(p["full_psr"]) < 2.0) == p["maf_beats_rr"],
    }
    out["rules_full_record"] = {k: dict(accuracy=acc(d2, f), n=len(d2)) for k, f in full_rules.items()}
    sub = [p for p in d2 if round((1 - p["psr"]) * p["n_trials"]) >= 3 and not np.isnan(p["burst_ratio"])]
    probe_rules = {
        "probe_ratio_ge_1p5_loses": lambda p: (p["burst_ratio"] < 1.5) == p["maf_beats_rr"],
        "probe_ratio_ge_2_loses": lambda p: (p["burst_ratio"] < 2.0) == p["maf_beats_rr"],
        "probe_run_over_iid_ge_1p5_loses": lambda p: (p["mean_fail_run"] / iid_run(p["psr"]) < 1.5) == p["maf_beats_rr"],
        "probe_run_over_iid_ge_2_loses": lambda p: (p["mean_fail_run"] / iid_run(p["psr"]) < 2.0) == p["maf_beats_rr"],
    }
    maj = max(np.mean([p["maf_beats_rr"] for p in sub]), 1 - np.mean([p["maf_beats_rr"] for p in sub])) if sub else float("nan")
    out["rules_probe_with_3_failures"] = dict(n=len(sub), n_total=len(d2), majority_baseline=float(maj),
                                             **{k: dict(accuracy=acc(sub, f)) for k, f in probe_rules.items()})
    out["d2_by_site"] = {}
    for s in sorted({p["site"] for p in d2}):
        ps = [p for p in d2 if p["site"] == s]
        out["d2_by_site"][s] = dict(n=len(ps), maf_wins=int(sum(p["maf_beats_rr"] for p in ps)),
                                   full_ratio_mean=float(np.mean([p["full_burst_ratio"] for p in ps])),
                                   full_run_mean=float(np.mean([p["full_mean_fail_run"] for p in ps])),
                                   full_run_over_iid_mean=float(np.mean([p["full_mean_fail_run"] / iid_run(p["full_psr"]) for p in ps])),
                                   probe_saw_no_failure=int(sum(p["psr"] >= 0.999 for p in ps)))
    out["d1_confusion"] = dict(tp=int(sum(p["offload_pays_probe"] and p["offload_pays_full"] for p in points)),
                               fp=int(sum(p["offload_pays_probe"] and not p["offload_pays_full"] for p in points)),
                               fn=int(sum((not p["offload_pays_probe"]) and p["offload_pays_full"] for p in points)),
                               fp_sites=sorted({p["site"] for p in points if p["offload_pays_probe"] and not p["offload_pays_full"]}))
    print("  full-record rules:", {k: round(100 * v["accuracy"], 1) for k, v in out["rules_full_record"].items()})
    print(f"  probe rules where >= 3 failures seen (n={len(sub)}, majority {100 * maj:.1f}%):",
          {k: round(100 * v["accuracy"], 1) for k, v in out["rules_probe_with_3_failures"].items() if isinstance(v, dict)})
    print("  D2 by site:", {s: f"{v['maf_wins']}/{v['n']}" for s, v in out["d2_by_site"].items()}, " D1 confusion:", out["d1_confusion"])
    # Majority baseline for D1 and the counts of distinct records and sites (the points are (site, record, rate) triples).
    t1 = [p["offload_pays_full"] for p in points]
    out["d1_majority_baseline"] = float(max(np.mean(t1), 1 - np.mean(t1)))
    out["n_distinct_records"] = len({(p["site"], p["file"].split("_e")[0]) for p in points})
    out["n_sites_d1"] = len({p["site"] for p in points})
    out["n_sites_d2"] = len({p["site"] for p in d2})
    out["n_distinct_records_d2"] = len({(p["site"], p["file"].split("_e")[0]) for p in d2})
    # Leave-one-site-out over the whole rule family (statistic and threshold both chosen on the other sites), so the
    # fixed rules above are not the only numbers: this is the honest estimate of what a node that had not seen these
    # sites would get. The family is the four relative and absolute statistics at a grid of thresholds, predicting
    # "max-age-first loses" when the statistic is at or above the threshold.
    family = [("burst_ratio", (1.25, 1.5, 1.75, 2.0, 2.5, 3.0)), ("run_over_iid", (1.25, 1.5, 1.75, 2.0, 2.5)),
              ("mean_fail_run", (1.5, 2.0, 3.0, 4.0)), ("psr_loss", (0.05, 0.1, 0.2, 0.3))]

    def stat(p, name, full):
        if name == "burst_ratio":
            return p["full_burst_ratio"] if full else p["burst_ratio"]
        if name == "run_over_iid":
            return (p["full_mean_fail_run"] * p["full_psr"]) if full else (p["mean_fail_run"] * p["psr"])
        if name == "mean_fail_run":
            return p["full_mean_fail_run"] if full else p["mean_fail_run"]
        return (1 - p["full_psr"]) if full else (1 - p["psr"])

    def loso_family(pts, full):
        sites = sorted({p["site"] for p in pts})
        correct = total = 0
        chosen = {}
        for held in sites:
            train = [p for p in pts if p["site"] != held]
            test = [p for p in pts if p["site"] == held]
            best = None
            for name, grid in family:
                for t in grid:
                    vals = [(stat(p, name, full), p["maf_beats_rr"]) for p in train]
                    vals = [(v, y) for v, y in vals if v == v]  # drop nan
                    if not vals:
                        continue
                    acc_tr = np.mean([((v >= t) == (not y)) for v, y in vals])
                    if best is None or acc_tr > best[0]:
                        best = (acc_tr, name, t)
            chosen[held] = dict(rule=best[1], threshold=best[2], train_accuracy=float(best[0]))
            for p in test:
                v = stat(p, best[1], full)
                if v != v:
                    continue
                correct += int((v >= best[2]) == (not p["maf_beats_rr"]))
                total += 1
        return dict(accuracy=(correct / total if total else float("nan")), n=total, chosen=chosen)

    out["d2_loso_family_full_record"] = loso_family(d2, full=True)
    out["d2_loso_family_probe"] = loso_family(sub, full=False)
    print(f"  D1 majority baseline {100 * out['d1_majority_baseline']:.1f}%; distinct records {out['n_distinct_records']} (D2 {out['n_distinct_records_d2']}), sites D1 {out['n_sites_d1']} D2 {out['n_sites_d2']}")
    print(f"  D2 LOSO over the rule family: full record {100 * out['d2_loso_family_full_record']['accuracy']:.1f}% (n={out['d2_loso_family_full_record']['n']}), "
          f"probe with >=3 failures {100 * out['d2_loso_family_probe']['accuracy']:.1f}% (n={out['d2_loso_family_probe']['n']}); chosen per held-out site:",
          {k: (v['rule'], v['threshold']) for k, v in out['d2_loso_family_full_record']['chosen'].items()})
    # Stationarity: how far the probe's PSR sits from the full record's, per site.
    drift = defaultdict(list)
    for p in points:
        drift[p["site"]].append(abs(p["probe_minus_full_psr"]))
    out["probe_vs_full_abs_psr_gap"] = {s: dict(mean=float(np.mean(v)), max=float(np.max(v)), frac_over_0p1=float(np.mean(np.array(v) > 0.1)), n=len(v))
                                       for s, v in drift.items()}
    print("  |probe PSR - full PSR| per site:", {s: (round(v["mean"], 3), round(v["frac_over_0p1"], 2)) for s, v in out["probe_vs_full_abs_psr_gap"].items()})
    print(f"\n{'site':7s} {'file':12s} {'rate':>5s} {'segs':>4s} {'probePSR':>8s} {'fullPSR':>7s} {'run':>5s} {'burst':>5s} {'pvar':>5s} {'D1':>4s} {'D2truth':>7s}")
    for p in points:
        print(f"{p['site']:7s} {p['file']:12s} {p['rate']:5d} {p['n_probe_segments']:4d} {p['psr']:8.3f} {p['full_psr']:7.3f} {p['mean_fail_run']:5.2f} "
              f"{p['burst_ratio']:5.2f} {p['power_var_db']:5.2f} {str(p['offload_pays_full'])[0]:>4s} {str(p['maf_beats_rr'])[0] if p['maf_beats_rr'] is not None else '-':>7s}")
    with open(os.path.join(RES, "cheap_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/cheap_stats.json")


if __name__ == "__main__":
    main()
