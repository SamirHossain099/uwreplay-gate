"""Gate, offloading arm: Wang et al. (IEEE JSAC 2023, arXiv 2210.14619) multi-tier underwater computing.

Reproduces the paper's single-hop offloading economy for one device group served by one hovering AUV with
the paper's own channel (eqs. 1-5, 11-15), computing (24-31), energy (51-54) and profit (61-68) models and
Table II values, then re-runs the same decisions with the device-to-AUV link replaced by what the replayed
measured channels deliver (results/psr_<channel>_r<rate>.json).

Two readings of the paper's bandwidth allocation are evaluated, because the paper's figures cannot be
reproduced to the digit (its eq. 15 mixes units and the station computing resource F is not in Table II):
  shared   : r_ki = 1/n_off for every offloader, as constraint (16) requires.
  fullband : r_ki = 1 for every offloader, which is the reading under which offloading is profitable in the
             paper's Figs. 7-9. The paper's headline (offloading beats non-offloading, the optimised strategy
             beats partial and random) holds only under this reading.
For each layout the break-even per-device rate R* at which full offloading stops paying is also computed, so a
reader can place any link model on one axis.

Link arms:
  parametric      : Shannon rate at the paper's SNR, no packet loss (the paper).
  awgn            : rate times the AWGN packet success rate of the 160 chip/s frame at that SNR (1.0 here).
  <channel>_psr   : rate times the measured 160 chip/s PSR at the same nominal SNR (ARQ goodput).
  <channel>_ceil  : the link at the goodput ceiling of the rate ladder on that channel (highest rate with
                    measured PSR >= 0.9 at 30 dB nominal SNR), shared among offloaders.
Writes results/gate_offloading.json.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linkbudget import JSAC, shannon_bps, snr_db_jsac  # noqa: E402
from modem import Rate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

MU, SIGMA = 1.25e-26, 3  # Table II
F_AUV_TOTAL = 10e9  # station computing resource F per AUV (not in Table II; 10 GHz assumed, cost term is small)
OMEGA = (10, 20)
LAMBDA = (1, 2)
VARRHO, CHI = 1, 2
N_DEV = 12
N_LAYOUTS = 40
D_RANGE_M = (10.0, 100.0)  # AUV hovers 10 m above the seabed devices (h0 = 10, d0 = 20); group spread 100 m
D_AUV_STATION_M = 500.0
SCHEMES = ("non_offloading", "full", "random", "partial", "greedy", "optimum")


def psr_lookup(psr_json: dict, arm: str):
    grid = np.array(psr_json["snr_grid"], dtype=float)
    vals = np.array([psr_json[arm][str(s)]["psr"] for s in psr_json["snr_grid"]])

    def f(snr_db: float) -> float:
        return float(np.interp(snr_db, grid, vals, left=vals[0], right=vals[-1]))
    return f


def layout(rng: np.random.Generator) -> dict:
    return dict(d=rng.uniform(*D_RANGE_M, N_DEV), z=rng.uniform(*JSAC["z_bits"], N_DEV),
                fcpu=rng.uniform(*JSAC["f_cpu_hz"], N_DEV), alpha=rng.uniform(*JSAC["alpha_cyc_per_bit"], N_DEV),
                omega=rng.uniform(*OMEGA, N_DEV), lam=rng.uniform(*LAMBDA, N_DEV))


def device_rates(L: dict, n_off: int, bw_mode: str, link: dict) -> np.ndarray:
    """Per-device device-to-AUV goodput in bit/s for a given number of offloaders."""
    r = 1.0 if bw_mode == "fullband" else 1.0 / n_off
    if link["kind"] == "fixed":
        return np.full(N_DEV, link["bps"] * r)
    snr = np.array([snr_db_jsac(di, JSAC["p_dev_w"], JSAC["f_khz"], r * JSAC["bw_l_hz"], JSAC["k"], JSAC["s"], JSAC["w"], JSAC["eta"]) for di in L["d"]])
    rate = r * np.array([shannon_bps(s, JSAC["bw_l_hz"]) for s in snr])
    psr = np.array([link["psr"](s) for s in snr]) if link["kind"] == "psr" else 1.0
    return rate * psr


def profit(o: np.ndarray, L: dict, bw_mode: str, link: dict) -> float:
    """Eq. (68) without the AUV cruising cost, which is identical across offloading schemes on a fixed path."""
    n_off = int(o.sum())
    if n_off == 0:
        return 0.0
    fm = 1.0 / n_off
    rate_da = device_rates(L, n_off, bw_mode, link)
    snr_as = snr_db_jsac(D_AUV_STATION_M, JSAC["p_auv_w"], JSAC["f_khz"], JSAC["bw_h_hz"], JSAC["k"], JSAC["s"], JSAC["w"], JSAC["eta"])
    rate_as = shannon_bps(snr_as, JSAC["bw_h_hz"])  # eq. 21, AUV to station, not replayed
    t_local = L["alpha"] * L["z"] / L["fcpu"]  # eq. 24
    e_local = MU * L["fcpu"] ** SIGMA * t_local  # eq. 51
    t_da = L["z"] / np.maximum(rate_da, 1e-9)  # eq. 26
    t_as = L["z"] / rate_as  # eq. 27
    t_m = L["alpha"] * L["z"] / (fm * F_AUV_TOTAL)  # eq. 28
    e_da = JSAC["p_dev_w"] * t_da  # transmit energy for the air time
    e_as = JSAC["p_auv_w"] * t_as
    e_m = MU * (fm * F_AUV_TOTAL) ** SIGMA * t_m  # eq. 53
    t_s = t_local - (t_m + t_da + t_as)  # eq. 61
    e_s = e_local - e_da  # eq. 62
    rev = o * (L["omega"] * t_s + L["lam"] * e_s)  # eq. 63
    cost = o * (VARRHO * e_m + CHI * e_as)  # eqs. 64-66
    return float(np.sum(rev - cost))


def breakeven_rate_bps(L: dict) -> float:
    """Per-device rate at which full offloading of the whole group has zero profit (bisection on a fixed link)."""
    o = np.ones(N_DEV, dtype=int)
    lo, hi = 1.0, 1e8
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if profit(o, L, "fullband", dict(kind="fixed", bps=mid)) > 0:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def run_arm(L_list: list, bw_mode: str, link: dict, rng: np.random.Generator) -> dict:
    out = {k: [] for k in SCHEMES}
    out["greedy_n_off"], out["optimum_n_off"] = [], []
    for L in L_list:
        pf = lambda o: profit(o, L, bw_mode, link)  # noqa: E731
        o_rand = (rng.random(N_DEV) < 0.5).astype(int)
        t_local = L["alpha"] * L["z"] / L["fcpu"]
        o_part = (t_local >= np.median(t_local)).astype(int)
        o = np.zeros(N_DEV, dtype=int)
        best = 0.0
        while True:
            gains = [(pf(np.where(np.arange(N_DEV) == i, 1, o)) - best, i) for i in np.where(o == 0)[0]]
            if not gains:
                break
            g, i = max(gains)
            if g <= 0:
                break
            o[i] = 1
            best += g
        bb, o_b = -np.inf, None
        for bits in itertools.product((0, 1), repeat=N_DEV):
            ob = np.array(bits)
            p = pf(ob)
            if p > bb:
                bb, o_b = p, ob
        out["non_offloading"].append(0.0)
        out["full"].append(pf(np.ones(N_DEV, dtype=int)))
        out["random"].append(pf(o_rand))
        out["partial"].append(pf(o_part))
        out["greedy"].append(best)
        out["optimum"].append(bb)
        out["greedy_n_off"].append(int(o.sum()))
        out["optimum_n_off"].append(int(o_b.sum()))
    summ = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for k, v in out.items()}
    summ["ranking"] = sorted(SCHEMES, key=lambda k: -summ[k]["mean"])
    summ["offloading_pays"] = bool(summ["optimum"]["mean"] > 0)
    return summ


def main() -> None:
    rng = np.random.default_rng(7)
    layouts = [layout(rng) for _ in range(N_LAYOUTS)]
    links = {"parametric": dict(kind="shannon")}
    ladder, channels = {}, []
    for name in ("blue", "red", "black", "green"):
        fn = os.path.join(RES, f"psr_{name}_r160.json")
        if os.path.exists(fn):
            pj = json.load(open(fn, encoding="utf-8"))
            if "awgn" not in links:
                links["awgn"] = dict(kind="psr", psr=psr_lookup(pj, "awgn"))
            links[f"{name}_psr"] = dict(kind="psr", psr=psr_lookup(pj, "measured"))
            channels.append(name)
        ladder[name] = {}
        for rate in (160, 320, 640, 1920):
            fr = os.path.join(RES, f"psr_{name}_r{rate}.json")
            if os.path.exists(fr):
                pr = json.load(open(fr, encoding="utf-8"))
                ladder[name][str(rate)] = dict(psr_30db=pr["measured"]["30.0"]["psr"], awgn_psr_30db=pr["awgn"]["30.0"]["psr"],
                                               bits_per_s=Rate(rate).bits_per_s)  # payload over frame duration (C4)
    ceilings = {}
    for name, lad in ladder.items():
        ok = [v["bits_per_s"] for v in lad.values() if v["psr_30db"] >= 0.9]
        if lad:
            ceilings[name] = max(ok, default=0.0)
            if ceilings[name] > 0:
                links[f"{name}_ceil"] = dict(kind="fixed", bps=ceilings[name])
    # Pooled arms (2026-10-20, after review): the gate arms above rest on record 1 of each site. Where the long replays
    # and the coherent sweep have been summarised, add arms at each site's pooled goodput with retransmissions for both
    # frame families, and plateau-PSR arms at the pooled 30 dB success of the FH frame where it is below 0.95.
    pooled = {}
    coh_path = os.path.join(RES, "coherent", "summary_coherent.json")
    if os.path.exists(coh_path):
        lad_c = json.load(open(coh_path, encoding="utf-8"))["ladder"]
        for site, L in lad_c.items():
            if L.get("fh_arq_bps"):
                links[f"{site}_fhpool"] = dict(kind="fixed", bps=L["fh_arq_bps"])
            if L.get("coherent_arq_bps"):
                links[f"{site}_coh"] = dict(kind="fixed", bps=L["coherent_arq_bps"])
            fp = L.get("fh_pooled") or {}
            for rate, p in (fp.get("psr_30db") or {}).items():
                if p < 0.95:
                    links[f"{site}_pool{rate}_psr"] = dict(kind="psr", psr=(lambda s, p=p: p))
            pooled[site] = dict(fh_arq_bps=L.get("fh_arq_bps"), coherent_arq_bps=L.get("coherent_arq_bps"), fh_psr_30db=fp.get("psr_30db"))
    snr_at = {str(d): float(snr_db_jsac(d, JSAC["p_dev_w"], JSAC["f_khz"], JSAC["bw_l_hz"], JSAC["k"], JSAC["s"], JSAC["w"], JSAC["eta"])) for d in (10, 30, 100)}
    results = dict(n_dev=N_DEV, n_layouts=N_LAYOUTS, d_range_m=D_RANGE_M, f_auv_total_hz=F_AUV_TOTAL, channels=channels,
                   paper_snr_db_full_band=snr_at, rate_ladder=ladder, goodput_ceiling_bps=ceilings,
                   plateau_psr_30db={n: links[f"{n}_psr"]["psr"](30.0) for n in channels}, pooled=pooled, arms={})
    be = [breakeven_rate_bps(L) for L in layouts]
    results["breakeven_rate_bps"] = dict(median=float(np.median(be)), min=float(np.min(be)), max=float(np.max(be)))
    # parametric per-device rates the paper implies, for the record
    L0 = layouts[0]
    results["parametric_rate_bps_example"] = dict(fullband=float(np.median(device_rates(L0, N_DEV, "fullband", links["parametric"]))),
                                                  shared_12=float(np.median(device_rates(L0, N_DEV, "shared", links["parametric"]))))
    print("break-even per-device rate R* (median over layouts):", round(results["breakeven_rate_bps"]["median"]), "bit/s")
    print("paper's per-device Shannon rate, full band / shared by 12:", results["parametric_rate_bps_example"])
    for bw_mode in ("fullband", "shared"):
        results["arms"][bw_mode] = {}
        for arm, link in links.items():
            rng_arm = np.random.default_rng(11)
            s = run_arm(layouts, bw_mode, link, rng_arm)
            results["arms"][bw_mode][arm] = s
            print(f"{bw_mode:8s} {arm:12s} full {s['full']['mean']:10.1f} random {s['random']['mean']:10.1f} partial {s['partial']['mean']:10.1f} "
                  f"greedy {s['greedy']['mean']:10.1f} optimum {s['optimum']['mean']:10.1f} n_off {s['optimum_n_off']['mean']:4.1f} pays {s['offloading_pays']}")
    with open(os.path.join(RES, "gate_offloading.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("ceilings", ceilings, "plateau PSR", results["plateau_psr_30db"])
    print("wrote results/gate_offloading.json")


if __name__ == "__main__":
    main()
