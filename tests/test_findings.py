"""Pin every number quoted in FINDINGS.md to results/*.json (standing rule 4). Run with the project venv."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as fh:
        return json.load(fh)


def close(a, b, tol):
    return abs(a - b) <= tol


def test_linkbudget():
    lb = load("linkbudget.json")
    j = {r["d_m"]: r for r in lb["jsac"]["device_to_auv"]}
    assert close(lb["jsac"]["thorp_db_per_km"], 8.28, 0.005)
    assert close(lb["jsac"]["noise_psd_db_at_f"], 21.3, 0.05)
    assert close(j[100]["snr_db"], 56.5, 0.05) and close(j[100]["rate_bps"], 187.6e3, 50)
    assert close(j[1000]["snr_db"], 34.0, 0.05) and close(j[1000]["rate_bps"], 113.1e3, 50)
    f = {r["d_m"]: r for r in lb["frontiers"]["sensor_to_auv"]}
    assert close(lb["frontiers"]["thorp_db_per_km"], 6.53, 0.005)
    assert close(f[1000]["snr_db"], 38.4, 0.05) and close(f[1000]["rate_bps"], 204.0e3, 50)


def test_psr_table():
    s = load("summary.json")["psr"]
    assert close(s["blue_r160"]["measured_psr_30db"], 1.0, 1e-9) and s["blue_r160"]["n_trials"] == 72
    assert close(s["black_r160"]["measured_psr_30db"], 1.0, 1e-9) and s["black_r160"]["n_trials"] == 50
    assert close(s["red_r160"]["measured_psr_30db"], 0.766, 0.0005) and s["red_r160"]["n_trials"] == 64
    for snr in ("-2.5", "0.0", "5.0", "10.0", "15.0", "20.0", "30.0"):
        assert 0.765 <= s["red_r160"]["measured"][snr] <= 0.845
        assert s["red_r160"]["awgn"][snr] == 1.0
    assert close(s["blue_r160"]["shift_db"], 3.6, 0.05) and close(s["blue_r320"]["shift_db"], 2.3, 0.05) and close(s["blue_r640"]["shift_db"], 2.1, 0.05)
    assert close(s["black_r160"]["shift_db"], -0.9, 0.05) and close(s["black_r320"]["shift_db"], -2.1, 0.05) and close(s["black_r640"]["shift_db"], -3.3, 0.05)
    assert close(s["red_r160"]["shift_db"], 0.6, 0.05) and close(s["red_r320"]["shift_db"], 1.8, 0.05) and close(s["red_r640"]["shift_db"], 4.2, 0.05)
    assert s["red_r1920"]["measured_snr_at_psr50"] != s["red_r1920"]["measured_snr_at_psr50"]  # nan: never reaches 0.5
    assert close(s["blue_r640"]["measured_psr_30db"], 1.0, 1e-9) and close(s["blue_r1920"]["measured_psr_30db"], 0.867, 0.0005)
    for r in ("160", "320", "640", "1920"):
        assert s[f"black_r{r}"]["measured_psr_30db"] == 1.0
        assert all(v == 1.0 for v in (s[f"blue_r{r}"]["awgn_psr_30db"], s[f"red_r{r}"]["awgn_psr_30db"], s[f"black_r{r}"]["awgn_psr_30db"]))
    assert close(s["red_r320"]["measured_psr_30db"], 0.711, 0.0005) and close(s["red_r640"]["measured_psr_30db"], 0.667, 0.0005)
    assert close(s["red_r1920"]["measured_psr_30db"], 0.244, 0.0005)
    assert close(s["blue_r160"]["p_rx_mean"], 0.62, 0.005) and close(s["red_r160"]["p_rx_mean"], 1.02, 0.005)
    assert close(s["black_r160"]["p_rx_mean"], 0.67, 0.005)


def test_offloading():
    g = load("gate_offloading.json")
    assert g["n_dev"] == 12 and g["n_layouts"] == 40
    assert close(g["breakeven_rate_bps"]["median"], 171221, 1) and close(g["breakeven_rate_bps"]["min"], 93958, 1)
    assert close(g["parametric_rate_bps_example"]["fullband"], 200266, 1)
    assert close(g["paper_snr_db_full_band"]["10"], 72.2, 0.05) and close(g["paper_snr_db_full_band"]["100"], 56.5, 0.05)
    fb = g["arms"]["fullband"]
    p = fb["parametric"]
    assert close(p["optimum"]["mean"], 134.9, 0.05) and close(p["optimum"]["sd"], 80.9, 0.05)
    assert close(p["full"]["mean"], 31.9, 0.05) and close(p["random"]["mean"], -13.2, 0.05) and close(p["partial"]["mean"], -121.5, 0.05)
    assert close(p["optimum_n_off"]["mean"], 6.85, 0.005)
    assert p["ranking"][:3] == ["optimum", "full", "non_offloading"]
    sh = g["arms"]["shared"]["parametric"]
    assert close(sh["optimum"]["mean"], 1.8, 0.05) and close(sh["optimum_n_off"]["mean"], 0.38, 0.01)
    r = fb["red_psr"]
    assert close(r["full"]["mean"], -22.4, 0.05) and close(r["optimum"]["mean"], 107.7, 0.05)
    assert r["ranking"].index("non_offloading") < r["ranking"].index("full")
    assert close(100 * (r["optimum"]["mean"] / p["optimum"]["mean"] - 1), -20, 0.5)
    for ch in ("blue_psr", "black_psr"):
        assert close(fb[ch]["optimum"]["mean"], 134.9, 0.05)
    # Record-1 ceilings under the preamble-inclusive payload rate (C4): 640 chips/s on Blue, 1920 on Black, none on Red.
    assert close(g["goodput_ceiling_bps"]["blue"], 232.7, 0.05) and close(g["goodput_ceiling_bps"]["black"], 698.2, 0.05) and g["goodput_ceiling_bps"]["red"] == 0
    assert close(fb["blue_ceil"]["full"]["mean"], -155524, 1) and close(fb["black_ceil"]["full"]["mean"], -51702, 1)
    assert fb["blue_ceil"]["optimum_n_off"]["mean"] == 0 and fb["blue_ceil"]["ranking"][0] == "non_offloading" and fb["blue_ceil"]["ranking"][-1] == "full"
    assert close(g["plateau_psr_30db"]["red"], 0.766, 0.0005)


def _case(ga, ch, rate, snr):
    for c in ga["cases"]:
        if c["channel"] == ch and c["chip_rate"] == rate and c["snr_db"] == snr:
            return c
    raise KeyError((ch, rate, snr))


def test_aoi():
    ga = load("gate_aoi.json")
    assert ga["n_sources"] == 4 and ga["horizon"] == 100_000
    c = _case(ga, "red", 160, 30.0)
    assert close(c["psr"], 0.766, 0.0005) and c["burst"]["n_slots"] == 64
    assert close(c["burst"]["burst_ratio"], 2.63, 0.005) and close(c["burst"]["mean_fail_run"], 2.14, 0.005) and close(c["burst"]["iid_mean_fail_run"], 1.31, 0.005)
    a = c["aoi"]
    assert close(a["iid"]["maf"], 3.27, 0.005) and close(a["iid"]["round_robin"], 3.73, 0.005) and close(a["iid"]["maf_gain_over_rr_pct"], 12.4, 0.05)
    assert close(a["gilbert"]["maf"], 4.63, 0.005) and close(a["gilbert"]["round_robin"], 4.09, 0.005) and close(a["gilbert"]["maf_gain_over_rr_pct"], -13.0, 0.05)
    assert close(a["gilbert"]["maf_burst_aware"], 3.92, 0.005) and close(a["gilbert"]["burst_aware_gain_over_maf_pct"], 15.3, 0.05)
    assert close(a["measured_tiled"]["maf_gain_over_rr_pct"], 22.7, 0.05)
    c = _case(ga, "red", 320, 30.0)
    assert close(c["psr"], 0.711, 0.0005) and close(c["burst"]["burst_ratio"], 2.40, 0.005)
    assert close(c["aoi"]["iid"]["maf_gain_over_rr_pct"], 14.6, 0.05) and close(c["aoi"]["gilbert"]["maf_gain_over_rr_pct"], -25.2, 0.05)
    assert close(c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"], -26.3, 0.05)
    c = _case(ga, "blue", 320, -2.5)
    assert close(c["psr"], 0.822, 0.0005) and close(c["burst"]["burst_ratio"], 3.21, 0.005) and c["burst"]["n_slots"] == 90
    assert close(c["aoi"]["iid"]["maf_gain_over_rr_pct"], 9.4, 0.05) and close(c["aoi"]["gilbert"]["maf_gain_over_rr_pct"], -10.1, 0.05)
    assert close(c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"], 7.7, 0.05)
    c = _case(ga, "black", 160, -10.0)
    assert close(c["psr"], 0.86, 0.0005) and close(c["burst"]["burst_ratio"], 0.0, 1e-9)
    assert c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] > 0
    c = _case(ga, "green", 1920, 30.0)
    assert close(c["psr"], 0.897, 0.0005) and c["burst"]["n_slots"] == 360 and close(c["burst"]["burst_ratio"], 7.10, 0.005)
    assert close(c["burst"]["mean_fail_run"], 3.70, 0.005) and close(c["burst"]["iid_mean_fail_run"], 1.11, 0.005)
    assert close(c["aoi"]["iid"]["maf_gain_over_rr_pct"], 5.6, 0.05) and close(c["aoi"]["gilbert"]["maf_gain_over_rr_pct"], -24.2, 0.05)
    assert close(c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"], -26.5, 0.05)
    c = _case(ga, "green", 1920, 5.0)
    assert close(c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"], -8.2, 0.05)


def test_green():
    s = load("summary.json")["psr"]
    assert s["green_r160"]["n_trials"] == 280 and s["green_r320"]["n_trials"] == 360
    assert s["green_r160"]["measured_psr_30db"] == 1.0 and s["green_r320"]["measured_psr_30db"] == 1.0
    assert close(s["green_r640"]["measured_psr_30db"], 0.489, 0.0005) and close(s["green_r1920"]["measured_psr_30db"], 0.897, 0.0005)
    assert close(s["green_r160"]["shift_db"], 2.8, 0.05) and close(s["green_r320"]["shift_db"], 1.8, 0.05) and close(s["green_r1920"]["shift_db"], 0.0, 0.05)
    assert s["green_r640"]["measured_snr_at_psr50"] != s["green_r640"]["measured_snr_at_psr50"]
    assert close(s["green_r160"]["p_rx_mean"], 0.61, 0.005)
    g = load("gate_offloading.json")
    assert close(g["goodput_ceiling_bps"]["green"], 116.4, 0.05)  # 320 chips/s at the preamble-inclusive rate (C4)
    assert close(g["arms"]["fullband"]["green_ceil"]["full"]["mean"], -311256, 1) and g["arms"]["fullband"]["green_ceil"]["optimum_n_off"]["mean"] == 0
    assert close(g["arms"]["fullband"]["green_psr"]["optimum"]["mean"], 134.9, 0.05)
    ga = load("gate_aoi.json")
    c = _case(ga, "green", 640, 30.0)
    assert close(c["burst"]["mean_fail_run"], 30.67, 0.005)


def test_no_em_dash():
    for fn in ("FINDINGS.md", "NOVELTY.md", "PLAN.md"):
        p = os.path.join(ROOT, fn)
        if os.path.exists(p):
            assert chr(0x2014) not in open(p, encoding="utf-8").read(), fn
    for dirpath, _, files in os.walk(os.path.join(ROOT, "src")):
        for f in files:
            if f.endswith(".py"):
                assert chr(0x2014) not in open(os.path.join(dirpath, f), encoding="utf-8").read(), f


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)


def test_aoi_long():
    d = load("aoi_long.json")
    cases = [c for c in d["cases"] if "iid" in c["aoi"]]
    assert len(cases) == 112
    assert sum(1 for c in cases if c["aoi"]["iid"]["maf_gain_over_rr_pct"] > 0) == 112
    assert sum(1 for c in cases if c["aoi"]["iid"]["maf_gain_over_rr_pct"] > 0 and c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] < 0) == 86
    assert sum(1 for c in cases if (c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] < 0) == (c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"] < 0)) == 83
    assert sum(1 for c in cases if c["burst"]["burst_ratio"] > 1.5) == 63

    def plateau(site, rate):
        cs = [c for c in cases if c["site"] == site and c["chip_rate"] == rate]
        return max(cs, key=lambda c: c["snr_db"])

    b = plateau("blue", 160)
    assert close(b["psr"], 0.847, 0.0005) and b["burst"]["n_slots"] == 1440
    assert close(b["burst"]["mean_fail_run"], 7.86, 0.005) and close(b["burst"]["iid_mean_fail_run"], 1.18, 0.005)
    assert close(b["aoi"]["iid"]["maf_gain_over_rr_pct"], 8.4, 0.05) and close(b["aoi"]["gilbert"]["maf_gain_over_rr_pct"], -77.6, 0.05)
    assert close(b["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"], -44.1, 0.05)
    assert close(b["aoi"]["gilbert"]["burst_aware_gain_over_maf_pct"], 38.3, 0.05)
    b2 = plateau("blue", 320)
    assert b2["burst"]["n_slots"] == 2320 and close(b2["aoi"]["gilbert"]["maf_gain_over_rr_pct"], -86.9, 0.05)
    y = plateau("yellow", 640)
    assert y["burst"]["n_slots"] == 982 and close(y["aoi"]["gilbert"]["maf_gain_over_rr_pct"], -59.0, 0.05)
    r = plateau("red", 320)
    assert r["burst"]["n_slots"] == 424 and close(r["aoi"]["gilbert"]["maf_gain_over_rr_pct"], -20.1, 0.05)
    g = plateau("green", 320)
    assert g["burst"]["n_slots"] == 7168 and close(g["burst"]["mean_fail_run"], 1.15, 0.005) and g["aoi"]["gilbert"]["maf_gain_over_rr_pct"] > 0
    for site, rate in (("purple", 160), ("purple", 320)):
        assert plateau(site, rate)["aoi"]["gilbert"]["maf_gain_over_rr_pct"] > 0
    # Run length separates the groups at the plateau; the conditional-failure ratio does not.
    pl = [plateau(c["site"], c["chip_rate"]) for c in {(c["site"], c["chip_rate"]): c for c in cases}.values()]
    assert all(c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] < 0 for c in pl if c["burst"]["mean_fail_run"] >= 2.0)
    assert all(c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] > 0 for c in pl if c["burst"]["mean_fail_run"] < 1.2)


def test_offloading2():
    g = load("gate_offloading2.json")
    assert g["params"]["n_sensors"] == 150 and g["params"]["n_clusters"] == 25 and g["params"]["n_layouts"] == 20
    assert close(g["params"]["paper_rate_bps_at"]["1000"], 203955, 1) and close(g["params"]["paper_rate_bps_at"]["100"], 314898, 1)

    def at5(link, d_n):
        return g["results"][link][str(d_n)]["rows"][4]

    p = at5("paper", 10)
    assert close(100 * p["hover_share"], 0.5, 0.05) and close(p["aoi_mean_s"], 993, 0.5) and close(p["planning_gain_pct"], 13.8, 0.05)
    assert close(at5("paper", 1)["planning_gain_pct"], 17.7, 0.05) and close(at5("paper", 100)["planning_gain_pct"], 13.5, 0.05)
    assert close(p["lifetime_missions"], 289352, 1)
    # FH rows at the pooled goodput with retransmissions under the preamble-inclusive rate (C4, review 2026-10-20).
    exp_fh = dict(fh_black=(698.2, 19.6, 1238, 11.3, 974), fh_yellow=(266.3, 38.7, 1636, 8.7, 372), fh_purple=(178.0, 48.5, 1955, 7.3, 248),
                  fh_green=(116.4, 59.0, 2466, 5.7, 162), fh_pink=(116.4, 59.0, 2466, 5.7, 162), fh_blue=(94.4, 63.9, 2808, 5.0, 132),
                  fh_red=(90.3, 65.0, 2891, 4.8, 126))
    for link, (bps, hover, aoi, gain, life) in exp_fh.items():
        r = at5(link, 10)
        assert close(g["links"][link]["bps"], bps, 0.05), link
        assert close(100 * r["hover_share"], hover, 0.05) and close(r["aoi_mean_s"], aoi, 0.5), link
        assert close(r["planning_gain_pct"], gain, 0.05) and close(r["lifetime_missions"], life, 0.5), link
    assert close(at5("fh_red", 100)["planning_gain_pct"], 0.9, 0.05) and close(100 * at5("fh_red", 100)["hover_share"], 94.9, 0.05)
    for link in g["results"]:
        for d_n in ("1", "10", "100"):
            assert g["results"][link][d_n]["k_star"] == 5
    exp = dict(coh_black=(6.3, 1057, 13.1, 3701), coh_blue=(8.6, 1084, 12.8, 2602), coh_red=(10.5, 1107, 12.5, 2073),
               coh_yellow=(14.1, 1156, 12.0, 1460), coh_pink=(20.4, 1250, 11.2, 925), coh_green=(29.8, 1423, 9.9, 555),
               coh_purple=(34.2, 1520, 9.3, 453))
    for link, (hover, aoi, gain, life) in exp.items():
        r = at5(link, 10)
        assert close(100 * r["hover_share"], hover, 0.05) and close(r["aoi_mean_s"], aoi, 0.5), link
        assert close(r["planning_gain_pct"], gain, 0.05) and close(r["lifetime_missions"], life, 0.5), link
    assert close(100 * at5("coh_green", 100)["hover_share"], 81.0, 0.05) and close(at5("coh_green", 100)["planning_gain_pct"], 3.0, 0.05)
    assert close(100 * at5("coh_black", 100)["hover_share"], 39.1, 0.05) and close(at5("coh_black", 100)["planning_gain_pct"], 8.8, 0.05)


def test_cheap_stats():
    c = load("cheap_stats.json")
    assert c["probe_s"] == 30.0 and close(c["breakeven_psr"], 0.862, 0.0005)
    assert c["n_points"] == 160 and c["n_points_aoi"] == 61
    r = c["rules"]
    assert close(100 * r["D1_probe_psr_at_pstar"]["accuracy"], 87.5, 0.05)
    assert close(100 * r["D1_psr"]["accuracy"], 86.2, 0.05) and close(100 * r["D1_mean_fail_run"]["accuracy"], 89.4, 0.05)
    assert close(100 * r["D1_burst_ratio"]["accuracy"], 90.0, 0.05) and r["D1_burst_ratio"]["n"] == 70
    assert close(100 * r["D1_power_var_db"]["accuracy"], 65.0, 0.05)
    assert c["d1_confusion"]["fp"] == 18 and c["d1_confusion"]["fn"] == 2 and c["d1_confusion"]["fp_sites"] == ["blue", "pink", "purple", "red"]
    fp = [p for p in c["points"] if p["offload_pays_probe"] and not p["offload_pays_full"]]
    assert sum(p["site"] == "blue" for p in fp) == 13 and sum(p["site"] == "red" for p in fp) == 3
    g = c["probe_vs_full_abs_psr_gap"]
    assert close(g["blue"]["mean"], 0.143, 0.0005) and close(g["red"]["mean"], 0.114, 0.0005) and g["purple"]["mean"] < 0.03
    by = c["d2_by_site"]
    assert (by["blue"]["maf_wins"], by["blue"]["n"]) == (0, 17) and (by["red"]["maf_wins"], by["red"]["n"]) == (0, 5)
    assert (by["pink"]["maf_wins"], by["pink"]["n"]) == (0, 3) and (by["purple"]["maf_wins"], by["purple"]["n"]) == (29, 29)
    assert (by["yellow"]["maf_wins"], by["yellow"]["n"]) == (6, 7)
    assert close(by["purple"]["full_run_mean"], 3.82, 0.005) and close(by["purple"]["full_ratio_mean"], 1.06, 0.005)
    f = c["rules_full_record"]
    assert close(100 * f["full_run_over_iid_ge_1p5_loses"]["accuracy"], 98.4, 0.05)
    assert close(100 * f["full_ratio_ge_1p5_loses"]["accuracy"], 90.2, 0.05)
    assert close(100 * f["full_run_ge_2_loses"]["accuracy"], 70.5, 0.05)
    p = c["rules_probe_with_3_failures"]
    assert p["n"] == 49 and close(100 * p["majority_baseline"], 69.4, 0.05)
    assert close(100 * p["probe_ratio_ge_2_loses"]["accuracy"], 93.9, 0.05)
    assert close(100 * p["probe_ratio_ge_1p5_loses"]["accuracy"], 89.8, 0.05)
    assert close(100 * p["probe_run_over_iid_ge_1p5_loses"]["accuracy"], 87.8, 0.05)
    assert close(100 * r["D2_fixed_run2"]["accuracy"], 45.9, 0.05) and close(100 * r["D2_majority_baseline"], 57.4, 0.05)
    # Added after the review (C6, C7): the D1 majority baseline, the record counts, and the held-out rule family.
    assert close(100 * c["d1_majority_baseline"], 61.3, 0.05)
    assert c["n_distinct_records"] == 49 and c["n_distinct_records_d2"] == 34 and c["n_sites_d1"] == 6 and c["n_sites_d2"] == 5
    assert close(100 * c["d2_loso_family_full_record"]["accuracy"], 100.0, 0.05) and c["d2_loso_family_full_record"]["n"] == 61
    assert close(100 * c["d2_loso_family_probe"]["accuracy"], 85.7, 0.05) and c["d2_loso_family_probe"]["n"] == 49


def test_coherent_ladder():
    with open(os.path.join(RES, "coherent", "summary_coherent.json"), encoding="utf-8") as fh:
        c = json.load(fh)
    lad = c["ladder"]
    assert len(c["rows"]) == 168
    assert lad["blue"]["n_files"] == 20 and lad["red"]["n_files"] == 4 and lad["yellow"]["n_files"] == 6 and lad["purple"]["n_files"] == 15
    exp_ceiling = dict(black=2653, red=1590, pink=663, green=398, blue=0, purple=0, yellow=0)
    exp_arq = dict(black=2653, blue=1865, red=1486, yellow=1047, pink=663, green=398, purple=325)
    exp_fh_arq = dict(black=698, blue=94, red=90, yellow=266, pink=116, green=116, purple=178)  # preamble-inclusive rates (C4)
    exp_out = dict(black=30.2, blue=15.8, yellow=14.6, red=10.5, pink=9.3, purple=7.3, green=3.1)
    for site in exp_ceiling:
        assert close(lad[site]["coherent_ceiling_bps"], exp_ceiling[site], 0.5), site
        assert close(lad[site]["coherent_arq_bps"], exp_arq[site], 0.5), site
        assert close(lad[site]["fh_arq_bps"], exp_fh_arq[site], 0.5), site
        assert close(lad[site]["out_snr_ceiling_db"], exp_out[site], 0.05), site
    assert lad["green"]["any_fb_truncated"] and lad["pink"]["any_fb_truncated"] and not lad["blue"]["any_fb_truncated"]
    p = lad["blue"]["coherent_pooled"]
    assert close(p["2000_bpsk"]["psr"], 0.837, 0.0005) and p["2000_bpsk"]["all_files"] and p["2000_bpsk"]["n"] == 1240
    assert close(p["4000_bpsk"]["psr"], 0.834, 0.0005) and close(p["4000_qpsk"]["psr"], 0.703, 0.0005) and close(p["2000_qpsk"]["psr"], 0.787, 0.0005)
    assert close(p["1000_bpsk"]["psr"], 0.647, 0.0005) and not p["1000_bpsk"]["all_files"]
    r = lad["red"]["coherent_pooled"]
    assert close(r["4000_bpsk"]["psr"], 0.934, 0.0005) and r["4000_bpsk"]["n"] == 320 and close(r["2000_bpsk"]["psr"], 0.759, 0.0005)
    y = lad["yellow"]["coherent_pooled"]
    assert close(y["4000_bpsk"]["psr"], 0.658, 0.0005) and close(y["2000_bpsk"]["psr"], 0.656, 0.0005)
    assert lad["yellow"]["partial_file_best"] is not None and "1000 qpsk" in lad["yellow"]["partial_file_best"]
    pu = lad["purple"]["coherent_pooled"]
    assert close(pu["2000_bpsk"]["psr"], 0.408, 0.0005) and close(pu["1000_bpsk"]["psr"], 0.363, 0.0005)
    g = lad["green"]["coherent_pooled"]
    assert g["1000_bpsk"]["psr"] == 1.0 and g["2000_bpsk"]["psr"] == 0.0
    assert all(v["psr"] == 1.0 for v in lad["black"]["coherent_pooled"].values())
    # Break-even ratio quoted in F11.
    go = load("gate_offloading.json")
    assert 64.5 <= go["breakeven_rate_bps"]["median"] / max(v["coherent_arq_bps"] for v in lad.values()) <= 65.5
    assert 244.5 <= go["breakeven_rate_bps"]["median"] / max(v["fh_arq_bps"] for v in lad.values()) <= 245.5
    # Pooled economy arms added after the review (C4): the sign of full offloading at the pooled plateau success.
    arms = go["arms"]["fullband"]
    assert arms["red_pool320_psr"]["full"]["mean"] < 0 and arms["blue_pool320_psr"]["full"]["mean"] < 0
    assert arms["red_pool160_psr"]["full"]["mean"] > 0 and abs(arms["blue_pool160_psr"]["full"]["mean"]) < 1
    for arm in ("black_fhpool", "blue_fhpool", "black_coh", "blue_coh"):
        assert arms[arm]["optimum_n_off"]["mean"] == 0 and arms[arm]["full"]["mean"] < -10000
