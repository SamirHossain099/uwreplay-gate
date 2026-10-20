"""Every number the paper quotes, from results/ into results/manuscript_numbers.json (standing rule 4).

    python manuscript_numbers.py && python render_manuscript.py

Each token is {"value": number, "text": string as printed}. Nothing here is typed from memory: every value is read
from a results file written by a script under src/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
SITES = ["blue", "black", "purple", "yellow", "red", "green", "pink"]


def j(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def main():
    N = {}

    def put(k, v, text=None, fmt="{:.1f}"):
        N[k] = {"value": v, "text": text if text is not None else fmt.format(v)}

    def pct(k, v):
        put(k, v, f"{v:.1f}")

    def comma(k, v):
        put(k, v, f"{int(round(v)):,}")

    def p3(k, v):
        put(k, v, f"{v:.3f}")

    # ---- link budgets (F2) and the JSAC economy (F3, F6)
    lb = j("linkbudget.json")
    js = {r["d_m"]: r for r in lb["jsac"]["device_to_auv"]}
    fr = {r["d_m"]: r for r in lb["frontiers"]["sensor_to_auv"]}
    put("jsac_snr_10m", j("gate_offloading.json")["paper_snr_db_full_band"]["10"])
    put("jsac_snr_100m", js[100]["snr_db"])
    put("jsac_snr_1km", js[1000]["snr_db"])
    put("jsac_rate_1km_kbps", js[1000]["rate_bps"] / 1e3)
    put("fr_snr_1km", fr[1000]["snr_db"])
    put("fr_rate_1km_kbps", fr[1000]["rate_bps"] / 1e3)
    put("fr_rate_100m_kbps", fr[100]["rate_bps"] / 1e3)
    go = j("gate_offloading.json")
    put("n_devices", go["n_dev"], str(go["n_dev"]))
    put("n_layouts", go["n_layouts"], str(go["n_layouts"]))
    comma("rstar_bps", go["breakeven_rate_bps"]["median"])
    put("rstar_kbps", go["breakeven_rate_bps"]["median"] / 1e3, f"{go['breakeven_rate_bps']['median'] / 1e3:.0f}")
    put("rstar_min_kbps", go["breakeven_rate_bps"]["min"] / 1e3, f"{go['breakeven_rate_bps']['min'] / 1e3:.0f}")
    put("paper_rate_kbps", go["parametric_rate_bps_example"]["fullband"] / 1e3, f"{go['parametric_rate_bps_example']['fullband'] / 1e3:.0f}")
    put("rstar_margin", go["parametric_rate_bps_example"]["fullband"] / go["breakeven_rate_bps"]["median"], fmt="{:.2f}")
    arms = go["arms"]["fullband"]
    put("off_full_param", arms["parametric"]["full"]["mean"])
    put("off_opt_param", arms["parametric"]["optimum"]["mean"])
    put("off_opt_sd", arms["parametric"]["optimum"]["sd"])
    put("off_noff_param", arms["parametric"]["optimum_n_off"]["mean"], fmt="{:.2f}")
    put("off_full_red", arms["red_psr"]["full"]["mean"])
    put("off_opt_red", arms["red_psr"]["optimum"]["mean"])
    pct("off_opt_red_drop", 100 * (1 - arms["red_psr"]["optimum"]["mean"] / arms["parametric"]["optimum"]["mean"]))
    comma("off_full_blue_ceil", -arms["blue_ceil"]["full"]["mean"])
    comma("off_full_black_ceil", -arms["black_ceil"]["full"]["mean"])
    comma("off_full_green_ceil", -arms["green_ceil"]["full"]["mean"])
    put("off_shared_opt", go["arms"]["shared"]["parametric"]["optimum"]["mean"])
    put("off_shared_noff", go["arms"]["shared"]["parametric"]["optimum_n_off"]["mean"], fmt="{:.2f}")
    p3("red_plateau_psr", go["plateau_psr_30db"]["red"])

    # ---- FH-BFSK sweeps: gate waterfall shifts (F4) and pooled plateau success (F5, long records)
    s = j("summary.json")["psr"]
    for site in ("blue", "black", "red"):
        for rate in (160, 320, 640):
            put(f"shift_{site}_{rate}", s[f"{site}_r{rate}"]["shift_db"], f"{s[f'{site}_r{rate}']['shift_db']:+.1f}")
    p3("red_r160_psr_gate", s["red_r160"]["measured_psr_30db"])
    put("red_r160_n_gate", s["red_r160"]["n_trials"], str(s["red_r160"]["n_trials"]))
    coh = j("coherent/summary_coherent.json")
    lad = coh["ladder"]
    for site in SITES:
        fp = lad[site]["fh_pooled"]
        for rate, v in fp["psr_30db"].items():
            p3(f"fh_{site}_{rate}_psr", v)
            put(f"fh_{site}_{rate}_n", fp["n"][rate], str(fp["n"][rate]))
        put(f"fh_{site}_arq", fp["arq_bps"], f"{fp['arq_bps']:.0f}")
        put(f"fh_{site}_ceiling", fp["ceiling_bps"], f"{fp['ceiling_bps']:.0f}")
    put("fh_best_arq", max(lad[s_]["fh_arq_bps"] for s_ in SITES), f"{max(lad[s_]['fh_arq_bps'] for s_ in SITES):.0f}")
    put("ratio_rstar_fh", go["breakeven_rate_bps"]["median"] / max(lad[s_]["fh_arq_bps"] for s_ in SITES), fmt="{:.0f}")
    n_long = len(list((R / "long").glob("psr_*_r*.json")))
    put("n_long_files", n_long, str(n_long))

    # ---- coherent arm (F11)
    put("n_coh_files", len(coh["rows"]), str(len(coh["rows"])))
    for site in SITES:
        L = lad[site]
        put(f"coh_{site}_ceiling", L["coherent_ceiling_bps"], f"{L['coherent_ceiling_bps']:,.0f}" if L["coherent_ceiling_bps"] else "none")
        put(f"coh_{site}_arq", L["coherent_arq_bps"], f"{L['coherent_arq_bps']:,.0f}")
        put(f"coh_{site}_out", L["out_snr_ceiling_db"])
        put(f"coh_{site}_files", L["n_files"], str(L["n_files"]))
        p3(f"coh_{site}_arq_psr", L["coherent_arq_psr"])
        put(f"coh_{site}_arq_scheme", 0, L["coherent_arq_scheme"].replace("bpsk", "BPSK").replace("qpsk", "QPSK").replace(" ", " sym/s "))
    pooled_blue = lad["blue"]["coherent_pooled"]
    p3("coh_blue_2000b_psr", pooled_blue["2000_bpsk"]["psr"])
    p3("coh_blue_4000b_psr", pooled_blue["4000_bpsk"]["psr"])
    p3("coh_blue_4000q_psr", pooled_blue["4000_qpsk"]["psr"])
    put("coh_blue_n_min", min(v["n"] for v in pooled_blue.values() if v["all_files"]), f"{min(v['n'] for v in pooled_blue.values() if v['all_files']):,}")
    put("coh_blue_n_max", max(v["n"] for v in pooled_blue.values() if v["all_files"]), f"{max(v['n'] for v in pooled_blue.values() if v['all_files']):,}")
    p3("coh_red_4000b_psr", lad["red"]["coherent_pooled"]["4000_bpsk"]["psr"])
    p3("coh_yellow_4000b_psr", lad["yellow"]["coherent_pooled"]["4000_bpsk"]["psr"])
    p3("coh_purple_2000b_psr", lad["purple"]["coherent_pooled"]["2000_bpsk"]["psr"])
    blue1 = next(r for r in coh["rows"].values() if r["site"] == "blue" and r["file"] == "blue_1" and r["symbol_rate"] == 4000 and r["mod"] == "bpsk")
    p3("coh_blue1_4000b_psr", blue1["measured_psr_30db"])
    blue1k = next(r for r in coh["rows"].values() if r["site"] == "blue" and r["file"] == "blue_1" and r["symbol_rate"] == 1000 and r["mod"] == "bpsk")
    p3("coh_blue1_1000b_psr", blue1k["measured_psr_30db"])
    put("coh_best_arq", max(lad[s_]["coherent_arq_bps"] for s_ in SITES), f"{max(lad[s_]['coherent_arq_bps'] for s_ in SITES):,.0f}")
    put("coh_worst_arq", min(lad[s_]["coherent_arq_bps"] for s_ in SITES), f"{min(lad[s_]['coherent_arq_bps'] for s_ in SITES):,.0f}")
    put("ratio_rstar_coh", go["breakeven_rate_bps"]["median"] / max(lad[s_]["coherent_arq_bps"] for s_ in SITES), fmt="{:.0f}")
    put("ratio_paper_coh", go["parametric_rate_bps_example"]["fullband"] / max(lad[s_]["coherent_arq_bps"] for s_ in SITES), fmt="{:.0f}")
    # delay windows from file 1 of each site (10% to 99% of the stored impulse-response energy)
    for site in SITES:
        r0 = next(r for r in coh["rows"].values() if r["site"] == site and (r["file"] == f"{site}_1" or r["file"] == site))
        put(f"dw_{site}_lo", r0["delay_window_ms"]["t10"])
        put(f"dw_{site}_hi", r0["delay_window_ms"]["t99"])
    # feedback taps needed per symbol rate on file 1 (BPSK rows), for the sites that exceed the 160-tap cap
    for site in ("green", "pink"):
        for rs in (1000, 2000, 4000):
            rr = [r for r in coh["rows"].values() if r["site"] == site and r["symbol_rate"] == rs and r["mod"] == "bpsk" and r["file"] in (f"{site}_1", site)]
            if rr:
                put(f"fb_{site}_{rs}", rr[0]["fb_needed"], str(rr[0]["fb_needed"]))
    # Red record 1 at 160 chips/s: the plateau band from -5 dB upward (gate sweep)
    red_curve = s["red_r160"]["measured"]
    band = [v for k, v in red_curve.items() if float(k) >= -5.0]
    p3("red_r160_lo", min(band))
    p3("red_r160_hi", max(band))
    put("rec_blue_s", json.loads((R / "long" / "psr_blue_blue_1_r160.json").read_text(encoding="utf-8"))["record_s"], fmt="{:.0f}")
    put("f_auv_total_ghz", go["f_auv_total_hz"] / 1e9, fmt="{:.0f}")
    # pooled economy arms (2026-10-20): plateau success of the FH frame and goodput with retransmissions, every record
    for arm, tok in (("red_pool160_psr", "red160"), ("red_pool320_psr", "red320"), ("blue_pool160_psr", "blue160"), ("blue_pool320_psr", "blue320")):
        if arm in arms:
            put(f"off_full_pool_{tok}", arms[arm]["full"]["mean"])
            put(f"off_opt_pool_{tok}", arms[arm]["optimum"]["mean"])
    for arm, tok in (("black_fhpool", "black_fh"), ("blue_fhpool", "blue_fh"), ("black_coh", "black_coh"), ("blue_coh", "blue_coh")):
        if arm in arms:
            comma(f"off_full_{tok}", -arms[arm]["full"]["mean"])

    # ---- freshness (F7, F10)
    al = j("aoi_long.json")
    cases = [c for c in al["cases"] if "iid" in c["aoi"]]
    put("aoi_n_cases", len(cases), str(len(cases)))
    put("aoi_n_sources", al["n_sources"], str(al["n_sources"]))
    comma("aoi_horizon", al["horizon"])
    put("aoi_inversions", sum(1 for c in cases if c["aoi"]["iid"]["maf_gain_over_rr_pct"] > 0 and c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] < 0), fmt="{:.0f}")
    put("aoi_agree", sum(1 for c in cases if (c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] < 0) == (c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"] < 0)), fmt="{:.0f}")
    put("aoi_clustered", sum(1 for c in cases if c["burst"]["burst_ratio"] > 1.5), fmt="{:.0f}")
    slots = [c["burst"]["n_slots"] for c in cases]
    put("aoi_slots_min", min(slots), str(min(slots)))
    comma("aoi_slots_max", max(slots))

    def plateau(site, rate):
        cs = [c for c in cases if c["site"] == site and c["chip_rate"] == rate]
        return max(cs, key=lambda c: c["snr_db"])

    # Channel-limited subset (review 2026-10-20): cases at 10 dB and above, where the plateau and not the noise sets the
    # failures; and the plateau cases proper, one per (site, rate) whose highest in-range SNR is 30 dB.
    cl = [c for c in cases if c["snr_db"] >= 10]
    put("aoi_cl_cases", len(cl), str(len(cl)))
    put("aoi_cl_inversions", sum(1 for c in cl if c["aoi"]["iid"]["maf_gain_over_rr_pct"] > 0 and c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] < 0), fmt="{:.0f}")
    put("aoi_cl_agree", sum(1 for c in cl if (c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] < 0) == (c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"] < 0)), fmt="{:.0f}")
    put("aoi_cl_direct_inversions", sum(1 for c in cl if c["aoi"]["iid"]["maf_gain_over_rr_pct"] > 0 and c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"] < 0), fmt="{:.0f}")
    pairs = sorted({(c["site"], c["chip_rate"]) for c in cases})
    pl = [plateau(s_, r_) for s_, r_ in pairs]
    pl30 = [c for c in pl if c["snr_db"] >= 10]
    put("aoi_pl_cases", len(pl30), str(len(pl30)))
    put("aoi_pl_disagree", sum(1 for c in pl30 if (c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] < 0) != (c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"] < 0)), fmt="{:.0f}")
    put("aoi_pl_chain_loses", sum(1 for c in pl30 if c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] < 0), fmt="{:.0f}")
    put("aoi_pl_direct_loses", sum(1 for c in pl30 if c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"] < 0), fmt="{:.0f}")
    put("aoi_pl_both_lose", sum(1 for c in pl30 if c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] < 0 and c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"] < 0), fmt="{:.0f}")
    put("aoi_max_chainabs_all", max(-c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] for c in cases))
    put("aoi_max_chainabs_pl", max(-c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] for c in pl30))
    put("aoi_max_directabs_pl", max(-c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"] for c in pl30))
    put("aoi_sub_plateau_pairs", len(pl) - len(pl30), str(len(pl) - len(pl30)))
    put("aoi_pairs", len(pl), str(len(pl)))
    put("aoi_pl_agree", len(pl30) - sum(1 for c in pl30 if (c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] < 0) != (c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"] < 0)), fmt="{:.0f}")
    put("aoi_clustered_cl", sum(1 for c in cl if c["burst"]["burst_ratio"] > 1.5), fmt="{:.0f}")
    put("fh_blue_160_bps", lad["blue"]["fh_pooled"]["bits_per_s"]["160"])
    for site, rate in (("blue", 160), ("blue", 320), ("yellow", 320), ("yellow", 640), ("yellow", 1920), ("red", 160), ("red", 320),
                       ("purple", 320), ("purple", 640), ("purple", 1920), ("pink", 640), ("pink", 1920), ("green", 320), ("purple", 160)):
        c = plateau(site, rate)
        k = f"aoi_{site}{rate}"
        p3(f"{k}_psr", c["psr"])
        put(f"{k}_snr", c["snr_db"], fmt="{:.1f}")
        comma(f"{k}_slots", c["burst"]["n_slots"])
        comma(f"{k}_positions", c["burst"]["n_slots"] / 2)  # two noise draws per channel position
        put(f"{k}_run", c["burst"]["mean_fail_run"], fmt="{:.2f}")
        put(f"{k}_iidrun", c["burst"]["iid_mean_fail_run"], fmt="{:.2f}")
        put(f"{k}_ratio", c["burst"]["burst_ratio"], fmt="{:.2f}")
        for arm, tag in (("iid", "iid"), ("gilbert", "chain"), ("measured_tiled", "direct")):
            v = c["aoi"][arm]["maf_gain_over_rr_pct"]
            put(f"{k}_{tag}", v, f"{v:+.1f}")
            put(f"{k}_{tag}abs", abs(v), f"{abs(v):.1f}")  # magnitude, for "loses by X%"
        put(f"{k}_ba", c["aoi"]["gilbert"]["burst_aware_gain_over_maf_pct"])
        put(f"{k}_ba_direct", c["aoi"]["measured_tiled"]["burst_aware_gain_over_maf_pct"])

    # ---- the data-collection planner (F12)
    g2 = j("gate_offloading2.json")

    def at5(link, dn):
        return g2["results"][link][str(dn)]["rows"][4]

    put("dc_n_sensors", g2["params"]["n_sensors"], str(g2["params"]["n_sensors"]))
    put("dc_n_clusters", g2["params"]["n_clusters"], str(g2["params"]["n_clusters"]))
    put("dc_n_layouts", g2["params"]["n_layouts"], str(g2["params"]["n_layouts"]))
    for dn in (1, 10, 100):
        p = at5("paper", dn)
        pct(f"dc_paper_hover_{dn}", 100 * p["hover_share"])
        pct(f"dc_paper_gain_{dn}", p["planning_gain_pct"])
        comma(f"dc_paper_aoi_{dn}", p["aoi_mean_s"])
        comma(f"dc_paper_life_{dn}", p["lifetime_missions"])
    rows = {"black": "fh_black", "mid": "fh_green", "low": "fh_red", "blue": "fh_blue"}
    for tag, link in rows.items():
        p = at5(link, 10)
        pct(f"dc_{tag}_hover", 100 * p["hover_share"])
        comma(f"dc_{tag}_aoi", p["aoi_mean_s"])
        pct(f"dc_{tag}_gain", p["planning_gain_pct"])
        comma(f"dc_{tag}_life", p["lifetime_missions"])
    pct("dc_low_gain_100", at5("fh_red", 100)["planning_gain_pct"])
    pct("dc_low_hover_100", 100 * at5("fh_red", 100)["hover_share"])
    for site in SITES:
        p = at5(f"coh_{site}", 10)
        pct(f"dc_coh_{site}_hover", 100 * p["hover_share"])
        pct(f"dc_coh_{site}_gain", p["planning_gain_pct"])
        comma(f"dc_coh_{site}_life", p["lifetime_missions"])
        comma(f"dc_coh_{site}_aoi", p["aoi_mean_s"])
    coh100 = [at5(f"coh_{s_}", 100) for s_ in SITES]
    pct("dc_coh_hover100_min", 100 * min(p["hover_share"] for p in coh100))
    pct("dc_coh_hover100_max", 100 * max(p["hover_share"] for p in coh100))
    pct("dc_coh_gain100_min", min(p["planning_gain_pct"] for p in coh100))
    pct("dc_coh_gain100_max", max(p["planning_gain_pct"] for p in coh100))
    coh10 = [at5(f"coh_{s_}", 10) for s_ in SITES]
    pct("dc_coh_hover10_min", 100 * min(p["hover_share"] for p in coh10))
    pct("dc_coh_hover10_max", 100 * max(p["hover_share"] for p in coh10))
    pct("dc_coh_gain10_min", min(p["planning_gain_pct"] for p in coh10))
    pct("dc_coh_gain10_max", max(p["planning_gain_pct"] for p in coh10))
    comma("dc_coh_life10_min", min(p["lifetime_missions"] for p in coh10))
    comma("dc_coh_life10_max", max(p["lifetime_missions"] for p in coh10))
    # FH rows of the planner at the pooled goodput with retransmissions (the links the script now uses for fh_*)
    for site in SITES:
        p = at5(f"fh_{site}", 10)
        pct(f"dc_fh_{site}_hover", 100 * p["hover_share"])
        comma(f"dc_fh_{site}_aoi", p["aoi_mean_s"])
        pct(f"dc_fh_{site}_gain", p["planning_gain_pct"])
        comma(f"dc_fh_{site}_life", p["lifetime_missions"])
        put(f"dc_fh_{site}_bps", g2["links"][f"fh_{site}"]["bps"], fmt="{:.0f}")
    pct("dc_fh_red_gain_100", at5("fh_red", 100)["planning_gain_pct"])
    pct("dc_fh_red_hover_100", 100 * at5("fh_red", 100)["hover_share"])
    put("dc_packet_ms_paper", 1024 / g2["params"]["paper_rate_bps_at"]["1000"] * 1e3, fmt="{:.0f}")
    put("dc_packet_s_80", 1024 / 80.0, fmt="{:.0f}")
    put("dc_energy_mj_paper", 2.8 * 1024 / g2["params"]["paper_rate_bps_at"]["1000"] * 1e3, fmt="{:.0f}")
    put("dc_energy_j_80", 2.8 * 1024 / 80.0, fmt="{:.0f}")
    put("dc_energy_ratio", (2.8 * 1024 / 80.0) / (2.8 * 1024 / g2["params"]["paper_rate_bps_at"]["1000"]), f"{(g2['params']['paper_rate_bps_at']['1000'] / 80.0):,.0f}")
    put("dc_kstar", at5("paper", 10)["k"], "5")

    # ---- the probe (F13)
    cs = j("cheap_stats.json")
    p3("pstar", cs["breakeven_psr"])
    put("probe_s", cs["probe_s"], f"{cs['probe_s']:.0f}")
    put("probe_n", cs["n_points"], str(cs["n_points"]))
    put("probe_n_aoi", cs["n_points_aoi"], str(cs["n_points_aoi"]))
    r = cs["rules"]
    pct("probe_d1_agree", 100 * r["D1_probe_psr_at_pstar"]["accuracy"])
    pct("probe_d1_psr", 100 * r["D1_psr"]["accuracy"])
    pct("probe_d1_run", 100 * r["D1_mean_fail_run"]["accuracy"])
    pct("probe_d1_ratio", 100 * r["D1_burst_ratio"]["accuracy"])
    put("probe_d1_ratio_n", r["D1_burst_ratio"]["n"], str(r["D1_burst_ratio"]["n"]))
    pct("probe_d1_power", 100 * r["D1_power_var_db"]["accuracy"])
    put("probe_fp", cs["d1_confusion"]["fp"], str(cs["d1_confusion"]["fp"]))
    put("probe_fn", cs["d1_confusion"]["fn"], str(cs["d1_confusion"]["fn"]))
    fp = [p for p in cs["points"] if p["offload_pays_probe"] and not p["offload_pays_full"]]
    put("probe_fp_blue", sum(p["site"] == "blue" for p in fp), str(sum(p["site"] == "blue" for p in fp)))
    put("probe_fp_red", sum(p["site"] == "red" for p in fp), str(sum(p["site"] == "red" for p in fp)))
    g = cs["probe_vs_full_abs_psr_gap"]
    p3("gap_blue", g["blue"]["mean"])
    p3("gap_red", g["red"]["mean"])
    p3("gap_moored_max", max(g[s_]["mean"] for s_ in ("green", "pink", "purple", "yellow")))
    by = cs["d2_by_site"]
    for site in by:
        put(f"d2_{site}_wins", by[site]["maf_wins"], str(by[site]["maf_wins"]))
        put(f"d2_{site}_n", by[site]["n"], str(by[site]["n"]))
    put("d2_purple_run", by["purple"]["full_run_mean"], fmt="{:.2f}")
    put("d2_purple_ratio", by["purple"]["full_ratio_mean"], fmt="{:.2f}")
    f = cs["rules_full_record"]
    pct("d2_full_runiid", 100 * f["full_run_over_iid_ge_1p5_loses"]["accuracy"])
    pct("d2_full_ratio", 100 * f["full_ratio_ge_1p5_loses"]["accuracy"])
    pct("d2_full_run", 100 * f["full_run_ge_2_loses"]["accuracy"])
    pr = cs["rules_probe_with_3_failures"]
    put("d2_probe_n", pr["n"], str(pr["n"]))
    pct("d2_probe_majority", 100 * pr["majority_baseline"])
    pct("d2_probe_ratio2", 100 * pr["probe_ratio_ge_2_loses"]["accuracy"])
    pct("d2_probe_ratio15", 100 * pr["probe_ratio_ge_1p5_loses"]["accuracy"])
    pct("d2_probe_runiid", 100 * pr["probe_run_over_iid_ge_1p5_loses"]["accuracy"])
    put("d2_probe_blind", cs["n_points_aoi"] - pr["n"], str(cs["n_points_aoi"] - pr["n"]))
    pct("d2_loso_lo", 100 * min(r[k]["accuracy"] for k in ("D2_mean_fail_run", "D2_burst_ratio", "D2_psr", "D2_power_var_db")))
    pct("d2_loso_hi", 100 * max(r[k]["accuracy"] for k in ("D2_mean_fail_run", "D2_burst_ratio", "D2_psr", "D2_power_var_db")))
    d2 = [p for p in cs["points"] if p["maf_beats_rr"] is not None]
    pct("d2_mobility_prior", 100 * np.mean([((p["site"] not in ("blue", "red")) == p["maf_beats_rr"]) for p in d2]))
    pct("d1_majority", 100 * cs["d1_majority_baseline"])
    put("n_records", cs["n_distinct_records"], str(cs["n_distinct_records"]))
    put("n_records_d2", cs["n_distinct_records_d2"], str(cs["n_distinct_records_d2"]))
    put("n_sites_d1", cs["n_sites_d1"], str(cs["n_sites_d1"]))
    put("n_sites_d2", cs["n_sites_d2"], str(cs["n_sites_d2"]))
    pct("d2_lf_full", 100 * cs["d2_loso_family_full_record"]["accuracy"])
    put("d2_lf_full_n", cs["d2_loso_family_full_record"]["n"], str(cs["d2_loso_family_full_record"]["n"]))
    pct("d2_lf_probe", 100 * cs["d2_loso_family_probe"]["accuracy"])
    put("d2_lf_probe_n", cs["d2_loso_family_probe"]["n"], str(cs["d2_loso_family_probe"]["n"]))

    # ---- artifact placeholders (filled at release)
    put("artifact_url", 0, "https://github.com/SamirHossain099/uwreplay-gate")
    put("artifact_doi", 0, "10.5281/zenodo.23493739")  # concept DOI, resolves to the latest release (Zenodo record 2026-10-20)

    (R / "manuscript_numbers.json").write_text(json.dumps(N, indent=1), encoding="utf-8")
    print(f"wrote results/manuscript_numbers.json with {len(N)} tokens")


if __name__ == "__main__":
    main()
