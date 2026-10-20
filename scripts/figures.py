"""Figures for the manuscript (PLAN item 6). Each figure is one file, 600 dpi PNG plus vector PDF, with the numbers
it draws cached next to it as JSON (RESEARCH.md section 11). Legends sit outside the plotting area.

Figure 1  PSR versus nominal in-band SNR, measured (solid) and AWGN (dashed), one panel per site, lines per rate,
          from results/long/ pooled per (site, rate, SNR).
Figure 3  Max-age-first gain over round-robin at each site's plateau under i.i.d. erasures, the fitted chain and the
          direct replay, from results/aoi_long.json.
Figures 2 and 4 (rate ladder against R*, offloading profit under the link models) are added once the coherent
sweep and item 4 are in.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figs")
SITE_ORDER = ["blue", "black", "purple", "yellow", "red", "green", "pink"]
SITE_LABEL = {"blue": "Blue: N. Atlantic 2010,\nmobile, 3-7 km", "black": "Black: Mariana 2024,\nvertical, 8.7 km",
              "purple": "Purple: N. Atlantic 2008,\n15 m deep, 0.06-1 km", "yellow": "Yellow: KAM11 2011,\n3 and 7 km",
              "red": "Red: Singapore 2024,\nharbour, 0.1-0.4 km", "green": "Green: Norway 2024,\n0.27 km",
              "pink": "Pink: Japan 2022,\n14 km"}
RATES = [160, 320, 640, 1920]
COLORS = {160: "#1b5e9e", 320: "#d9772b", 640: "#2e8b57", 1920: "#8b2e8b"}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0})


def save(fig, name: str, data: dict) -> None:
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, f"{name}.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, f"{name}.pdf"), bbox_inches="tight")
    with open(os.path.join(FIGS, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    plt.close(fig)
    print("wrote", name)


def pooled_psr() -> dict:
    """{site: {rate: {snr: (measured psr, awgn psr, n)}}} pooled over files and elements with trial weights."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0, 0])))
    files = glob.glob(os.path.join(RES, "long", "psr_*_r*.json"))
    # Black has no long replays (its record gives 21 to 50 segments either way); its gate sweeps stand in.
    files += glob.glob(os.path.join(RES, "psr_black_r*.json"))
    for fn in files:
        m = re.search(r"psr_(\w+?)(?:_(\w+?))?(?:_e\d+-\d+)?_r(\d+)\.json$", os.path.basename(fn))
        site, rate = m.group(1), int(m.group(3))
        pj = json.load(open(fn, encoding="utf-8"))
        for snr in pj["snr_grid"]:
            mm, aa = pj["measured"][str(snr)], pj["awgn"][str(snr)]
            a = acc[site][rate][snr]
            a[0] += mm["psr"] * mm["n"]
            a[1] += aa["psr"] * aa["n"]
            a[2] += mm["n"]
            a[3] += aa["n"]
    out = {}
    for site in acc:
        out[site] = {}
        for rate in acc[site]:
            out[site][rate] = {snr: (v[0] / v[2], v[1] / v[3], v[2]) for snr, v in sorted(acc[site][rate].items())}
    return out


def figure_psr_sites() -> None:
    data = pooled_psr()
    sites = [s for s in SITE_ORDER if s in data]
    n = len(sites)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.1, 1.45 * nrow), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    cache = {}
    for i, site in enumerate(sites):
        ax = axes[i]
        cache[site] = {}
        for rate in RATES:
            if rate not in data[site]:
                continue
            snrs = sorted(data[site][rate])
            meas = [data[site][rate][s][0] for s in snrs]
            awgn = [data[site][rate][s][1] for s in snrs]
            ax.plot(snrs, meas, "-", color=COLORS[rate], marker="o", ms=2.2, label=f"{rate} chips/s, measured")
            ax.plot(snrs, awgn, "--", color=COLORS[rate], lw=0.8, label=f"{rate} chips/s, AWGN")
            cache[site][rate] = dict(snr_db=snrs, measured=meas, awgn=awgn, n=[data[site][rate][s][2] for s in snrs])
        ax.set_title(f"({chr(97 + i)}) {SITE_LABEL[site]}", fontsize=7, loc="left")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-16, 31)
        ax.grid(True, lw=0.3, alpha=0.5)
        if i % ncol == 0:
            ax.set_ylabel("Packet success rate")
        if i >= n - ncol:
            ax.set_xlabel("Nominal in-band SNR (dB)")
    for j in range(n, axes.size):
        axes[j].axis("off")
    seen = {}
    for ax in axes[:n]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    order = [f"{r} chips/s, {k}" for r in RATES for k in ("measured", "AWGN")]
    labels = [lab for lab in order if lab in seen]
    fig.legend([seen[lab] for lab in labels], labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04 - 0.02 * nrow))
    fig.tight_layout(h_pad=1.2, w_pad=0.4)
    save(fig, "fig1_psr_sites", cache)


def figure_aoi() -> None:
    d = json.load(open(os.path.join(RES, "aoi_long.json"), encoding="utf-8"))
    cases = [c for c in d["cases"] if "iid" in c["aoi"]]
    groups = defaultdict(list)
    for c in cases:
        groups[(c["site"], c["chip_rate"])].append(c)
    rows = []
    for site in SITE_ORDER:
        for rate in RATES:
            if (site, rate) in groups:
                c = max(groups[(site, rate)], key=lambda c: c["snr_db"])
                # Only channel-limited cases: a site whose success is above 0.95 at 30 dB has its highest in-range case at a
                # sub-plateau SNR, where the failures are noise-limited and independent by construction (review 2026-10-20).
                if c["snr_db"] >= 10:
                    rows.append(c)
    labels = [f"{c['site'].capitalize()} {c['chip_rate']}" for c in rows]
    iid = [c["aoi"]["iid"]["maf_gain_over_rr_pct"] for c in rows]
    chain = [c["aoi"]["gilbert"]["maf_gain_over_rr_pct"] for c in rows]
    direct = [c["aoi"]["measured_tiled"]["maf_gain_over_rr_pct"] for c in rows]
    runs = [c["burst"]["mean_fail_run"] for c in rows]
    x = np.arange(len(rows))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.1, 4.0), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.35))
    w = 0.27
    ax.bar(x - w, iid, w, color="#9e9e9e", label="i.i.d. erasures at the measured PSR")
    ax.bar(x, chain, w, color="#1b5e9e", label="two-state chain fitted to the replayed outcomes")
    ax.bar(x + w, direct, w, color="#d9772b", label="replayed outcome sequences")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Max-age-first gain over round-robin (%)")
    ax.set_ylim(min(-170, min(direct) - 5), 40)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    ax.set_title("(a)", loc="left", fontsize=8)
    ax2.bar(x, runs, 0.5, color="#555555")
    ax2.axhline(2.0, color="#d9772b", lw=0.8, ls="--")
    ax2.set_ylabel("Failure run\n(slots)")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=60, ha="right")
    ax2.set_title("(b)", loc="left", fontsize=8)
    ax2.grid(True, axis="y", lw=0.3, alpha=0.5)
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(h_pad=0.4)
    save(fig, "fig4_aoi_inversion", dict(labels=labels, iid=iid, chain=chain, direct=direct, mean_fail_run=runs,
                                          psr=[c["psr"] for c in rows], snr_db=[c["snr_db"] for c in rows],
                                          n_slots=[c["burst"]["n_slots"] for c in rows]))


def figure_ladder() -> None:
    """Goodput ceilings per site (FH-BFSK, and the coherent frame when its summary exists) against the break-even
    rate R* of the JSAC economy and the per-device Shannon rate its link budget assumes."""
    g = json.load(open(os.path.join(RES, "gate_offloading.json"), encoding="utf-8"))
    fh = json.load(open(os.path.join(RES, "summary.json"), encoding="utf-8"))["psr"]
    coh_path = os.path.join(RES, "coherent", "summary_coherent.json")
    coh = json.load(open(coh_path, encoding="utf-8"))["ladder"] if os.path.exists(coh_path) else {}
    # ARQ goodput (rate times pooled packet success at 30 dB, best scheme evaluated on every file of the site) for
    # both frame families, from the coherent summary, which also pools the FH-BFSK long replays; before that summary
    # exists, the gate's FH ceilings stand in.
    sites = [s for s in SITE_ORDER if s in coh or any(k.startswith(s + "_r") for k in fh)]
    fh_val = {}
    for s in sites:
        if s in coh and coh[s].get("fh_arq_bps") is not None:
            fh_val[s] = coh[s]["fh_arq_bps"]
        else:
            rows = [v for k, v in fh.items() if k.startswith(s + "_r") and v["measured_psr_30db"] >= 0.9]
            fh_val[s] = max((v["bits_per_s"] for v in rows), default=0.0)
    y = np.arange(len(sites))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    h = 0.36 if coh else 0.6
    floor = 10.0  # log axis: a zero value is drawn as a stub at the floor
    ax.barh(y + (h / 2 if coh else 0), [max(fh_val[s], floor) for s in sites], h, color="#1b5e9e", label="FH-BFSK frame (JANUS-like)")
    if coh:
        ax.barh(y - h / 2, [max(coh.get(s, {}).get("coherent_arq_bps", 0.0), floor) for s in sites], h, color="#d9772b",
                label="coherent single-carrier frame, DFE")
    for i, s in enumerate(sites):
        if fh_val[s] <= 0:
            ax.text(floor * 1.15, y[i] + (h / 2 if coh else 0), "none", va="center", ha="left", fontsize=6.5, color="#1b5e9e")
        if coh and coh.get(s, {}).get("coherent_arq_bps", 0.0) <= 0:
            ax.text(floor * 1.15, y[i] - h / 2, "none", va="center", ha="left", fontsize=6.5, color="#d9772b")
    fh_ceil = fh_val
    rs = g["breakeven_rate_bps"]
    ax.axvline(rs["median"], color="k", lw=0.9, ls="-", label="break-even rate R* (median over layouts)")
    ax.axvspan(rs["min"], rs["median"], color="k", alpha=0.08, lw=0)
    ax.axvline(g["parametric_rate_bps_example"]["fullband"], color="#8b2e8b", lw=0.9, ls="--", label="link budget of the published model, 10-100 m")
    ax.set_xscale("log")
    ax.set_xlim(floor, 5e5)
    ax.set_yticks(y)
    ax.set_yticklabels([s.capitalize() for s in sites])
    ax.invert_yaxis()
    ax.set_xlabel("Goodput with retransmissions at 30 dB nominal SNR (bit/s)")
    ax.grid(True, axis="x", lw=0.3, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    save(fig, "fig2_goodput_ladder", dict(sites=sites, fh_arq_bps=fh_ceil,
                                           coherent_arq_bps={s: coh.get(s, {}).get("coherent_arq_bps") for s in sites},
                                           coherent_ceiling_bps={s: coh.get(s, {}).get("coherent_ceiling_bps") for s in sites},
                                           breakeven=rs, parametric_rate_bps=g["parametric_rate_bps_example"]))


def figure_offloading() -> None:
    """Profit of the five offloading schemes of the JSAC economy (full-band reading) under the link models."""
    g = json.load(open(os.path.join(RES, "gate_offloading.json"), encoding="utf-8"))
    arms = g["arms"]["fullband"]
    # Pooled arms (every record of a site): Red's plateau success at 320 chips/s, then each frame family's goodput with
    # retransmissions on Black and Blue. The gate's record-1 arms stay in the JSON but are not drawn.
    order = [("parametric", "published link model"), ("red_pool320_psr", "Red plateau success, 320 chips/s"),
             ("black_fhpool", "Black, robust frame"), ("blue_fhpool", "Blue, robust frame"),
             ("black_coh", "Black, coherent frame"), ("blue_coh", "Blue, coherent frame")]
    order = [(k, lab) for k, lab in order if k in arms]
    schemes = [("non_offloading", "no offloading"), ("partial", "partial"), ("random", "random"), ("full", "full offloading"),
               ("optimum", "optimum")]
    cols = ["#9e9e9e", "#2e8b57", "#8b2e8b", "#d9772b", "#1b5e9e"]
    x = np.arange(len(order))
    w = 0.16
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    cache = {}
    for j, (sk, sl) in enumerate(schemes):
        vals = [arms[k][sk]["mean"] for k, _ in order]
        cache[sk] = vals
        ax.bar(x + (j - 2) * w, vals, w, color=cols[j], label=sl)
        for xi, v in zip(x + (j - 2) * w, vals):
            if abs(v) < 1e-9:
                ax.text(xi, 8, "0", ha="center", va="bottom", fontsize=5.5, color=cols[j])
    ax.set_yscale("symlog", linthresh=100)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in order], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("Profit per layout (model units)")
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    save(fig, "fig3_offloading_profit", dict(link_models=[k for k, _ in order], schemes=cache,
                                              optimum_n_off=[arms[k]["optimum_n_off"]["mean"] for k, _ in order]))


if __name__ == "__main__":
    figure_psr_sites()
    figure_ladder()
    figure_aoi()
    figure_offloading()
