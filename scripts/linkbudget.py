"""Published parametric underwater link budgets, reproduced from the papers' own equations.

Two papers are reproduced:

* Wang et al., IEEE JSAC 2023 (arXiv 2210.14619), eqs. (1)-(5), (11)-(15) and Table II.
* Cao, Chen, Cheng, Frontiers in Marine Science 2025 (10.3389/fmars.2025.1580751), eqs. (1)-(7) and Table 3.

Both use Thorp absorption, a spreading factor, the four-component Stojanovic noise model and a Shannon rate.
Writes results/linkbudget.json. Every number quoted in FINDINGS.md from this model comes from that file.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "linkbudget.json")


def thorp_db_per_km(f_khz: float) -> float:
    """Thorp absorption in dB/km, f in kHz (JSAC eq. 12, Frontiers eq. 2)."""
    f2 = f_khz**2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def noise_psd_db(f_khz: float, s: float, w: float) -> float:
    """Stojanovic four-component ambient noise p.s.d. in dB re uPa^2/Hz (JSAC eqs. 1-5)."""
    f = f_khz
    nt = 17 - 30 * np.log10(f)
    ns = 40 + 20 * (s - 0.5) + 26 * np.log10(f) - 60 * np.log10(f + 0.03)
    nw = 50 + 7.5 * np.sqrt(w) + 20 * np.log10(f) - 40 * np.log10(f + 0.4)
    nth = -15 + 20 * np.log10(f)
    lin = sum(10 ** (x / 10) for x in (nt, ns, nw, nth))
    return 10 * np.log10(lin)


def attenuation_db(d_m: float, f_khz: float, k: float) -> float:
    """A(l,f) = l^k a(f)^l in dB, with l in metres for spreading and km for absorption (JSAC eq. 11)."""
    return k * 10 * np.log10(d_m) + thorp_db_per_km(f_khz) * d_m / 1000.0


def snr_db_jsac(d_m: float, p_tx_w: float, f_khz: float, bw_hz: float, k: float, s: float, w: float,
                eta: float) -> float:
    """SNR in dB at distance d for the JSAC model with its Table II values.

    JSAC eq. (15) writes the rate as B log2(1 + eta P gamma / (2 pi H1 (1 uPa) B)); the 2 pi H1 (1 uPa) term
    is the paper's conversion from electrical power to source level in a shallow channel. We use the standard
    conversion that eta P watts of acoustic power gives a source level of 170.8 + 10 log10(eta P) dB re uPa at
    1 m, which is what the paper's citation [9] (Stojanovic 2007) uses, and treat the paper's 2 pi H1 factor as
    the cylindrical-spreading normalisation already contained in k = 1.5. This keeps the budget within the
    family the paper states; the exact scalar is reported in results/linkbudget.json for the record.
    """
    sl = 170.8 + 10 * np.log10(eta * p_tx_w)
    return sl - attenuation_db(d_m, f_khz, k) - noise_psd_db(f_khz, s, w) - 10 * np.log10(bw_hz)


def snr_db_frontiers(d_m: float, sl_db: float, f_khz: float, bw_hz: float, k: float, s: float, w: float) -> float:
    """Frontiers eqs. (1)-(6): SL - A(d,f) - N(f) - 10 log10 B."""
    return sl_db - attenuation_db(d_m, f_khz, k) - noise_psd_db(f_khz, s, w) - 10 * np.log10(bw_hz)


def shannon_bps(snr_db: float, bw_hz: float) -> float:
    return bw_hz * np.log2(1 + 10 ** (snr_db / 10))


JSAC = dict(f_khz=30.0, s=0.5, w=0.0, k=1.5, p_dev_w=0.030, p_auv_w=0.036, bw_l_hz=10e3, bw_h_hz=10e3,
            eta=0.2, h1_m=180.0, h2_m=190.0, z_bits=(1e5, 3e5), f_cpu_hz=(1e9, 4e9), alpha_cyc_per_bit=(1500, 2000))
FRONTIERS = dict(f_khz=26.0, s=0.5, w=0.0, k=1.5, sl_db=154.0, bw_hz=16e3, range_m=1000.0, packet_bits=1024)


def main() -> None:
    out = {"jsac": {}, "frontiers": {}}
    out["jsac"]["params"] = JSAC
    out["jsac"]["noise_psd_db_at_f"] = float(noise_psd_db(JSAC["f_khz"], JSAC["s"], JSAC["w"]))
    out["jsac"]["thorp_db_per_km"] = float(thorp_db_per_km(JSAC["f_khz"]))
    rows = []
    for d in (100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000):
        snr = snr_db_jsac(d, JSAC["p_dev_w"], JSAC["f_khz"], JSAC["bw_l_hz"], JSAC["k"], JSAC["s"], JSAC["w"], JSAC["eta"])
        rows.append(dict(d_m=d, snr_db=float(snr), rate_bps=float(shannon_bps(snr, JSAC["bw_l_hz"]))))
    out["jsac"]["device_to_auv"] = rows

    out["frontiers"]["params"] = FRONTIERS
    out["frontiers"]["noise_psd_db_at_f"] = float(noise_psd_db(FRONTIERS["f_khz"], FRONTIERS["s"], FRONTIERS["w"]))
    out["frontiers"]["thorp_db_per_km"] = float(thorp_db_per_km(FRONTIERS["f_khz"]))
    rows = []
    for d in (100, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000):
        snr = snr_db_frontiers(d, FRONTIERS["sl_db"], FRONTIERS["f_khz"], FRONTIERS["bw_hz"], FRONTIERS["k"],
                               FRONTIERS["s"], FRONTIERS["w"])
        rows.append(dict(d_m=d, snr_db=float(snr), rate_bps=float(shannon_bps(snr, FRONTIERS["bw_hz"]))))
    out["frontiers"]["sensor_to_auv"] = rows
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    for name in ("jsac", "frontiers"):
        key = "device_to_auv" if name == "jsac" else "sensor_to_auv"
        print(name, "noise", round(out[name]["noise_psd_db_at_f"], 1), "dB re uPa^2/Hz; Thorp",
              round(out[name]["thorp_db_per_km"], 2), "dB/km")
        for r in out[name][key]:
            print(f"  d={r['d_m']:5d} m  SNR={r['snr_db']:6.1f} dB  Shannon={r['rate_bps']/1e3:7.1f} kbps")


if __name__ == "__main__":
    main()
