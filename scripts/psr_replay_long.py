"""Long replay sweeps for PLAN items 2 and 3: the same frame, SNR grid and protocol as psr_replay.py, over every
segment of one library channel file and a chosen element range, so the AoI arm gets consecutive slots.

Usage: python -u src/psr_replay_long.py <site> <file_key> <chip_rate> [--elements a-b] [--smoke]
  site      blue, red, black, green, yellow, purple, pink
  file_key  channel file without .mat (blue_7, green, yellow_3, purple_12, pink_1)
  --elements a-b  inclusive element range (default 0-0; Green uses its 64 time-diversity slots as elements)

Differences from the gate sweep (psr_replay.py): no 45-segment cap, the element range is a parameter, the carrier
is read from the channel file for the sites the gate did not cover (Blue, Red, Black and Green keep the gate's
carriers), and the noise file follows the library's pairing table (uwa-channels.github.io/noises/). Green's noise
model has one channel, so noise is generated from element 0 for every slot (FINDINGS F1).

Writes results/long/psr_<site>_<file_key>[_e<a>-<b>]_r<rate>.json and caches noiseless replays in
results/long/replay_cache/. Existing gate results in results/ are never touched.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modem import BAND_HZ, PAYLOAD_BITS, Rate, demodulate, frame_chips, modulate  # noqa: E402
from psr_replay import DATA, FS, N_NOISE_SEEDS, SNR_GRID, TAIL_S, band_power  # noqa: E402
from uwa_channels import load_channel, load_noise, noisegen, replay  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "long")
CACHE = os.path.join(OUT_DIR, "replay_cache")

# Carriers the gate used (psr_replay.CHANNELS); other sites take params.fc from the file.
GATE_FC = {"blue": 13000.0, "red": 25000.0, "black": 18000.0, "green": 6000.0}
SITES = {
    "blue": "North Atlantic 2010, mobile, 3-7 km, 13 kHz",
    "red": "Singapore 2024, drifting, 0.1-0.4 km, harbour, 25 kHz",
    "black": "Mariana Trench 2024, vertical 8.72 km, 18 kHz",
    "green": "Norway 2024, moored 0.27 km, 64 probes 10 min apart, 6 kHz",
    "yellow": "Hawaii 2011 (KAM11), moored, 3 and 7 km, 13 kHz",
    "purple": "North Atlantic 2008, moored, shallow 15 m, 0.06 to 1 km, 12.5 kHz",
    "pink": "Japan 2022, moored, 14 km, 6 kHz",
}


def noise_file(site: str, file_key: str) -> str:
    """Library pairing table: Blue, Red, Black, Green one noise file each; Yellow 1-3 share noise 1, 4, 5, 6 take
    noise 2, 3, 4; Purple k takes noise ((k-1) mod 5)+1 (the same five files for each of the three arrays);
    Pink k takes noise k."""
    if site in ("blue", "red", "black", "green"):
        return f"{site}_noise.mat"
    k = int(file_key.split("_")[1])
    if site == "yellow":
        return "yellow_noise_1.mat" if k <= 3 else f"yellow_noise_{k - 2}.mat"
    if site == "purple":
        return f"purple_noise_{(k - 1) % 5 + 1}.mat"
    if site == "pink":
        return f"pink_noise_{k}.mat"
    raise ValueError(site)


def scalar(group, name: str) -> float:
    return float(np.array(group[name]).squeeze())


def main(site: str, file_key: str, chip_rate: float, elements: list[int], smoke: bool = False) -> None:
    rate = Rate(chip_rate)
    rng = np.random.default_rng(20261018)
    nz_name = noise_file(site, file_key)
    ch = load_channel(os.path.join(DATA, f"{file_key}.mat"))
    nz = load_noise(os.path.join(DATA, nz_name))
    fs_delay = scalar(ch["params"], "fs_delay")
    fc_file = scalar(ch["params"], "fc")
    fc = GATE_FC.get(site, fc_file)
    fc_noise = scalar(nz, "fc") if "fc" in nz else float("nan")
    key = "phi_hat" if "phi_hat" in ch else "theta_hat"
    n_rec = ch[key].shape[0]
    m_ch = ch["h_hat"].shape[1]
    beta = np.array(nz["beta"])
    n_noise_ch = int(beta.shape[1]) if beta.ndim == 3 else 1
    noise_el = (lambda el: el if el < n_noise_ch else 0)
    if max(elements) >= m_ch:
        raise SystemExit(f"{file_key}: element range {elements[0]}-{elements[-1]} but the file has {m_ch} elements")

    payload = rng.integers(0, 2, PAYLOAD_BITS)
    x = modulate(frame_chips(payload), fc, FS, rate)
    x = np.concatenate([x, np.zeros(int(TAIL_S * FS))])
    dur = x.size / FS
    seg_len = int(np.ceil(dur * fs_delay))
    # replay() needs start + T + L <= record length, where L is the tap count of h_hat; a start that leaves
    # less than that raises a broadcast error. Segments stay back to back at seg_len spacing.
    n_taps = int(ch["h_hat"].shape[2])
    starts = list(range(0, n_rec - (seg_len + n_taps), seg_len))
    if smoke:
        starts = starts[:2]
        elements = elements[:1]
    el_tag = "" if elements == [0] else f"_e{elements[0]:02d}-{elements[-1]:02d}"
    tag = f"{site}_{file_key}{el_tag}_r{int(chip_rate)}"
    os.makedirs(CACHE, exist_ok=True)
    print(tag, "file", file_key, "noise", nz_name, "fc used", fc, "fc file", fc_file, "fc noise", fc_noise,
          "elements in file", m_ch, "noise channels", n_noise_ch, flush=True)
    print(tag, "record", round(n_rec / fs_delay, 1), "s; frame+tail", round(dur, 3), "s; segments", len(starts),
          "x elements", len(elements), "x seeds", N_NOISE_SEEDS, "x SNR", len(SNR_GRID), flush=True)
    if abs(fc_noise - fc_file) > 1.0 and not np.isnan(fc_noise):
        print("WARNING noise fc differs from channel fc", fc_noise, fc_file, flush=True)

    outs = {}
    t0 = time.time()
    for el in elements:
        for st in starts:
            fn = os.path.join(CACHE, f"{file_key}_r{int(chip_rate)}_e{el}_s{st}.npy")
            if os.path.exists(fn):
                y = np.load(fn)
            else:
                y = replay(x, FS, [el], ch, start=st)[:, 0].astype(np.float32)
                np.save(fn, y)
            outs[(el, st)] = y
        print("element", el, "replayed", round(time.time() - t0, 1), "s", flush=True)
    print("replays done", round(time.time() - t0, 1), "s", flush=True)

    p_rx = {k: band_power(v.astype(np.float64), FS, fc) for k, v in outs.items()}
    p_rx_mean = float(np.mean(list(p_rx.values())))
    w_long = noisegen((int(10 * FS), 1), FS, [noise_el(elements[0])], nz)[:, 0]
    p_n1 = band_power(w_long, FS, fc)
    p_n1_total = float(np.mean(w_long**2))
    print("P_rx mean", p_rx_mean, "P_n1 in-band", p_n1, "(total", p_n1_total, ")", flush=True)

    results = dict(channel=site, file=file_key, noise_file=nz_name, site=SITES[site], chip_rate=chip_rate,
                   n_slots=rate.n_slots, bits_per_s=rate.bits_per_s, fc=fc, fc_file=fc_file, fc_noise=fc_noise, fs=FS,
                   fs_delay=fs_delay, record_s=n_rec / fs_delay, frame_s=rate.frame_duration_s(), slot_s=dur,
                   n_segments=len(starts), segment_starts=starts, elements=elements, n_elements_in_file=m_ch,
                   p_rx_mean=p_rx_mean, p_n1_inband=p_n1, p_n1_total=p_n1_total,
                   p_rx_segments={f"{k[0]}:{k[1]}": v for k, v in p_rx.items()}, snr_grid=SNR_GRID,
                   n_noise_seeds=N_NOISE_SEEDS, protocol="psr_replay.py gate protocol, all segments, no cap",
                   measured={}, awgn={})
    t0 = time.time()
    for snr_db in SNR_GRID:
        sigma = np.sqrt(p_rx_mean / (p_n1 * 10 ** (snr_db / 10)))
        succ = []
        for el in elements:
            for st in starts:
                y = outs[(el, st)].astype(np.float64)
                for sd in range(N_NOISE_SEEDS):
                    w = noisegen((y.size, 1), FS, [noise_el(el)], nz)[:, 0]
                    bits, off, met = demodulate(y + sigma * w, fc, FS, rate=rate)
                    succ.append([el, st, sd, int(np.array_equal(bits, payload)), off])
        arr = np.array(succ)
        results["measured"][str(snr_db)] = dict(psr=float(arr[:, 3].mean()), n=int(arr.shape[0]), trials=arr.tolist())
        n_tr = arr.shape[0]
        ok = 0
        for _ in range(n_tr):
            delay = rng.integers(0, int(0.1 * FS))
            xa = np.concatenate([np.zeros(delay), x])
            noise_psd = 1.0 / (10 ** (snr_db / 10) * BAND_HZ)
            w = rng.standard_normal(xa.size) * np.sqrt(noise_psd * FS / 2)
            bits, _, _ = demodulate(xa + w, fc, FS, rate=rate)
            ok += int(np.array_equal(bits, payload))
        results["awgn"][str(snr_db)] = dict(psr=ok / n_tr, n=int(n_tr))
        print(f"SNR {snr_db:+6.1f} dB  measured PSR {results['measured'][str(snr_db)]['psr']:.3f}  "
              f"AWGN PSR {results['awgn'][str(snr_db)]['psr']:.3f}   ({round(time.time() - t0)} s)", flush=True)
    out = os.path.join(OUT_DIR, f"psr_{tag}{'_smoke' if smoke else ''}.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh)
    os.replace(tmp, out)
    print("wrote", out, flush=True)


def parse_elements(argv: list[str]) -> list[int]:
    if "--elements" in argv:
        a, b = argv[argv.index("--elements") + 1].split("-")
        return list(range(int(a), int(b) + 1))
    return [0]


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), parse_elements(sys.argv), smoke="--smoke" in sys.argv)
