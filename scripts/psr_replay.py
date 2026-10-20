"""Packet success rate of the JANUS-like frame on replayed measured channels and on AWGN at matched SNR.

Usage: python src/psr_replay.py <channel> <chip_rate> [--smoke]
Channels: blue, red, black, green (see CHANNELS). Chip rates: 160 (JANUS), 320, 640, 1920 chips/s in the same
4160 Hz band. Writes results/psr_<channel>_r<rate>.json and caches noiseless replayed frames in
results/replay_cache/.

SNR definition. The library normalises the stored channel so a unit-power input spanning the record gives unit
output power per element on average, and normalises the noise to unit (pseudo-)power per element. Both numbers
depend on the signal band, so we measure them: P_rx is the mean power of the noiseless replayed frame over all
segments of the record, and P_n1 is the mean power of unit-scale library noise inside the 4160 Hz frame band.
Nominal SNR = P_rx / (sigma_n^2 P_n1); sigma_n is set from the target SNR. Per-segment received power then
varies as the channel does, which is the fading a modem sees at fixed transmit power. The AWGN arm uses white
Gaussian noise at the same in-band SNR with no channel, which is what a parametric path-loss model assumes.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modem import BAND_HZ, PAYLOAD_BITS, Rate, demodulate, frame_chips, modulate  # noqa: E402
from uwa_channels import load_channel, load_noise, noisegen, replay  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("UWA_LIBRARY_ROOT", "N:/Datasets/UWChannelLibrary")  # the library's .mat files (Zenodo 10.5281/zenodo.21287414)
CACHE = os.path.join(ROOT, "results", "replay_cache")
FS = 96000.0
TAIL_S = 0.35  # zeros after the frame so the channel's delay spread fits
SNR_GRID = [-15.0, -12.5, -10.0, -7.5, -5.0, -2.5, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0]
N_NOISE_SEEDS = 2
MAX_SEGMENTS = 45

CHANNELS = {
    "blue": dict(file="blue_1.mat", noise="blue_noise.mat", fc=13000.0, elements=[0], site="North Atlantic 2010, mobile, 3-7 km"),
    "red": dict(file="red_1.mat", noise="red_noise.mat", fc=25000.0, elements=[0], site="Singapore 2024, drifting, 0.1-0.4 km, harbour"),
    "black": dict(file="black.mat", noise="black_noise.mat", fc=18000.0, elements=[0], site="Mariana Trench 2024, vertical 8.72 km"),
    "green": dict(file="green.mat", noise="green_noise.mat", fc=6000.0, elements=[0, 16, 32, 48],
                  site="Norway 2024, moored 0.27 km, 64 probes 10 min apart"),
}


def band_power(x: np.ndarray, fs: float, fc: float) -> float:
    """Mean power of x inside [fc - BAND/2, fc + BAND/2]."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1 / fs)
    m = (f >= fc - BAND_HZ / 2) & (f <= fc + BAND_HZ / 2)
    return float(2 * np.sum(np.abs(X[m]) ** 2) / x.size**2)


def main(name: str, chip_rate: float, smoke: bool = False) -> None:
    cfg = CHANNELS[name]
    rate = Rate(chip_rate)
    rng = np.random.default_rng(20261018)
    ch = load_channel(os.path.join(DATA, cfg["file"]))
    nz = load_noise(os.path.join(DATA, cfg["noise"]))
    fs_delay = float(np.array(ch["params"]["fs_delay"]).squeeze())
    key = "phi_hat" if "phi_hat" in ch else "theta_hat"
    n_rec = ch[key].shape[0]
    fc = cfg["fc"]
    payload = rng.integers(0, 2, PAYLOAD_BITS)
    x = modulate(frame_chips(payload), fc, FS, rate)
    x = np.concatenate([x, np.zeros(int(TAIL_S * FS))])
    dur = x.size / FS
    seg_len = int(np.ceil(dur * fs_delay))
    starts = list(range(0, n_rec - seg_len, seg_len))[:MAX_SEGMENTS]
    if smoke:
        starts = starts[:2]
        cfg = dict(cfg, elements=cfg["elements"][:1])
    os.makedirs(CACHE, exist_ok=True)
    tag = f"{name}_r{int(chip_rate)}"
    print(tag, "record", round(n_rec / fs_delay, 1), "s; frame+tail", round(dur, 3), "s; segments", len(starts),
          "x elements", len(cfg["elements"]), flush=True)

    outs = {}
    t0 = time.time()
    for el in cfg["elements"]:
        for st in starts:
            fn = os.path.join(CACHE, f"{tag}_e{el}_s{st}.npy")
            if os.path.exists(fn):
                y = np.load(fn)
            else:
                y = replay(x, FS, [el], ch, start=st)[:, 0].astype(np.float32)
                np.save(fn, y)
            outs[(el, st)] = y
    print("replays done", round(time.time() - t0, 1), "s", flush=True)

    p_rx = {k: band_power(v.astype(np.float64), FS, fc) for k, v in outs.items()}
    p_rx_mean = float(np.mean(list(p_rx.values())))
    n_noise_ch = int(np.array(nz["beta"]).shape[1]) if np.array(nz["beta"]).ndim == 3 else 1
    noise_el = (lambda el: el if el < n_noise_ch else 0)  # Green: 64 time-diversity slots, one hydrophone noise model
    w_long = noisegen((int(10 * FS), 1), FS, [noise_el(cfg["elements"][0])], nz)[:, 0]
    p_n1 = band_power(w_long, FS, fc)
    p_n1_total = float(np.mean(w_long**2))
    print("P_rx mean", p_rx_mean, "P_n1 in-band", p_n1, "(total", p_n1_total, ")", flush=True)

    results = dict(channel=name, site=cfg["site"], chip_rate=chip_rate, n_slots=rate.n_slots, bits_per_s=rate.bits_per_s,
                   fc=fc, fs=FS, frame_s=rate.frame_duration_s(), slot_s=dur, n_segments=len(starts),
                   elements=cfg["elements"], p_rx_mean=p_rx_mean, p_n1_inband=p_n1, p_n1_total=p_n1_total,
                   p_rx_segments={f"{k[0]}:{k[1]}": v for k, v in p_rx.items()}, snr_grid=SNR_GRID,
                   n_noise_seeds=N_NOISE_SEEDS, measured={}, awgn={})
    t0 = time.time()
    for snr_db in SNR_GRID:
        sigma = np.sqrt(p_rx_mean / (p_n1 * 10 ** (snr_db / 10)))
        succ = []
        for el in cfg["elements"]:
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
    out = os.path.join(ROOT, "results", f"psr_{tag}{'_smoke' if smoke else ''}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), smoke="--smoke" in sys.argv)
