"""Packet success of the coherent single-carrier frame (coherent.py) on replayed measured channels and on AWGN.

Usage: python -u src/psr_coherent.py <site> <file_key> [--elements a-b] [--schemes 1000b,2000b,4000b,2000q,4000q] [--smoke]

Same protocol as psr_replay_long.py (segments back to back over the record, the library's noise pairing, the
nominal SNR measured in the frame's own band, two noise seeds, an AWGN arm at matched in-band SNR), with the
FH-BFSK frame replaced by the coherent frame at each (symbol rate, alphabet) of the ladder. The feedback span of
the equaliser is set per channel file from the 10% to 99% energy window of the stored impulse response at the
symbol rate, capped at coherent.MAX_FB; the cap and the window are recorded so a truncated channel is visible.

Writes results/coherent/psr_coh_<site>_<file_key>[_e<a>-<b>]_<rate>_<mod>.json and caches noiseless replays
in results/coherent/replay_cache/. Existing results are never touched.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coherent import MAX_FB, PAYLOAD_BITS, Scheme, demodulate, frame_symbols, modulate  # noqa: E402
from psr_replay import DATA, FS, TAIL_S  # noqa: E402
from psr_replay_long import GATE_FC, SITES, noise_file, scalar  # noqa: E402
from uwa_channels import load_channel, load_noise, noisegen, replay  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "coherent")
CACHE = os.path.join(OUT_DIR, "replay_cache")
SNR_GRID = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0]
N_NOISE_SEEDS = 2
MAX_SEGMENTS = 40
DEFAULT_SCHEMES = "1000b,2000b,4000b,1000q,2000q,4000q"


def parse_schemes(s: str) -> list[Scheme]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        out.append(Scheme(float(tok[:-1]), {"b": "bpsk", "q": "qpsk"}[tok[-1]]))
    return out


def band_power(x: np.ndarray, fs: float, fc: float, bw: float) -> float:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1 / fs)
    m = (f >= fc - bw / 2) & (f <= fc + bw / 2)
    return float(2 * np.sum(np.abs(X[m]) ** 2) / x.size**2)


def delay_window_ms(ch, element: int) -> tuple[float, float, float]:
    """(10%, 99%) energy points of the mean power-delay profile of one element, in ms, and the stored window."""
    h = ch["h_hat"]
    hr, hi = np.asarray(h["real"])[:, element, :], np.asarray(h["imag"])[:, element, :]
    fsd = scalar(ch["params"], "fs_delay")
    pdp = (hr**2 + hi**2).mean(0)
    c = np.cumsum(pdp) / pdp.sum()
    t10, t99 = [float(np.searchsorted(c, q) / fsd * 1e3) for q in (0.10, 0.99)]
    return t10, t99, float(hr.shape[1] / fsd * 1e3)


def main(site: str, file_key: str, elements: list[int], schemes: list[Scheme], smoke: bool = False) -> None:
    rng = np.random.default_rng(20261019)
    nz_name = noise_file(site, file_key)
    ch = load_channel(os.path.join(DATA, f"{file_key}.mat"))
    nz = load_noise(os.path.join(DATA, nz_name))
    fs_delay = scalar(ch["params"], "fs_delay")
    fc_file = scalar(ch["params"], "fc")
    fc = GATE_FC.get(site, fc_file)
    key = "phi_hat" if "phi_hat" in ch else "theta_hat"
    n_rec = ch[key].shape[0]
    m_ch = ch["h_hat"].shape[1]
    n_taps = int(ch["h_hat"].shape[2])
    beta = np.array(nz["beta"])
    n_noise_ch = int(beta.shape[1]) if beta.ndim == 3 else 1
    noise_el = (lambda el: el if el < n_noise_ch else 0)
    if max(elements) >= m_ch:
        raise SystemExit(f"{file_key}: element range {elements[0]}-{elements[-1]} but the file has {m_ch} elements")
    t10, t99, stored = delay_window_ms(ch, elements[0])
    os.makedirs(CACHE, exist_ok=True)
    el_tag = "" if elements == [0] else f"_e{elements[0]:02d}-{elements[-1]:02d}"
    print(f"{site} {file_key} fc {fc} noise {nz_name} elements {elements} delay window 10-99% {t10:.1f}-{t99:.1f} ms "
          f"(stored {stored:.0f} ms)", flush=True)

    for scheme in schemes:
        tag = f"{site}_{file_key}{el_tag}_{int(scheme.symbol_rate)}_{scheme.mod}"
        out_path = os.path.join(OUT_DIR, f"psr_coh_{tag}{'_smoke' if smoke else ''}.json")
        if os.path.exists(out_path):
            print(tag, "exists, skipping", flush=True)
            continue
        fb_needed = int(np.ceil((t99 - t10) / 1e3 * scheme.symbol_rate)) + 8
        fb_symbols = int(min(fb_needed, MAX_FB))
        payload = rng.integers(0, 2, PAYLOAD_BITS)
        x = modulate(frame_symbols(payload, scheme), fc, FS, scheme)
        x = np.concatenate([x, np.zeros(int(TAIL_S * FS))])
        dur = x.size / FS
        seg_len = int(np.ceil(dur * fs_delay))
        starts = list(range(0, n_rec - (seg_len + n_taps), seg_len))[:MAX_SEGMENTS]
        if smoke:
            starts = starts[:2]
            elements = elements[:1]
        print(tag, "frame", round(scheme.frame_s, 3), "s; payload bit/s", round(scheme.bits_per_s), "; band", scheme.band_hz,
              "Hz; fb taps", fb_symbols, "(needed", fb_needed, "); segments", len(starts), "x elements", len(elements), flush=True)
        t0 = time.time()
        outs = {}
        for el in elements:
            for st in starts:
                fn = os.path.join(CACHE, f"{tag}_e{el}_s{st}.npy")
                if os.path.exists(fn):
                    y = np.load(fn)
                else:
                    y = replay(x, FS, [el], ch, start=st)[:, 0].astype(np.float32)
                    np.save(fn, y)
                outs[(el, st)] = y
        print(tag, "replays done", round(time.time() - t0, 1), "s", flush=True)
        p_rx = {k: band_power(v.astype(np.float64), FS, fc, scheme.band_hz) for k, v in outs.items()}
        p_rx_mean = float(np.mean(list(p_rx.values())))
        w_long = noisegen((int(10 * FS), 1), FS, [noise_el(elements[0])], nz)[:, 0]
        p_n1 = band_power(w_long, FS, fc, scheme.band_hz)
        results = dict(site=site, description=SITES.get(site, ""), file=file_key, noise_file=nz_name, fc=fc, fc_file=fc_file, fs=FS,
                       scheme=dict(symbol_rate=scheme.symbol_rate, mod=scheme.mod, bits_per_s=scheme.bits_per_s, frame_s=scheme.frame_s,
                                   band_hz=scheme.band_hz, payload_bits=PAYLOAD_BITS, n_symbols=scheme.n_symbols),
                       fb_symbols=fb_symbols, fb_needed=fb_needed, fb_truncated=bool(fb_needed > MAX_FB),
                       delay_window_ms=dict(t10=t10, t99=t99, stored=stored), slot_s=dur, n_segments=len(starts), elements=elements,
                       segment_starts=starts, p_rx_mean=p_rx_mean, p_n1_inband=p_n1,
                       p_rx_segments={f"{k[0]}:{k[1]}": v for k, v in p_rx.items()}, snr_grid=SNR_GRID, n_noise_seeds=N_NOISE_SEEDS,
                       measured={}, awgn={})
        t0 = time.time()
        for snr_db in SNR_GRID:
            sigma = np.sqrt(p_rx_mean / (p_n1 * 10 ** (snr_db / 10)))
            trials, out_snrs = [], []
            for el in elements:
                for st in starts:
                    y = outs[(el, st)].astype(np.float64)
                    for sd in range(N_NOISE_SEEDS):
                        w = noisegen((y.size, 1), FS, [noise_el(el)], nz)[:, 0]
                        bits, off, met, snr_out, err, f_off = demodulate(y + sigma * w, fc, FS, scheme, fb_symbols)
                        trials.append([el, st, sd, int(np.array_equal(bits, payload)), off, round(snr_out, 2), round(f_off, 2)])
                        out_snrs.append(snr_out)
            arr = np.array(trials)
            results["measured"][str(snr_db)] = dict(psr=float(arr[:, 3].mean()), n=int(arr.shape[0]),
                                                    out_snr_db=dict(mean=float(np.mean(out_snrs)), median=float(np.median(out_snrs)),
                                                                    p10=float(np.percentile(out_snrs, 10))),
                                                    trials=arr.tolist())
            n_tr = arr.shape[0]
            ok, out_a = 0, []
            for _ in range(n_tr):
                delay = rng.integers(0, int(0.1 * FS))
                xa = np.concatenate([np.zeros(delay), x])
                noise_psd = 1.0 / (10 ** (snr_db / 10) * scheme.band_hz)
                w = rng.standard_normal(xa.size) * np.sqrt(noise_psd * FS / 2)
                bits, _, _, snr_out, _, _ = demodulate(xa + w, fc, FS, scheme, fb_symbols)
                ok += int(np.array_equal(bits, payload))
                out_a.append(snr_out)
            results["awgn"][str(snr_db)] = dict(psr=ok / n_tr, n=int(n_tr), out_snr_db=dict(mean=float(np.mean(out_a))))
            print(f"{tag} SNR {snr_db:+6.1f} dB  measured PSR {results['measured'][str(snr_db)]['psr']:.3f} out-SNR {np.mean(out_snrs):5.1f} dB  "
                  f"AWGN PSR {ok / n_tr:.3f} out-SNR {np.mean(out_a):5.1f} dB   ({round(time.time() - t0)} s)", flush=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh)
        print("wrote", out_path, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    site, file_key = args[0], args[1]
    elements = [0]
    schemes = DEFAULT_SCHEMES
    if "--elements" in args:
        a, b = args[args.index("--elements") + 1].split("-")
        elements = list(range(int(a), int(b) + 1))
    if "--schemes" in args:
        schemes = args[args.index("--schemes") + 1]
    main(site, file_key, elements, parse_schemes(schemes), smoke="--smoke" in args)
