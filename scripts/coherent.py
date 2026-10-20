"""A coherent single-carrier frame with a fractionally spaced decision-feedback equaliser (PLAN item 1).

The frame is BPSK or QPSK at a symbol rate R_s inside a (1 + alpha) R_s band (root-raised-cosine, alpha = 0.25),
so 4000 sym/s occupies 5 kHz, 2000 sym/s 2.5 kHz and 1000 sym/s 1.25 kHz, all inside every library channel's
band except that 4000 sym/s touches the edge of Pink's 4 kHz band. It carries a known 512-symbol preamble (a
fixed pseudo-random BPSK pattern) used for Doppler estimation, synchronisation and equaliser training, then 1024
payload bits plus 8 tail bits through the same rate-1/2 K=9 convolutional code as the FH-BFSK frame (modem.py),
Gray-mapped onto BPSK (2064 symbols) or QPSK (1032 symbols).

The receiver is the structure the channel library's own validation uses (its Section V, after Stojanovic 1993):
complex baseband at two samples per symbol, a carrier-offset estimate from the preamble that is removed together
with the matching wideband time-scale (a = f_off / fc, by resampling), a T/2-spaced feedforward filter and a
symbol-spaced feedback filter updated jointly by RLS, and a second-order decision-directed PLL on the equaliser
output for what remains. Training mode over the preamble, decision-directed over the payload, soft decisions into
the Viterbi decoder. The feedback span is set per channel from the measured delay window (`fb_symbols`), capped by
`MAX_FB`, so a channel whose delay spread exceeds the cap (Green, Pink) is reported as such rather than silently
truncated. The 2026-10-19 first sweep used a 256-symbol preamble, 512 payload bits and no Doppler step; it failed
on every mobile Blue file and on the long-delay Yellow and Purple files, which is why both were changed.

Nothing here is tuned to a channel beyond the feedback span; the RLS forgetting factor and the PLL gains are
fixed for every site and rate.
"""
from __future__ import annotations

import numpy as np

from modem import TAIL_BITS, conv_encode, viterbi_decode

ALPHA = 0.25
PREAMBLE_SYMBOLS = 512  # RLS needs a few times the tap count to converge; long-delay sites need over 100 feedback taps
PAYLOAD_BITS = 1024
SPS = 2  # samples per symbol at the equaliser
MAX_FB = 160  # feedback taps (symbols)
RLS_LAMBDA = 0.998
PLL_K1 = 0.02
PLL_K2 = PLL_K1 / 10

# Preamble: a fixed pseudo-random BPSK pattern (seed fixed so every run uses the same one).
_rng_pre = np.random.default_rng(1993)
PREAMBLE = 1.0 - 2.0 * _rng_pre.integers(0, 2, PREAMBLE_SYMBOLS)


class Scheme:
    """Symbol rate and alphabet. `bits_per_s` counts payload bits over the whole frame (preamble included)."""

    def __init__(self, symbol_rate: float, mod: str = "bpsk"):
        self.symbol_rate = float(symbol_rate)
        self.mod = mod
        self.bits_per_symbol = {"bpsk": 1, "qpsk": 2}[mod]
        self.coded_bits = 2 * (PAYLOAD_BITS + TAIL_BITS)
        self.data_symbols = self.coded_bits // self.bits_per_symbol
        self.n_symbols = PREAMBLE_SYMBOLS + self.data_symbols
        self.band_hz = (1 + ALPHA) * self.symbol_rate

    @property
    def frame_s(self) -> float:
        return self.n_symbols / self.symbol_rate

    @property
    def bits_per_s(self) -> float:
        return PAYLOAD_BITS / self.frame_s

    def __repr__(self):
        return f"Scheme({self.symbol_rate:.0f} sym/s, {self.mod})"


def rrc_taps(sps: int, alpha: float = ALPHA, span_symbols: int = 12) -> np.ndarray:
    """Root-raised-cosine impulse response, unit energy."""
    n = span_symbols * sps
    t = np.arange(-n, n + 1) / sps
    h = np.empty(t.size)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(4 * alpha * ti) - 1.0) < 1e-9:
            h[i] = (alpha / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))) / (
                np.pi * ti * (1 - (4 * alpha * ti) ** 2))
    return h / np.sqrt(np.sum(h**2))


def map_symbols(coded: np.ndarray, mod: str) -> np.ndarray:
    if mod == "bpsk":
        return (1.0 - 2.0 * coded).astype(complex)
    b = coded.reshape(-1, 2)
    return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)  # Gray: bit0 -> I sign, bit1 -> Q sign


def soft_bits(z: np.ndarray, mod: str) -> np.ndarray:
    """Soft decisions with the Viterbi convention: positive means coded bit 0."""
    if mod == "bpsk":
        return np.real(z)
    return np.stack([np.real(z), np.imag(z)], axis=1).reshape(-1) * np.sqrt(2)


def frame_symbols(payload: np.ndarray, scheme: Scheme) -> np.ndarray:
    bits = np.concatenate([payload.astype(np.int8), np.zeros(TAIL_BITS, dtype=np.int8)])
    coded = conv_encode(bits)
    return np.concatenate([PREAMBLE.astype(complex), map_symbols(coded, scheme.mod)])


def modulate(symbols: np.ndarray, fc: float, fs: float, scheme: Scheme) -> np.ndarray:
    """Unit-mean-square real passband signal; fs must be an integer multiple of 2 R_s."""
    sps_tx = int(round(fs / scheme.symbol_rate))
    if abs(sps_tx * scheme.symbol_rate - fs) > 1e-6:
        raise ValueError("fs must be an integer multiple of the symbol rate")
    up = np.zeros(symbols.size * sps_tx, dtype=complex)
    up[::sps_tx] = symbols
    h = rrc_taps(sps_tx)
    bb = np.convolve(up, h)
    t = np.arange(bb.size) / fs
    pb = np.real(bb * np.exp(2j * np.pi * fc * t))
    return pb / np.sqrt(np.mean(pb**2))


def to_baseband(x: np.ndarray, fc: float, fs: float, scheme: Scheme) -> np.ndarray:
    """Mix down, matched-filter with the RRC pulse, and decimate to SPS samples per symbol."""
    sps_rx = int(round(fs / scheme.symbol_rate))
    t = np.arange(x.size) / fs
    bb = x * np.exp(-2j * np.pi * fc * t)
    h = rrc_taps(sps_rx)
    y = np.convolve(bb, h)[h.size // 2: h.size // 2 + x.size]
    dec = sps_rx // SPS
    return y[::dec]


def synchronise(y: np.ndarray, max_delay_s: float, scheme: Scheme) -> tuple[int, float]:
    """Preamble correlation at SPS samples per symbol; returns (start sample, normalised peak)."""
    n_max = int(max_delay_s * scheme.symbol_rate * SPS)
    pre = PREAMBLE
    p_up = np.zeros(PREAMBLE_SYMBOLS * SPS, dtype=complex)
    p_up[::SPS] = pre
    seg = y[: n_max + p_up.size + 1]
    c = np.correlate(seg, p_up, mode="valid")
    c = c[: n_max + 1]
    # Normalise by local energy so a loud multipath tail does not win.
    e = np.convolve(np.abs(seg) ** 2, np.ones(p_up.size), mode="valid")[: n_max + 1]
    metric = np.abs(c) / np.sqrt(np.maximum(e, 1e-12) * PREAMBLE_SYMBOLS)
    k = int(np.argmax(metric))
    return k, float(metric[k])


def dfe(y: np.ndarray, start: int, scheme: Scheme, fb_symbols: int, ff_symbols: int = 12,
        lam: float = RLS_LAMBDA, k1: float = PLL_K1, k2: float = PLL_K2):
    """T/2-spaced feedforward, symbol-spaced feedback, joint RLS, second-order decision-directed PLL.

    Returns (equalised symbols for the data part, output SNR over the training part in dB, mean |e|^2 over data).
    """
    n_ff = ff_symbols * SPS
    n_fb = int(min(max(fb_symbols, 1), MAX_FB))
    n_sym = scheme.n_symbols
    pre_cursor = (ff_symbols // 3) * SPS  # taps ahead of the cursor
    # Pad so the window never runs off the end.
    y = np.concatenate([np.zeros(n_ff, dtype=complex), y, np.zeros(n_ff + n_sym * SPS, dtype=complex)])
    base = start + n_ff
    n_taps = n_ff + n_fb
    w = np.zeros(n_taps, dtype=complex)
    P = np.eye(n_taps, dtype=complex) * 100.0
    theta = 0.0
    psi_acc = 0.0
    fb_hist = np.zeros(n_fb, dtype=complex)
    out = np.empty(n_sym, dtype=complex)
    err2 = np.empty(n_sym)
    tx = PREAMBLE
    for n in range(n_sym):
        c = base + n * SPS
        u_ff = y[c - pre_cursor: c - pre_cursor + n_ff][::-1] * np.exp(-1j * theta)
        u = np.concatenate([u_ff, fb_hist])
        z = np.vdot(w, u)  # w^H u
        if n < PREAMBLE_SYMBOLS:
            d = tx[n]
        else:
            d = _decide(z, scheme.mod)
        e = d - z
        # RLS
        Pu = P @ u
        g = Pu / (lam + np.vdot(u, Pu).real)
        w = w + g * np.conj(e)
        P = (P - np.outer(g, np.conj(Pu))) / lam
        # PLL on the decision-directed phase error of the equaliser output
        psi = np.imag(z * np.conj(d))
        psi_acc += psi
        theta += k1 * psi + k2 * psi_acc
        fb_hist = np.concatenate([[d], fb_hist[:-1]])
        out[n] = z
        err2[n] = abs(e) ** 2
    train = out[3 * PREAMBLE_SYMBOLS // 4: PREAMBLE_SYMBOLS]  # last quarter of the training block
    ref = PREAMBLE[3 * PREAMBLE_SYMBOLS // 4:]
    gamma = np.mean(train * np.conj(ref))
    snr_out = 10 * np.log10(np.sum(np.abs(ref) ** 2) / max(np.sum(np.abs(ref - train / gamma) ** 2), 1e-12))
    return out[PREAMBLE_SYMBOLS:], float(snr_out), float(np.mean(err2[PREAMBLE_SYMBOLS:]))


def _decide(z: complex, mod: str) -> complex:
    if mod == "bpsk":
        return 1.0 + 0j if z.real >= 0 else -1.0 + 0j
    return (np.sign(z.real) + 1j * np.sign(z.imag)) / np.sqrt(2)


DOPPLER_COARSE_HZ = np.arange(-40.0, 40.1, 5.0)
DOPPLER_FINE_HZ = 0.25
DOPPLER_MAX_HZ = 40.0


def estimate_doppler(y: np.ndarray, fc: float, max_delay_s: float, scheme: Scheme) -> tuple[float, int, float]:
    """Carrier offset of the received preamble, as a proxy for the wideband Doppler scale a = f_off / fc.

    Coarse: synchronise after demixing each candidate offset and keep the best correlation. Fine: despread the
    preamble at the chosen start by the known symbols and pick the periodogram peak on a 0.25 Hz grid. Returns
    (f_off_hz, start, metric). Nothing in this is specific to a site; a static channel returns about zero.
    """
    fs_bb = scheme.symbol_rate * SPS
    t = np.arange(y.size) / fs_bb
    best = (-1.0, 0.0, 0)
    for f in DOPPLER_COARSE_HZ:
        st, met = synchronise(y * np.exp(-2j * np.pi * f * t), max_delay_s, scheme)
        if met > best[0]:
            best = (met, float(f), st)
    met, f_coarse, start = best
    z = y[start: start + PREAMBLE_SYMBOLS * SPS: SPS][:PREAMBLE_SYMBOLS] * np.conj(PREAMBLE[: min(PREAMBLE_SYMBOLS, (y.size - start + SPS - 1) // SPS)])
    n = np.arange(z.size) / scheme.symbol_rate
    grid = np.arange(f_coarse - 6.0, f_coarse + 6.0 + 1e-9, DOPPLER_FINE_HZ)
    grid = grid[np.abs(grid) <= DOPPLER_MAX_HZ]
    per = np.abs(np.exp(-2j * np.pi * np.outer(grid, n)) @ z)
    return float(grid[int(np.argmax(per))]), int(start), float(met)


def compensate_doppler(y: np.ndarray, f_off: float, fc: float, scheme: Scheme) -> np.ndarray:
    """Remove the carrier offset and the matching time-scale (wideband Doppler: a = f_off / fc)."""
    fs_bb = scheme.symbol_rate * SPS
    t = np.arange(y.size) / fs_bb
    y1 = y * np.exp(-2j * np.pi * f_off * t)
    a = f_off / fc
    if abs(a) < 1e-7:
        return y1
    # The received signal is x(t (1 + a)); resample at t / (1 + a) to undo it.
    src = t / (1.0 + a)
    return np.interp(src, t, y1.real) + 1j * np.interp(src, t, y1.imag)


def demodulate(x: np.ndarray, fc: float, fs: float, scheme: Scheme, fb_symbols: int, max_delay_s: float = 0.30,
               doppler: bool = True):
    """Return (payload bits, start sample at the symbol rate, sync metric, output SNR dB, data error power,
    estimated carrier offset in Hz)."""
    y = to_baseband(x, fc, fs, scheme)
    f_off = 0.0
    if doppler:
        f_off, _, _ = estimate_doppler(y, fc, max_delay_s, scheme)
        y = compensate_doppler(y, f_off, fc, scheme)
    start, metric = synchronise(y, max_delay_s, scheme)
    z, snr_out, err = dfe(y, start, scheme, fb_symbols)
    soft = soft_bits(z, scheme.mod)
    bits = viterbi_decode(soft, PAYLOAD_BITS + TAIL_BITS)
    return bits[:PAYLOAD_BITS], int(start), float(metric), snr_out, err, f_off


def awgn_selftest(scheme: Scheme, snr_db_list=(-4, -2, 0, 2, 4, 6), n_frames=20, fc=13000.0, fs=96000.0,
                  seed=1, freq_offset_hz: float = 0.0, fb_symbols: int = 8, time_scale: float = 0.0, doppler: bool = True):
    """Packet success in AWGN, SNR = signal power over noise power inside the (1+alpha)R_s band.

    `freq_offset_hz` applies a pure carrier offset (transmit at fc + offset); `time_scale` applies wideband Doppler
    by resampling the passband signal to x(t (1 + a)), which shifts the carrier by a fc and compresses the symbols.
    """
    rng = np.random.default_rng(seed)
    res = {}
    for snr_db in snr_db_list:
        ok, snrs, offs = 0, [], []
        for _ in range(n_frames):
            payload = rng.integers(0, 2, PAYLOAD_BITS)
            x = modulate(frame_symbols(payload, scheme), fc + freq_offset_hz, fs, scheme)
            if time_scale:
                t_src = np.arange(int(x.size / (1 + time_scale))) * (1 + time_scale)
                x = np.interp(t_src, np.arange(x.size), x)
            delay = rng.integers(0, int(0.1 * fs))
            x = np.concatenate([np.zeros(delay), x, np.zeros(int(0.05 * fs))])
            noise_psd = 1.0 / (10 ** (snr_db / 10) * scheme.band_hz)
            w = rng.standard_normal(x.size) * np.sqrt(noise_psd * fs / 2)
            bits, _, _, snr_out, _, f_off = demodulate(x + w, fc, fs, scheme, fb_symbols, doppler=doppler)
            ok += int(np.array_equal(bits, payload))
            snrs.append(snr_out)
            offs.append(f_off)
        res[snr_db] = ok / n_frames
        print(f"AWGN {scheme} SNR {snr_db:+5.1f} dB in-band: PSR {res[snr_db]:.3f}  out-SNR {np.mean(snrs):5.1f} dB  "
              f"f_off est {np.mean(offs):+6.2f} Hz (true {freq_offset_hz + time_scale * fc:+6.2f})", flush=True)
    return res


if __name__ == "__main__":
    import time

    for rs in (1000, 2000, 4000):
        for mod in ("bpsk", "qpsk"):
            s = Scheme(rs, mod)
            t0 = time.time()
            print(s, "frame", round(s.frame_s, 3), "s; payload bit/s", round(s.bits_per_s))
            awgn_selftest(s, snr_db_list=(-2, 0, 2, 4, 6, 8), n_frames=10)
            print("  ", round((time.time() - t0) / 60, 3), "s per frame", flush=True)
